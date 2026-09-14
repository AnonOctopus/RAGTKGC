import json
import logging

import numpy as np
from basic import flip_dict
import time as ti
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _aggregate_sizes(sizes):
    """Return min/max/avg statistics for a list of integer history sizes."""
    if not sizes:
        return {"count": 0, "min": None, "max": None, "avg": None}
    return {
        "count": len(sizes),
        "min": int(min(sizes)),
        "max": int(max(sizes)),
        "avg": round(sum(sizes) / len(sizes), 4),
    }


def time_period(dataset):
    """Divisor applied to a timestamp id when rendering a fact's time.

    Args:
        dataset: dataset name.

    Returns:
        int: the divisor.
    """
    return 24 if dataset in ("icews14", "icews18") else 1


def fact_to_line(fact_row, times_id, period, entities, index_target=False):
    """Render one fact from all_facts.txt as a history text line.

    Args:
        fact_row: one tab-separated all_facts.txt line — subject, relation,
            object, timestamp — holding names rather than ids.
        times_id: timestamp string to timestamp id.
        period: divisor applied to the timestamp id, from time_period.
        entities: entity name to id. Read only when index_target is set.
        index_target: prefix the object with its entity id, as "18.Thailand".
            The subject and relation stay bare names even so: only the position
            the model has to generate is indexed.

    Returns:
        str: the history line, newline-terminated.

    Raises:
        KeyError: the timestamp is absent from times_id, or the object is
            absent from entities while index_target is set.
    """
    fact = fact_row.strip().split('\t')
    time_in_id = times_id[fact[3]]
    sub_out, rel_out, obj_out = fact[0], fact[1], fact[2]
    if index_target:
        obj_out = f"{entities[obj_out]}.{obj_out}"
    # Floor division, not true division: the day index is integral, and a
    # trailing ".0" is a divergence from the format the baseline publishes.
    return f"{int(time_in_id) // period}: [{sub_out}, {rel_out}, {obj_out}] \n"


class Retriever:
    def __init__(self,
                 test, all_facts,
                 entities, relations, times_id,
                 num_relations, chains, rel_keys, dataset,
                 retrieve_type='TLogic', rule_length_all=False,
                 inverse_body_object_match=False, early_stop_at_num_facts=False,
                 index_target=False,
                 # --- new parameters ---
                 confidence_threshold=None,
                 model_type="t5",
                 top_k_rules=None,
                 save_metadata=False,
                 num_facts=None):
        """
        Parameters
        ----------
        confidence_threshold : float or None
            If set, split retrieved facts into two groups:
              - above (conf >= threshold): general, high-confidence rules
              - below (conf <  threshold): specific, low-confidence rules
            Each group is kept sorted by recency.
        model_type : {"t5", "llm"}
            How to format the split history in the output text.
            "t5"  → [specific_facts] + [general_facts] (general closer to query)
            "llm" → labelled sections for each group
        top_k_rules : int or None
            After collecting all fired rules, keep only the top-k by confidence
            before applying num_facts trimming.
        save_metadata : bool
            If set, build_tl streams one JSONL line per sample — the fired rules
            with their confidences and fact indices — to metadata_path, so
            filters can be re-applied later without retrieving again. The header
            line records retrieval-time state only; rendering options are not
            part of it, because the stored fact indices carry no format.
        """
        self.retrieve_type = retrieve_type
        self.dataset = dataset
        self.test = test
        self.all_facts = all_facts

        self.entities = entities
        self.relations = relations
        self.times_id = times_id
        self.num_relations = num_relations
        self.chains = chains
        self.rule_length_all = rule_length_all
        self.inverse_body_object_match = inverse_body_object_match
        self.early_stop_at_num_facts = early_stop_at_num_facts
        self.index_target = index_target
        self.confidence_threshold = confidence_threshold
        self.model_type = model_type
        self.top_k_rules = top_k_rules
        self.num_facts = num_facts
        self.save_metadata = save_metadata

        # Retrieval for rules longer than one body relation is not implemented:
        # build_tl matches only body_rels[-1] and anchors it at the query subject
        # instead of at the intermediate entity, so the facts it returns are not
        # instances of the rule. Reject such a bank up front rather than returning
        # plausible-looking wrong facts. Checked once over the bank, not per sample.
        if rule_length_all:
            _long = sum(
                1
                for _rel_rules in chains.values()
                for _rule in _rel_rules
                if len(_rule.get("body_rels", [])) > 1
            )
            if _long:
                raise NotImplementedError(
                    f"The rule bank contains {_long} rules with more than one body "
                    "relation, and multi-hop retrieval is not implemented. Pass "
                    "--length_1_only to restrict retrieval to length-1 rules, or "
                    "implement matching that anchors each hop at the correct entity."
                )

        # early_stop_at_num_facts compares a running fact count against num_facts,
        # so without a cap the comparison would raise TypeError partway through the
        # first sample. Fail here instead, with a message that says what to pass.
        if early_stop_at_num_facts and num_facts is None:
            raise ValueError(
                "early_stop_at_num_facts requires num_facts to be set — it stops "
                "consulting rules once that many facts have been collected, and "
                "there is nothing to compare against otherwise. Pass --num_facts N."
            )

        if early_stop_at_num_facts and top_k_rules is not None:
            logger.warning(
                "early_stop_at_num_facts is set but top_k_rules=%d is also active. "
                "Top K rules requires seeing all rules before ranking, so "
                "early_stop_at_num_facts will have no effect.",
                top_k_rules,
            )

        # Where build_tl streams per-sample metadata (JSONL) when save_metadata=True.
        # Set by the caller (retrieve.py) before get_output(); extra header fields
        # known only to the caller (split, mining, base_answers_path) go in
        # metadata_extra_header and are merged into the header line.
        self.metadata_path = None
        self.metadata_extra_header = {}

        self.build_stats = {}  # populated after build_tl / build_bs runs

        self.entities_flip = flip_dict(self.entities)
        self.relations_flip = flip_dict(self.relations)
        col_sub = []
        col_rel = []
        col_obj = []
        col_time = []
        for row in all_facts:
            row = row.strip().split('\t')
            col_sub.append(row[0])
            col_rel.append(row[1])
            col_obj.append(row[2])
            col_time.append(row[3])
        self.col_obj = np.array(col_obj)
        self.col_sub = np.array(col_sub)
        self.col_time = np.array(col_time)
        self.col_rel = np.array(col_rel)
        self.rel_keys = np.array(rel_keys)

        # Row indices by relation / subject / object, filled on first use. Each
        # lookup they replace is a full scan of a fixed-width unicode column,
        # and build_tl performs one per rule per sample. Every fact falls in
        # exactly one bucket per column, so a fully populated cache holds one
        # index per fact however many distinct values exist.
        self._rel_idx = {}
        self._sub_idx = {}
        self._obj_idx = {}
        
    # ------------------------------------------------------------------
    # History text helpers
    # ------------------------------------------------------------------

    def _time_period(self):
        return time_period(self.dataset)

    def _fact_to_line(self, fi):
        """Format a single fact (by index into all_facts) as a history text line."""
        return fact_to_line(self.all_facts[fi], self.times_id,
                            self._time_period(), self.entities,
                            self.index_target)

    def _indices_to_history_lines(self, indices_newest_first):
        """Return history text lines ordered oldest-first from indices sorted newest-first."""
        return [self._fact_to_line(fi) for fi in reversed(indices_newest_first)]

    def _build_split_history(self, above_idx, below_idx):
        """
        Combine the two confidence-split groups into a single history line list.

        above_idx : fact indices with conf >= threshold (general/high-confidence rules)
        below_idx : fact indices with conf <  threshold (specific/low-confidence rules)
        Both lists are sorted newest-first; history lines are produced oldest-first.

        t5  model : [specific_lines] + [general_lines]  — general closer to the query
        llm model : labelled sections per group
        """
        above_lines = self._indices_to_history_lines(above_idx)
        below_lines = self._indices_to_history_lines(below_idx)

        if self.confidence_threshold is None or not below_idx:
            # No split — return the single group as-is
            return above_lines

        if self.model_type == "t5":
            # Specific facts first so that general (high-conf) facts sit
            # immediately before the query line
            return below_lines + above_lines
        else:
            # LLM: add a short label before each section
            combined = []
            if below_lines:
                combined.append("[Specific context - lower confidence rules:]\n")
                combined.extend(below_lines)
            if above_lines:
                combined.append("[General context - higher confidence rules:]\n")
                combined.extend(above_lines)
            return combined

    # ------------------------------------------------------------------
    # Build methods
    # ------------------------------------------------------------------

    def prepare_bs(self, i):
        sub, rel, _, time, _ = self.test[i].strip().split("\t")
        idx_t = np.where(self.col_time < time)[0] #cannot be equal to
        s_t = set(idx_t)
        idx0 = np.where(self.col_sub == sub)[0]
        s0 = set(idx0)
        idx = list(s0 & s_t)
        idx.sort(reverse=True)
        time = self.times_id[time]
        return time, sub, rel, idx
    
    def build_bs(self):
        """Pure entity-based retrieval (no rules); used as a baseline."""
        test_text = []
        test_idx = []
        test_rule_ids = []
        stats_initial = []
        stats_after_numfacts = []

        for i in tqdm(range(0, len(self.test))):
            num_facts = self.num_facts
            time, sub, rel, idx = self.prepare_bs(i)

            stats_initial.append(len(idx))
            if num_facts is not None:
                idx = idx[:num_facts]
            stats_after_numfacts.append(len(idx))

            facts = [self.all_facts[k] for k in idx]
            histories = self.collect_hist(i, facts, len(facts))
            history_query = self.build_history_query(time, sub, rel, histories=histories)

            test_idx.append(idx)
            test_text.append(history_query)
            test_rule_ids.append([])

        self.build_stats = {
            "initial": _aggregate_sizes(stats_initial),
            "after_num_facts": _aggregate_sizes(stats_after_numfacts),
        }
        logger.info("History size statistics (build_bs):\n%s", json.dumps(self.build_stats, indent=2))
        return test_idx, test_text, test_rule_ids
    
    def _column_indices(self, cache, column, value):
        """Row indices in all_facts where a column equals a value, memoised.

        Args:
            cache: dict memoising results for this column.
            column: the fact column to compare against.
            value: the value to match.

        Returns:
            np.ndarray: sorted unique row indices; empty when the value never
                occurs in that column.
        """
        # `is None`, not truthiness: a cached empty array is falsy and would be
        # recomputed on every lookup.
        cached = cache.get(value)
        if cached is None:
            cached = np.where(column == value)[0]
            cache[value] = cached
        return cached

    def tlogic_prepro(self, i):
        """Split one test quad into its history anchor and query fields.

        Args:
            i: index into self.test.

        Returns:
            tuple: (anchor, cut, head_rel, time, test_sub, test_rel), where
                anchor holds the all_facts rows whose subject is the query
                subject and whose timestamp is strictly earlier, and cut is the
                number of rows strictly earlier than the query timestamp.
        """
        test_sub, test_rel, _, test_time, _ = self.test[i].strip().split("\t")
        # No guard against the target quadruple leaking into its own history: the
        # filter below is strictly `<`, and the target shares its own timestamp, so
        # it can never enter the prefix. (The inherited guard here recomputed the
        # target's row index with a formula that was only correct for the test split,
        # and measured zero removals across all splits of both datasets.) If this
        # filter is ever relaxed to `<=`, a guard becomes necessary again — index it
        # by the split's offset into all_facts, not by arithmetic from the end.
        #
        # all_facts is non-decreasing in the time column, verified for every
        # distinct timestamp of ICEWS14 and ICEWS18, so the rows strictly
        # earlier than test_time are exactly [0, cut). Materialising them as a
        # set cost more than everything else in this method combined.
        cut = int(np.searchsorted(self.col_time, test_time, side="left"))
        idx_sub = self._column_indices(self._sub_idx, self.col_sub, test_sub)
        s_0 = idx_sub[idx_sub < cut]
        head_rel = self.relations[test_rel]
        time = self.times_id[test_time]
        return s_0, cut, head_rel, time, test_sub, test_rel

    def build_tl(self):
        """Rule-guided retrieval (TLogic / RAGTKGC style) with optional top-k and threshold split."""
        test_text = []
        test_idx = []
        test_rule_ids = []

        if self.save_metadata:
            stats_metadata_full = []  # unfiltered fact count per sample (proves no num_facts cap in metadata)
            # Stream metadata to a JSONL file: line 1 is the header, every
            # subsequent line is one sample. Keeps peak RAM flat at ~one sample
            # instead of materialising the entire (potentially multi-GB) list.
            if not self.metadata_path:
                raise ValueError("save_metadata=True but metadata_path was not set on the Retriever.")
            _meta_fh = open(self.metadata_path, "w", encoding="utf-8")
            # Retrieval-time state only. index_target and model_type decide how a
            # fact is rendered, and the metadata stores fact indices rather than
            # rendered text, so recording them here would describe a choice this
            # file does not embody — and apply_history_filters.py would inherit a
            # format the caller never asked for.
            _meta_header = {
                "dataset": self.dataset,
                "inverse_body_object_match": self.inverse_body_object_match,
            }
            _meta_header.update(self.metadata_extra_header or {})
            _meta_fh.write(json.dumps(_meta_header) + "\n")

        # Statistics accumulators — populated per example
        stats_initial = []
        stats_after_topk = [] if self.top_k_rules is not None else None
        stats_after_numfacts = []
        stats_above_thresh = [] if self.confidence_threshold is not None else None
        stats_below_thresh = [] if self.confidence_threshold is not None else None

        for i in tqdm(range(len(self.test))):
            num_facts = self.num_facts
            s_0, cut, head_rel, time, test_sub, test_rel = self.tlogic_prepro(i)

            # Query-only text line (history prefix empty); stored in metadata.
            query_line = self.build_history_query(time, test_sub, test_rel)[0]

            if str(head_rel) not in self.chains:
                if self.save_metadata:
                    _meta_fh.write(json.dumps({
                        "sample_idx": i,
                        "query_line": query_line,
                        "fired_rules": [],
                    }) + "\n")
                    stats_metadata_full.append(0)
                history_query = self.build_history_query(time, test_sub, test_rel)
                test_idx.append([])
                test_text.append(history_query)
                test_rule_ids.append([])
                stats_initial.append(0)
                stats_after_numfacts.append(0)
                if stats_after_topk is not None:
                    stats_after_topk.append(0)
                if stats_above_thresh is not None:
                    stats_above_thresh.append(0)
                    stats_below_thresh.append(0)
                continue

            # Anchor arrays for subject and (optionally) object matching. Both
            # are sorted index arrays restricted to the pre-query prefix.
            s_0_sub = s_0
            s_0_obj = None
            if self.inverse_body_object_match:
                idx_obj = self._column_indices(self._obj_idx, self.col_obj, test_sub)
                s_0_obj = idx_obj[idx_obj < cut]

            # Select rule indices to evaluate
            idx_chain = [
                k for k in range(len(self.chains[str(head_rel)]))
                if self.rule_length_all
                or len(self.chains[str(head_rel)][k]['body_rels']) == 1
            ]

            # ------------------------------------------------------------------
            # Collect all fired rules
            # Each entry: (confidence, rule_id, frozenset of matched fact indices)
            # Chains are already sorted by confidence descending; we sort again
            # after filtering to be safe when top_k_rules is used.
            # ------------------------------------------------------------------
            fired_rules = []          # (conf, rule_id, fact_set)
            cumulative_facts = set()  # used only for the early-stop heuristic

            for k in idx_chain:
                chain_rule = self.chains[str(head_rel)][k]
                rule_id = chain_rule.get('rule_id')
                conf = float(chain_rule.get('conf', 0.0))
                # Only the LAST body relation is consulted, anchored at the query
                # subject. Correct for length-1 rules; wrong for longer ones, where
                # b2 in head(s,o) <- b1(s,x), b2(x,o) starts at x, not at s. The
                # constructor rejects banks containing longer rules, so reaching
                # this line with len(body_rels) > 1 is impossible.
                body_rel_last = chain_rule['body_rels'][-1]
                rel = body_rel_last % self.num_relations
                idx_rel = self._column_indices(self._rel_idx, self.col_rel, self.rel_keys[rel])

                if self.inverse_body_object_match and body_rel_last >= self.num_relations:
                    idx_anchor = s_0_obj if s_0_obj is not None else s_0_sub
                else:
                    idx_anchor = s_0_sub

                # assume_unique: both inputs are index arrays without repeats
                # (np.where output, and an array built from a set), so the
                # internal unique() and sort() calls are wasted work. The
                # result is sorted either way.
                idx_case = np.intersect1d(idx_rel, idx_anchor, assume_unique=True)
                if idx_case.size == 0:
                    continue

                fact_set = set(idx_case.tolist())
                fired_rules.append((conf, rule_id, fact_set))
                cumulative_facts.update(fact_set)

                # Early stop only when top_k_rules is inactive — otherwise we
                # need all rules before we can rank them by confidence.
                if self.early_stop_at_num_facts and self.top_k_rules is None:
                    if len(cumulative_facts) >= num_facts:
                        break

            # Initial size = union of all fired-rule facts (before any filtering)
            all_initial = set().union(*(fs for _, _, fs in fired_rules)) if fired_rules else set()
            stats_initial.append(len(all_initial))

            # ------------------------------------------------------------------
            # Metadata capture: full pre-filter state (all rules, no num_facts cap)
            # Must happen BEFORE top_k_rules so the metadata always reflects
            # the complete retrieval result.
            # ------------------------------------------------------------------
            if self.save_metadata:
                # Only the fired rules are stored. The fact list, each fact's
                # effective confidence, the rules that retrieved it and its
                # rendered text all follow from fact_indices, and
                # apply_history_filters derives them.
                #
                # Rules stay in the order retrieval produced them, not sorted by
                # confidence: the rule-id order of the output follows this order
                # whenever top_k_rules is unset, and a stable sort in the
                # re-filter path recovers the sorted slice when it is set.
                _meta_fh.write(json.dumps({
                    "sample_idx": i,
                    "query_line": query_line,
                    "fired_rules": [
                        {"conf": _c, "rule_id": _r, "fact_indices": sorted(_fs)}
                        for _c, _r, _fs in fired_rules
                    ],
                }) + "\n")
                stats_metadata_full.append(len(all_initial))  # no cap — full unfiltered count

            # ------------------------------------------------------------------
            # Apply top_k_rules: keep only the k highest-confidence fired rules
            #
            # The filter order from here on — top_k_rules, then union the
            # surviving rules' facts, then sort newest-first, then cap at
            # num_facts — is duplicated in
            # apply_history_filters._apply_filters, which re-derives the same
            # history from saved metadata. Changing the order here without
            # changing it there makes the two paths produce different histories
            # from identical inputs.
            # ------------------------------------------------------------------
            if self.top_k_rules is not None:
                fired_rules = sorted(fired_rules, key=lambda x: x[0], reverse=True)[:self.top_k_rules]
                all_topk = set().union(*(fs for _, _, fs in fired_rules)) if fired_rules else set()
                stats_after_topk.append(len(all_topk))

            if not fired_rules:
                history_query = self.build_history_query(time, test_sub, test_rel)
                test_idx.append([])
                test_text.append(history_query)
                test_rule_ids.append([])
                stats_after_numfacts.append(0)
                if stats_above_thresh is not None:
                    stats_above_thresh.append(0)
                    stats_below_thresh.append(0)
                continue

            # ------------------------------------------------------------------
            # Build per-fact mappings
            # fact_to_conf    : effective confidence = max over all rules that
            #                   retrieved this fact (best-case interpretation)
            # fact_to_rule_ids: ordered list of rule_ids that retrieved this fact
            # ------------------------------------------------------------------
            fact_to_conf = {}
            fact_to_rule_ids = {}
            for conf, rule_id, fact_set in fired_rules:
                for fi in fact_set:
                    if fi not in fact_to_conf or conf > fact_to_conf[fi]:
                        fact_to_conf[fi] = conf
                    if rule_id is not None:
                        fact_to_rule_ids.setdefault(fi, []).append(rule_id)

            # ------------------------------------------------------------------
            # Sort by fact index descending (≈ most recent first) and trim
            # ------------------------------------------------------------------
            idx = sorted(fact_to_conf.keys(), reverse=True)
            if num_facts is not None:
                idx = idx[:num_facts]
            stats_after_numfacts.append(len(idx))

            # ------------------------------------------------------------------
            # Filter rule_ids: only include rules whose facts survived the trim
            # ------------------------------------------------------------------
            seen_rule_ids = set()
            surviving_rule_ids = []
            for fi in idx:
                for rid in fact_to_rule_ids.get(fi, []):
                    if rid not in seen_rule_ids:
                        surviving_rule_ids.append(rid)
                        seen_rule_ids.add(rid)

            # ------------------------------------------------------------------
            # Confidence threshold split
            # ------------------------------------------------------------------
            if self.confidence_threshold is not None:
                above_idx = [fi for fi in idx if fact_to_conf[fi] >= self.confidence_threshold]
                below_idx = [fi for fi in idx if fact_to_conf[fi] < self.confidence_threshold]
                stats_above_thresh.append(len(above_idx))
                stats_below_thresh.append(len(below_idx))
            else:
                above_idx = idx
                below_idx = []

            # ------------------------------------------------------------------
            # Build history text and store results
            # ------------------------------------------------------------------
            histories = self._build_split_history(above_idx, below_idx)
            history_query = self.build_history_query(time, test_sub, test_rel, histories=histories)

            test_idx.append(idx)
            test_text.append(history_query)
            test_rule_ids.append(surviving_rule_ids)
            ti.sleep(0.001)

        if self.save_metadata:
            _meta_fh.close()

        # Store and log accumulated statistics
        self.build_stats = {
            "initial": _aggregate_sizes(stats_initial),
            "after_num_facts": _aggregate_sizes(stats_after_numfacts),
        }
        if stats_after_topk is not None:
            self.build_stats["after_top_k_rules"] = _aggregate_sizes(stats_after_topk)
        if stats_above_thresh is not None:
            self.build_stats["above_threshold"] = _aggregate_sizes(stats_above_thresh)
            self.build_stats["below_threshold"] = _aggregate_sizes(stats_below_thresh)
        if self.save_metadata:
            self.build_stats["metadata_full_facts"] = _aggregate_sizes(stats_metadata_full)

        logger.info("History size statistics (build_tl):\n%s", json.dumps(self.build_stats, indent=2))
        if self.save_metadata:
            logger.info(
                "METADATA NOTE: 'metadata_full_facts' shows fact counts BEFORE num_facts cap — "
                "these are the counts stored in the metadata JSON. "
                "'after_num_facts' shows what the regular output files contain."
            )

        return test_idx, test_text, test_rule_ids

    def collect_hist(self, i, facts, num_facts):
        """Build history lines from fact strings, oldest first.

        Args:
            i: index of the query being built. Accepted for call symmetry with
                the other history builders.
            facts: fact rows, newest first.
            num_facts: how many of them to keep.

        Returns:
            list[str]: history lines, oldest first.
        """
        period = self._time_period()
        return [
            fact_to_line(fact_str, self.times_id, period, self.entities,
                         self.index_target)
            for fact_str in reversed(facts[:num_facts])
        ]

    def build_history_query(self, time, test_sub, test_rel, histories=''):
        """Append the query line to a rendered history.

        Args:
            time: timestamp id of the query.
            test_sub: query subject name.
            test_rel: query relation name.
            histories: already-rendered history lines.

        Returns:
            list[str]: a single-element list holding the full prompt.
        """
        period = self._time_period()
        # The query has no object, so index_target does not apply here; the
        # subject and relation are bare names in both variants.
        return [''.join(histories)
                + f"{int(time) // period}: [{test_sub}, {test_rel},\n"]

    def call_function(self, func_name):
        func = getattr(self, func_name)
        if func and callable(func):
            return func()
        logger.error("Retrieve function not found: %s", func_name)
        raise ValueError(f"Retrieve function not found: {func_name}")

    def get_output(self):
        type_retr = "bs" if self.retrieve_type == 'bs' else "tl"
        return self.call_function("build_" + type_retr)

