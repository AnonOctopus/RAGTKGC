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


class Retriever:
    def __init__(self,
                 test, all_facts,
                 entities, relations, times_id,
                 num_relations, chains, rel_keys, dataset,
                 retrieve_type='TLogic', rule_length_all=False,
                 inverse_body_object_match=False, early_stop_at_num_facts=False,
                 use_ids=False,
                 # --- new parameters ---
                 confidence_threshold=None,
                 model_type="t5",
                 top_k_rules=None):
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
        self.use_ids = use_ids
        self.confidence_threshold = confidence_threshold
        self.model_type = model_type
        self.top_k_rules = top_k_rules

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
        
    # ------------------------------------------------------------------
    # History text helpers
    # ------------------------------------------------------------------

    def _time_period(self):
        return 24 if self.dataset in ("icews14", "icews18") else 1

    def _fact_to_line(self, fi):
        """Format a single fact (by index into all_facts) as a history text line."""
        fact = self.all_facts[fi].strip().split('\t')
        period = self._time_period()
        time_in_id = self.times_id[fact[3]]
        if self.use_ids:
            sub_out = str(self.entities[fact[0]])
            rel_out = str(self.relations[fact[1]])
            obj_out = str(self.entities[fact[2]])
        else:
            sub_out, rel_out, obj_out = fact[0], fact[1], fact[2]
        return f"{int(time_in_id) / period}: [{sub_out}, {rel_out}, {obj_out}] \n"

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
            num_facts = 50
            time, sub, rel, idx = self.prepare_bs(i)

            stats_initial.append(len(idx))
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
    
    def tlogic_prepro(self, i):
        test_sub, test_rel, _, test_time, _ = self.test[i].strip().split("\t")
        idx_test = len(self.all_facts) - (len(self.test) - 1) + i - 1
        idx_t = np.where(self.col_time < test_time)[0]
        s_t = set(idx_t)
        if idx_test in s_t:
            s_t.remove(idx_test)
        idx_test_sub = np.where(self.col_sub == test_sub)[0]
        s_test_sub = set(idx_test_sub)
        s_0 = s_t & s_test_sub
        head_rel = self.relations[test_rel]
        time = self.times_id[test_time]
        return s_0, s_t, head_rel, time, test_sub, test_rel

    def build_tl(self):
        """Rule-guided retrieval (TLogic / RAGTKGC style) with optional top-k and threshold split."""
        test_text = []
        test_idx = []
        test_rule_ids = []

        # Statistics accumulators — populated per example
        stats_initial = []
        stats_after_topk = [] if self.top_k_rules is not None else None
        stats_after_numfacts = []
        stats_above_thresh = [] if self.confidence_threshold is not None else None
        stats_below_thresh = [] if self.confidence_threshold is not None else None

        for i in tqdm(range(len(self.test))):
            num_facts = 50
            s_0, s_t, head_rel, time, test_sub, test_rel = self.tlogic_prepro(i)

            if str(head_rel) not in self.chains:
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

            # Anchor arrays for subject and (optionally) object matching
            s_0_sub = np.array(list(s_0))
            s_0_obj = None
            if self.inverse_body_object_match:
                idx_test_obj = np.where(self.col_obj == test_sub)[0]
                s_test_obj = set(idx_test_obj)
                s_0_obj = np.array(list(s_t & s_test_obj))

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
                body_rel_last = chain_rule['body_rels'][-1]
                rel = body_rel_last % self.num_relations
                idx_rel = np.where(self.col_rel == self.rel_keys[rel])[0]

                if self.inverse_body_object_match and body_rel_last >= self.num_relations:
                    idx_anchor = s_0_obj if s_0_obj is not None else s_0_sub
                else:
                    idx_anchor = s_0_sub

                idx_case = np.intersect1d(idx_rel, idx_anchor)
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
            # Apply top_k_rules: keep only the k highest-confidence fired rules
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
            idx = sorted(fact_to_conf.keys(), reverse=True)[:num_facts]
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

        logger.info("History size statistics (build_tl):\n%s", json.dumps(self.build_stats, indent=2))
        return test_idx, test_text, test_rule_ids

    def collect_hist(self, i, facts, num_facts):
        """Legacy helper used by build_bs; builds history lines from fact strings."""
        period = self._time_period()
        histories = []
        facts = list(reversed(facts[:num_facts]))  # oldest-first
        for fact_str in facts:
            fact = fact_str.strip().split('\t')
            time_in_id = self.times_id[fact[3]]
            if self.use_ids:
                sub_out = str(self.entities[fact[0]])
                rel_out = str(self.relations[fact[1]])
                obj_out = str(self.entities[fact[2]])
            else:
                sub_out, rel_out, obj_out = fact[0], fact[1], fact[2]
            histories.append(
                f"{int(time_in_id) / period}: [{sub_out}, {rel_out}, {obj_out}] \n"
            )
        return histories

    def build_history_query(self, time, test_sub, test_rel, histories=''):
        period = self._time_period()
        if self.use_ids:
            sub_out = str(self.entities[test_sub])
            rel_out = str(self.relations[test_rel])
        else:
            sub_out = test_sub
            rel_out = test_rel
        return [''.join(histories) + f"{int(time) / period}: [{sub_out}, {rel_out},\n"]

    def call_function(self, func_name):
        func = getattr(self, func_name)
        if func and callable(func):
            return func()
        logger.error("Retrieve function not found: %s", func_name)
        raise ValueError(f"Retrieve function not found: {func_name}")

    def get_output(self):
        type_retr = "bs" if self.retrieve_type == 'bs' else "tl"
        return self.call_function("build_" + type_retr)

