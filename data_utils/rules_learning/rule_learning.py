import os
import json
import itertools
import hashlib
import numpy as np
from collections import Counter


def canonical_rule_payload(rule):
    return {
        "head_rel": int(rule["head_rel"]),
        "body_rels": [int(x) for x in rule["body_rels"]],
        "var_constraints": [list(map(int, x)) for x in rule["var_constraints"]],
    }


def make_rule_id(rule):
    payload = canonical_rule_payload(rule)
    payload_str = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload_str.encode("utf-8")).hexdigest()


def update_common_rule_pool_from_rules_file(rules_file_path, mining_alg, rule_pool_file_path=None, write_rule_ids_back=True):
    """
    Import an already computed rules JSON file and merge it into the common rule pool.

    Parameters:
        rules_file_path (str): path to per-algorithm rules JSON
        mining_alg (str): mining algorithm label to register in found_by/algorithm_stats
        rule_pool_file_path (str|None): optional explicit path for common_rule_pool.json
        write_rule_ids_back (bool): if True, persist generated rule_id/found_by back into rules file

    Returns:
        dict: summary with counts and output paths
    """

    with open(rules_file_path, "r", encoding="utf-8") as fin:
        rules_dict = json.load(fin)

    if rule_pool_file_path is None:
        rule_pool_file_path = os.path.join(os.path.dirname(rules_file_path), "common_rule_pool.json")

    if os.path.exists(rule_pool_file_path):
        with open(rule_pool_file_path, "r", encoding="utf-8") as fin:
            rule_pool = json.load(fin)
    else:
        rule_pool = {}

    total_rules = 0
    new_rules = 0
    updated_rules = 0

    for rel_key in list(rules_dict.keys()):
        rel_rules = rules_dict[rel_key]
        for rule in rel_rules:
            total_rules += 1
            rule_id = rule.get("rule_id") or make_rule_id(rule)
            rule["rule_id"] = rule_id
            rule["found_by"] = sorted(set(rule.get("found_by", []) + [mining_alg]))

            canonical = canonical_rule_payload(rule)
            if rule_id not in rule_pool:
                new_rules += 1
                rule_pool[rule_id] = {
                    "rule_id": rule_id,
                    "head_rel": canonical["head_rel"],
                    "body_rels": canonical["body_rels"],
                    "var_constraints": canonical["var_constraints"],
                    "found_by": [],
                    "algorithm_stats": {},
                }
            else:
                updated_rules += 1

            found_by = set(rule_pool[rule_id].get("found_by", []))
            found_by.add(mining_alg)
            rule_pool[rule_id]["found_by"] = sorted(found_by)
            rule_pool[rule_id]["algorithm_stats"][mining_alg] = {
                "conf": rule.get("conf"),
                "rule_supp": rule.get("rule_supp"),
                "body_supp": rule.get("body_supp"),
            }

    with open(rule_pool_file_path, "w", encoding="utf-8") as fout:
        json.dump(rule_pool, fout, indent=2)

    if write_rule_ids_back:
        with open(rules_file_path, "w", encoding="utf-8") as fout:
            json.dump(rules_dict, fout)

    return {
        "rules_file_path": rules_file_path,
        "rule_pool_file_path": rule_pool_file_path,
        "total_rules_seen": total_rules,
        "new_rules_added": new_rules,
        "existing_rules_updated": updated_rules,
    }


def add_rule_ids_from_common_pool(rules_file_path, rule_pool_file_path=None):
    """
    Update an already computed rules JSON file by adding rule_id from common_rule_pool.json.

    This helper does not modify the common pool; it only enriches the rules file.

    Parameters:
        rules_file_path (str): path to per-algorithm rules JSON
        rule_pool_file_path (str|None): optional explicit path for common_rule_pool.json

    Returns:
        dict: summary with counts and paths
    """

    if rule_pool_file_path is None:
        rule_pool_file_path = os.path.join(os.path.dirname(rules_file_path), "common_rule_pool.json")

    if not os.path.exists(rule_pool_file_path):
        raise FileNotFoundError(f"Common rule pool not found: {rule_pool_file_path}")

    with open(rules_file_path, "r", encoding="utf-8") as fin:
        rules_dict = json.load(fin)
    with open(rule_pool_file_path, "r", encoding="utf-8") as fin:
        rule_pool = json.load(fin)

    # Build canonical lookup for robust matching even if the key format changes.
    canonical_to_id = {}
    for pool_rule_id, pool_rule in rule_pool.items():
        canonical = canonical_rule_payload(pool_rule)
        canonical_key = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        canonical_to_id[canonical_key] = pool_rule_id

    total_rules = 0
    ids_added = 0
    ids_updated = 0
    ids_missing = 0

    for rel_key in list(rules_dict.keys()):
        rel_rules = rules_dict[rel_key]
        for rule in rel_rules:
            total_rules += 1

            computed_id = make_rule_id(rule)
            canonical_key = json.dumps(
                canonical_rule_payload(rule), sort_keys=True, separators=(",", ":")
            )

            matched_id = computed_id if computed_id in rule_pool else canonical_to_id.get(canonical_key)

            if matched_id is None:
                ids_missing += 1
                continue

            existing = rule.get("rule_id")
            rule["rule_id"] = matched_id
            if existing is None:
                ids_added += 1
            elif existing != matched_id:
                ids_updated += 1

    with open(rules_file_path, "w", encoding="utf-8") as fout:
        json.dump(rules_dict, fout)

    return {
        "rules_file_path": rules_file_path,
        "rule_pool_file_path": rule_pool_file_path,
        "total_rules_seen": total_rules,
        "rule_ids_added": ids_added,
        "rule_ids_updated": ids_updated,
        "rule_ids_missing_in_pool": ids_missing,
    }


class Rule_Learner(object):
    def __init__(self, edges, id2relation, inv_relation_id, dataset, mining_alg="unknown"):
        """
        Initialize rule learner object.

        Parameters:
            edges (dict): edges for each relation
            id2relation (dict): mapping of index to relation
            inv_relation_id (dict): mapping of relation to inverse relation
            dataset (str): dataset name

        Returns:
            None
        """

        self.edges = edges
        self.id2relation = id2relation
        self.inv_relation_id = inv_relation_id
        self.mining_alg = mining_alg

        self.found_rules = []
        self.found_rule_ids = set()
        self.rules_dict = dict()
        self.output_dir = "../../data/processed_new/" + dataset + "/output/" + dataset + "/"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.rule_pool_file = self.output_dir + "common_rule_pool.json"

    def _canonical_rule_payload(self, rule):
        return canonical_rule_payload(rule)

    def _make_rule_id(self, rule):
        return make_rule_id(rule)

    def _load_common_rule_pool(self):
        if os.path.exists(self.rule_pool_file):
            with open(self.rule_pool_file, "r", encoding="utf-8") as fin:
                return json.load(fin)
        return {}

    def _save_common_rule_pool(self, rule_pool):
        with open(self.rule_pool_file, "w", encoding="utf-8") as fout:
            json.dump(rule_pool, fout, indent=2)

    def _update_common_rule_pool(self):
        rule_pool = self._load_common_rule_pool()

        for _, rel_rules in self.rules_dict.items():
            for rule in rel_rules:
                rule_id = rule["rule_id"]
                canonical = self._canonical_rule_payload(rule)

                if rule_id not in rule_pool:
                    rule_pool[rule_id] = {
                        "rule_id": rule_id,
                        "head_rel": canonical["head_rel"],
                        "body_rels": canonical["body_rels"],
                        "var_constraints": canonical["var_constraints"],
                        "found_by": [],
                        "algorithm_stats": {},
                    }

                found_by = set(rule_pool[rule_id].get("found_by", []))
                found_by.add(self.mining_alg)
                rule_pool[rule_id]["found_by"] = sorted(found_by)
                rule_pool[rule_id]["algorithm_stats"][self.mining_alg] = {
                    "conf": rule.get("conf"),
                    "rule_supp": rule.get("rule_supp"),
                    "body_supp": rule.get("body_supp"),
                }

        self._save_common_rule_pool(rule_pool)

    def create_rule(self, walk, custom_generated = False):
        """
        Create a rule given a cyclic temporal random walk.
        The rule contains information about head relation, body relations,
        variable constraints, confidence, rule support, and body support.
        A rule is a dictionary with the content
        {"head_rel": int, "body_rels": list, "var_constraints": list,
         "conf": float, "rule_supp": int, "body_supp": int}

        Parameters:
            walk (dict): cyclic temporal random walk
                         {"entities": list, "relations": list, "timestamps": list}

        Returns:
            rule (dict): created rule
        """

        if not custom_generated:
            rule = dict()
            rule["head_rel"] = int(walk["relations"][0])
            rule["body_rels"] = [
                self.inv_relation_id[x] for x in walk["relations"][1:][::-1]
            ]
            rule["var_constraints"] = self.define_var_constraints(
                walk["entities"][1:][::-1]
            )
        else:
            rule = walk

        rule["rule_id"] = self._make_rule_id(rule)
        rule["found_by"] = [self.mining_alg]

        if rule["rule_id"] not in self.found_rule_ids:
            self.found_rules.append(rule.copy())
            self.found_rule_ids.add(rule["rule_id"])
            (
                rule["conf"],
                rule["rule_supp"],
                rule["body_supp"],
            ) = self.estimate_confidence(rule, full_samples = True if custom_generated else False)

            if rule["conf"]:
                self.update_rules_dict(rule)

    def define_var_constraints(self, entities):
        """
        Define variable constraints, i.e., state the indices of reoccurring entities in a walk.

        Parameters:
            entities (list): entities in the temporal walk

        Returns:
            var_constraints (list): list of indices for reoccurring entities
        """

        var_constraints = []
        for ent in set(entities):
            all_idx = [idx for idx, x in enumerate(entities) if x == ent]
            var_constraints.append(all_idx)
        var_constraints = [x for x in var_constraints if len(x) > 1]

        return sorted(var_constraints)

    def estimate_confidence(self, rule, num_samples=500, full_samples = False):
        """
        Estimate the confidence of the rule by sampling bodies and checking the rule support.

        Parameters:
            rule (dict): rule
                         {"head_rel": int, "body_rels": list, "var_constraints": list}
            num_samples (int): number of samples

        Returns:
            confidence (float): confidence of the rule, rule_support/body_support
            rule_support (int): rule support
            body_support (int): body support
        """

        all_bodies = []
        

        if not full_samples:

            for _ in range(num_samples):
                sample_successful, body_ents_tss = self.sample_body(
                    rule["body_rels"], rule["var_constraints"]
                )
                if sample_successful:
                    all_bodies.append(body_ents_tss)

        else:

            for i in range(len(self.edges[rule["body_rels"][0]])):
                sample_successful, body_ents_tss = self.sample_body(
                    rule["body_rels"], rule["var_constraints"], index = i
                )
                if sample_successful:
                    all_bodies.append(body_ents_tss)

        all_bodies.sort()
        unique_bodies = list(x for x, _ in itertools.groupby(all_bodies))
        body_support = len(unique_bodies)

        confidence, rule_support = 0, 0
        if body_support:
            rule_support = self.calculate_rule_support(unique_bodies, rule["head_rel"])
            confidence = round(rule_support / body_support, 6)

        return confidence, rule_support, body_support

    def sample_body(self, body_rels, var_constraints, index = None):
        """
        Sample a walk according to the rule body.
        The sequence of timesteps should be non-decreasing.

        Parameters:
            body_rels (list): relations in the rule body
            var_constraints (list): variable constraints for the entities

        Returns:
            sample_successful (bool): if a body has been successfully sampled
            body_ents_tss (list): entities and timestamps (alternately entity and timestamp)
                                  of the sampled body
        """

        sample_successful = True
        body_ents_tss = []
        cur_rel = body_rels[0]
        rel_edges = self.edges[cur_rel]
        next_edge = rel_edges[np.random.choice(len(rel_edges))] if index == None else rel_edges[index]
        cur_ts = next_edge[3]
        cur_node = next_edge[2]
        body_ents_tss.append(next_edge[0])
        body_ents_tss.append(cur_ts)
        body_ents_tss.append(cur_node)

        for cur_rel in body_rels[1:]:
            next_edges = self.edges[cur_rel]
            mask = (next_edges[:, 0] == cur_node) * (next_edges[:, 3] >= cur_ts)
            filtered_edges = next_edges[mask]

            if len(filtered_edges):
                next_edge = filtered_edges[np.random.choice(len(filtered_edges))]
                cur_ts = next_edge[3]
                cur_node = next_edge[2]
                body_ents_tss.append(cur_ts)
                body_ents_tss.append(cur_node)
            else:
                sample_successful = False
                break

        if sample_successful and var_constraints:
            # Check variable constraints
            body_var_constraints = self.define_var_constraints(body_ents_tss[::2])
            if body_var_constraints != var_constraints:
                sample_successful = False

        return sample_successful, body_ents_tss

    def calculate_rule_support(self, unique_bodies, head_rel):
        """
        Calculate the rule support. Check for each body if there is a timestamp
        (larger than the timestamps in the rule body) for which the rule head holds.

        Parameters:
            unique_bodies (list): bodies from self.sample_body
            head_rel (int): head relation

        Returns:
            rule_support (int): rule support
        """

        rule_support = 0
        head_rel_edges = self.edges[head_rel]
        for body in unique_bodies:
            mask = (
                (head_rel_edges[:, 0] == body[0])
                * (head_rel_edges[:, 2] == body[-1])
                * (head_rel_edges[:, 3] > body[-2])
            )

            if True in mask:
                rule_support += 1

        return rule_support

    def update_rules_dict(self, rule):
        """
        Update the rules if a new rule has been found.

        Parameters:
            rule (dict): generated rule from self.create_rule

        Returns:
            None
        """

        try:
            self.rules_dict[rule["head_rel"]].append(rule)
        except KeyError:
            self.rules_dict[rule["head_rel"]] = [rule]

    def sort_rules_dict(self):
        """
        Sort the found rules for each head relation by decreasing confidence.

        Parameters:
            None

        Returns:
            None
        """

        for rel in self.rules_dict:
            self.rules_dict[rel] = sorted(
                self.rules_dict[rel], key=lambda x: x["conf"], reverse=True
            )

    def save_rules(self, dt, rule_lengths, num_walks, transition_distr, seed):
        """
        Save all rules.

        Parameters:
            dt (str): time now
            rule_lengths (list): rule lengths
            num_walks (int): number of walks
            transition_distr (str): transition distribution
            seed (int): random seed

        Returns:
            None
        """

        self._update_common_rule_pool()

        rules_dict = {int(k): v for k, v in self.rules_dict.items()}
        filename = "{0}_r{1}_n{2}_{3}_s{4}_rules.json".format(
            dt, rule_lengths, num_walks, transition_distr, seed
        )
        filename = filename.replace(" ", "")
        with open(self.output_dir + filename, "w", encoding="utf-8") as fout:
            json.dump(rules_dict, fout)

    def save_rules_verbalized(
        self, dt, rule_lengths, num_walks, transition_distr, seed
    ):
        """
        Save all rules in a human-readable format.

        Parameters:
            dt (str): time now
            rule_lengths (list): rule lengths
            num_walks (int): number of walks
            transition_distr (str): transition distribution
            seed (int): random seed

        Returns:
            None
        """

        rules_str = ""
        for rel in self.rules_dict:
            for rule in self.rules_dict[rel]:
                rules_str += verbalize_rule(rule, self.id2relation) + "\n"

        filename = "{0}_r{1}_n{2}_{3}_s{4}_rules.txt".format(
        # filename = "YAGO_rules.txt".format(
            dt, rule_lengths, num_walks, transition_distr, seed
        )
        filename = filename.replace(" ", "")
        with open(self.output_dir + filename, "w", encoding="utf-8") as fout:
            fout.write(rules_str)


def verbalize_rule(rule, id2relation):
    """
    Verbalize the rule to be in a human-readable format.

    Parameters:
        rule (dict): rule from Rule_Learner.create_rule
        id2relation (dict): mapping of index to relation

    Returns:
        rule_str (str): human-readable rule
    """

    if rule["var_constraints"]:
        var_constraints = rule["var_constraints"]
        constraints = [x for sublist in var_constraints for x in sublist]
        for i in range(len(rule["body_rels"]) + 1):
            if i not in constraints:
                var_constraints.append([i])
        var_constraints = sorted(var_constraints)
    else:
        var_constraints = [[x] for x in range(len(rule["body_rels"]) + 1)]

    rule_str = "{0:8.6f}  {1:4}  {2:4}  {3}(X0,X{4},T{5}) <- "
    obj_idx = [
        idx
        for idx in range(len(var_constraints))
        if len(rule["body_rels"]) in var_constraints[idx]
    ][0]
    rule_str = rule_str.format(
        rule["conf"],
        rule["rule_supp"],
        rule["body_supp"],
        id2relation[rule["head_rel"]],
        obj_idx,
        len(rule["body_rels"]),
    )

    for i in range(len(rule["body_rels"])):
        sub_idx = [
            idx for idx in range(len(var_constraints)) if i in var_constraints[idx]
        ][0]
        obj_idx = [
            idx for idx in range(len(var_constraints)) if i + 1 in var_constraints[idx]
        ][0]
        rule_str += "{0}(X{1},X{2},T{3}), ".format(
            id2relation[rule["body_rels"][i]], sub_idx, obj_idx, i
        )

    return rule_str[:-2]


def rules_statistics(rules_dict):
    """
    Show statistics of the rules.

    Parameters:
        rules_dict (dict): rules

    Returns:
        None
    """

    print(
        "Number of relations with rules: ", len(rules_dict)
    )  # Including inverse relations
    print("Total number of rules: ", sum([len(v) for k, v in rules_dict.items()]))

    lengths = []
    for rel in rules_dict:
        lengths += [len(x["body_rels"]) for x in rules_dict[rel]]
    rule_lengths = [(k, v) for k, v in Counter(lengths).items()]
    print("Number of rules by length: ", sorted(rule_lengths))


if __name__ == "__main__":

    result = update_common_rule_pool_from_rules_file(
        rules_file_path="../../data/processed_new/icews14/output/icews14/080525134642_r[1]_n200_exp_s1_rules.json",
        mining_alg="gtkg"
    )
    print(result)

    
    # result = add_rule_ids_from_common_pool(
    #     rules_file_path="../../data/processed_new/icews14/output/icews14/080525131706_r[1]_n200_exp_s1_rules.json"
    # )
    # print(result)