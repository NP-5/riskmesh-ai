"""Re-engineer Agent: propose -> backtest -> explain -> require human approval -> activate. Never silent."""
import itertools
import numpy as np
import pandas as pd
from .db import audit

BASE_RULE = {"id": "rules-v1", "name": "Shared device = high risk", "combinator": "AND",
             "conditions": [{"feature": "shared_device_count", "op": ">=", "value": 3}]}
OPS = {">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b, ">": lambda a, b: a > b}


def rule_text(rule): return f" {rule['combinator']} ".join(f"{c['feature']} {c['op']} {c['value']}" for c in rule["conditions"])


def make_history(seed=11) -> pd.DataFrame:
    """Synthetic labelled history of candidate groups (NOT real data)."""
    rng, rows = np.random.default_rng(seed), []
    for _ in range(120):   # genuine rings
        lead = float(rng.gamma(4, 1.0))
        rows.append(dict(label=1, shared_device_count=int(rng.choice([2, 3, 4, 5, 6], p=[.1, .3, .3, .2, .1])),
                         beneficiary_concentration=float(rng.beta(6, 2.2)), escalation_signal=float(np.clip(rng.normal(66, 17), 0, 100)),
                         customers=int(rng.integers(3, 7)), lead_structural=lead, lead_escalation=lead * float(rng.uniform(.6, .9))))
    for _ in range(280):   # legitimate shared infrastructure (offices, households, vendors)
        vendor = rng.random() < 0.2
        rows.append(dict(label=0, shared_device_count=int(rng.choice(range(1, 8), p=[.2, .25, .2, .15, .1, .06, .04])),
                         beneficiary_concentration=float(rng.beta(6, 2.2) if vendor else rng.beta(2.5, 3.5)),
                         escalation_signal=float(np.clip(rng.normal(32, 18), 0, 100)), customers=int(rng.integers(2, 12)),
                         lead_structural=0.0, lead_escalation=0.0))
    return pd.DataFrame(rows)


def evaluate_rule(rule, hist: pd.DataFrame) -> dict:
    mask = np.ones(len(hist), bool)
    for c in rule["conditions"]:
        m = OPS[c["op"]](hist[c["feature"]], c["value"]); mask &= m.to_numpy()
    y = hist.label.to_numpy() == 1
    tp, fp, fn, tn = int((mask & y).sum()), int((mask & ~y).sum()), int((~mask & y).sum()), int((~mask & ~y).sum())
    uses_esc = any(c["feature"] == "escalation_signal" for c in rule["conditions"])
    lead = hist.loc[mask & y, "lead_escalation" if uses_esc else "lead_structural"]
    return {"precision": round(tp / max(tp + fp, 1), 3), "recall": round(tp / max(tp + fn, 1), 3),
            "false_positive_rate": round(fp / max(fp + tn, 1), 3), "false_negative_rate": round(fn / max(fn + tp, 1), 3),
            "affected_legitimate_customers": int(hist.loc[mask & ~y, "customers"].sum()),
            "investigation_volume": int(mask.sum()), "avg_lead_time_hours": round(float(lead.mean()), 2) if len(lead) else None,
            "counts": {"tp": tp, "fp": fp, "fn": fn, "tn": tn}}


class RuleRegistry:
    def __init__(self):
        self.history = make_history()
        self.versions = {"rules-v1": {**BASE_RULE, "status": "active"}}
        self.active = "rules-v1"
        self.proposals = {}

    def active_rule(self): return self.versions[self.active]

    def flags(self, ring) -> bool:
        feats = {"shared_device_count": ring["max_on_device"], "beneficiary_concentration": ring["concentration"],
                 "escalation_signal": ring["escalation"]}
        return all(OPS[c["op"]](feats[c["feature"]], c["value"]) for c in self.active_rule()["conditions"])

    def backtest(self, rule, baseline=None) -> dict:
        before = evaluate_rule(baseline or self.active_rule(), self.history); after = evaluate_rule(rule, self.history)
        delta = {k: round(after[k] - before[k], 3) for k in before if isinstance(before[k], (int, float)) and after.get(k) is not None}
        audit("SYSTEM", "backtest_engine", "RULE_BACKTEST", rule_text(rule), {"after": {k: v for k, v in after.items() if k != 'counts'}})
        return {"rule": rule_text(rule), "baseline_rule": rule_text(baseline or self.active_rule()), "before": before, "after": after,
                "delta": delta, "data": "Synthetic historical groups (400); illustrative only."}

    def propose(self, min_recall=0.70) -> dict:
        base, best = self.active_rule(), None
        for conc, esc in itertools.product([.4, .5, .6, .7, .8], [30, 40, 50, 60, 70]):
            cand = {"conditions": base["conditions"] + [{"feature": "beneficiary_concentration", "op": ">=", "value": conc},
                                                        {"feature": "escalation_signal", "op": ">=", "value": esc}], "combinator": "AND"}
            m = evaluate_rule(cand, self.history)
            if m["recall"] >= min_recall and (best is None or (m["precision"], -m["affected_legitimate_customers"]) > (best[1]["precision"], -best[1]["affected_legitimate_customers"])):
                best = (cand, m)
        if best is None:
            raise ValueError("no candidate rule met the minimum recall; nothing proposed")
        pid, ver = f"PROP_{len(self.proposals) + 1}", f"rules-v{len(self.versions) + 1}"
        rule = {"id": ver, "name": "Shared device + concentration + escalation", **best[0]}
        bt = self.backtest(rule)
        prop = {"proposal_id": pid, "proposed_version": ver, "status": "PENDING_HUMAN_APPROVAL", "rule": rule, "backtest": bt,
                "explanation": (f"Adding beneficiary concentration and 6-hour escalation conditions to '{rule_text(base)}' raises precision from "
                                f"{bt['before']['precision']} to {bt['after']['precision']} and cuts affected legitimate customers from "
                                f"{bt['before']['affected_legitimate_customers']} to {bt['after']['affected_legitimate_customers']}, at a recall cost "
                                f"of {abs(bt['delta']['recall'])}. Requires human approval; production rules are unchanged until then."),
                "proposed_by": "reengineer_agent"}
        self.proposals[pid] = prop
        audit("SYSTEM", "reengineer_agent", "RULE_PROPOSED", rule_text(rule), {"proposal_id": pid})
        return prop

    def decide(self, pid, approve: bool, actor: str, reason: str) -> dict:
        p = self.proposals[pid]
        if p["status"] != "PENDING_HUMAN_APPROVAL":
            raise ValueError(f"proposal already {p['status']}")
        if not actor or not reason:
            raise ValueError("actor and reason are required")
        p["status"] = "ACTIVATED" if approve else "REJECTED"; p["decided_by"], p["decision_reason"] = actor, reason
        if approve:
            self.versions[self.active]["status"] = "superseded"
            self.versions[p["proposed_version"]] = {**p["rule"], "status": "active"}; self.active = p["proposed_version"]
        audit("SYSTEM", actor, "RULE_ACTIVATED" if approve else "RULE_REJECTED", reason, {"proposal_id": pid, "version": p["proposed_version"]})
        return p


RULES = RuleRegistry()
