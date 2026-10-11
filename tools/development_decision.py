"""
The decision logic after E1, E2 and E3: which part of the page pipeline do the results point at?

    E1 good + E2 poor   -> the boxes are the weak part: segmentation / layout
    E1 poor + E2 poor   -> the recognizer does not read these pages well even with the right boxes
    E1 good + E2 good   -> the present pipeline is viable on these pages
    E3 strongly sensitive -> the exactness of the box matters, whatever else is true

"Good" and "poor" need numbers. The numbers are in development_pages/experiments/DECISION_RULES.json, written BEFORE
any result exists and hashed into every decision. THEY ARE A PROPOSAL BY CLAUDE CODE, NOT A RULING OF THE PROJECT
OWNER, and not a registered hypothesis. The owner may replace the file before the evaluation runs; replacing it after
results have been seen must be said openly wherever the decision is reported. Between "good" and "poor" lies
"intermediate", and the engine then says that the evidence does not decide.

The output never states a cause. These are development pages with a sparse, single-reviewer reference; the decision
names where to look next, and what it rests on.

    .venv\\Scripts\\python -B tools\\development_decision.py --write-rules      create the rules file (refuses to overwrite)
"""
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import development_eval_core as core

RULES_PATH = core.EXPERIMENTS / "DECISION_RULES.json"
# CH-06 (owner decision D-08, 2026-10-09): no good/poor verdict while the rules are not approved by the owner. The block
# below is the mechanism only; approving the mechanism did not approve any threshold. A rules file without the block counts
# as not approved.
NOT_ASSESSED = "NOT ASSESSED — decision rules not owner-approved"
DEFAULT_RULES = {
    "status": "PROPOSED by Claude Code on 2026-10-08, before any E1/E2/E3 result existed. Not a ruling of the project owner. Not a registered hypothesis.",
    "approval": {"approved": False, "approved_by": None, "approved_at": None, "supersedes": None},
    "basis": "The only earlier figure for this recognizer on real printed Tamil is the R1 pilot under oracle layout: macro-CER 5.62 % with a 95 % interval of "
             "1.89-10.95 %, against an automated reference. 'Good' for E1 is set at the upper end of that interval; 'poor' at twice that.",
    "measured_on": "the primary set (verified, usable, Tamil script); part B (the stratified sample) is used when it holds at least 30 scored regions, else the whole primary set",
    "E1": {"good_if_cer_at_most": 0.10, "poor_if_cer_above": 0.20},
    "E2": {"good_if_box_recall_at_least": 0.90, "and_cer_excess_over_E1_at_most": 0.05,
           "poor_if_box_recall_below": 0.75, "or_cer_excess_over_E1_above": 0.15},
    "E3": {"strong_if_cer_spread_at_least": 0.05, "or_cer_at_0px_above_best_by_at_least": 0.03},
    "minimum_regions": 30,
}


def load_rules(path: Path = RULES_PATH) -> tuple:
    """(rules, sha256 of the file, source). The file if it exists; otherwise the built-in proposal, said so."""
    if Path(path).is_file():
        return json.loads(Path(path).read_text(encoding="utf-8")), core.sha_file(path), str(path)
    return DEFAULT_RULES, core.sha_json(DEFAULT_RULES), "built-in proposal (no rules file found)"


def _pick(metrics: dict, minimum: int) -> tuple:
    b = metrics.get("primary|part:B")
    if b and b["n"] >= minimum:
        return b, "part B (stratified sample)"
    return metrics.get("primary"), "whole primary set"


def _pick_pair(e1_metrics, e2_metrics, minimum: int) -> tuple:
    """E1 and E2 numbers on the SAME scope, so that "E2 excess over E1" compares like with like: part B for both when
    both have at least `minimum` scored regions there, otherwise the whole primary set for both. With only one of the
    two runs present, that run's own choice (_pick) is used."""
    a = (e1_metrics or {}).get("primary_padding", {})
    b = (e2_metrics or {}).get("end_to_end", {})
    if not (e1_metrics and e2_metrics):
        return _pick(a, minimum) + _pick(b, minimum)
    ba, bb = a.get("primary|part:B"), b.get("primary|part:B")
    if ba and bb and ba["n"] >= minimum and bb["n"] >= minimum:
        return ba, "part B (stratified sample)", bb, "part B (stratified sample)"
    return a.get("primary"), "whole primary set", b.get("primary"), "whole primary set"


def grade_e1(m, rules) -> str:
    if not m or m["n"] < rules["minimum_regions"]:
        return "not decidable (too few regions)"
    if m["cer"] <= rules["E1"]["good_if_cer_at_most"]:
        return "good"
    return "poor" if m["cer"] > rules["E1"]["poor_if_cer_above"] else "intermediate"


def grade_e2(m2, m1, recall, rules) -> str:
    if not m2 or not m1 or m2["n"] < rules["minimum_regions"]:
        return "not decidable (too few regions)"
    excess, r = m2["cer"] - m1["cer"], rules["E2"]
    if recall >= r["good_if_box_recall_at_least"] and excess <= r["and_cer_excess_over_E1_at_most"]:
        return "good"
    if recall < r["poor_if_box_recall_below"] or excess > r["or_cer_excess_over_E1_above"]:
        return "poor"
    return "intermediate"


def grade_e3(common: dict, rules) -> tuple:
    if not common or "0" not in common:
        return "not decidable", None
    cers = {int(k): v["cer"] for k, v in common.items()}
    spread, at0 = max(cers.values()) - min(cers.values()), cers[0] - min(cers.values())
    strong = spread >= rules["E3"]["strong_if_cer_spread_at_least"] or at0 >= rules["E3"]["or_cer_at_0px_above_best_by_at_least"]
    return ("strong" if strong else "not strong"), {"cer_by_margin": cers, "spread": spread, "cer_at_0px_minus_best": at0, "best_margin_px": min(cers, key=lambda k: (cers[k], abs(k)))}


def approved(rules: dict) -> bool:
    a = rules.get("approval")
    return isinstance(a, dict) and a.get("approved") is True


def decide(e1_metrics: dict, e2_metrics: dict, e2_boxes: dict, e3_common: dict, rules: dict) -> dict:
    """The decision record. Inputs are the metrics files of the three runs (any may be None if its run failed).
    Unapproved rules: the numbers are reported, and no grade, case or verdict is formed from them."""
    if not approved(rules):
        return _not_assessed(e1_metrics, e2_metrics, e2_boxes, e3_common, rules)
    m1, scope1, m2, scope2 = _pick_pair(e1_metrics, e2_metrics, rules["minimum_regions"])
    recall = (e2_boxes or {}).get("pooled", {}).get("recall")
    g1 = grade_e1(m1, rules) if e1_metrics else "not run"
    g2 = grade_e2(m2, m1, recall, rules) if e2_metrics and e1_metrics and recall is not None else "not run"
    g3, d3 = grade_e3((e3_common or {}).get("common_set", {}), rules) if e3_common else ("not run", None)
    if g1 == "good" and g2 == "poor":
        case, text = "A", ("With verified boxes the recognizer reads these words within the 'good' range, and with the boxes found by the program it does not. "
                           "On this evidence the found boxes are the weaker part: segmentation and layout are where to work next.")
    elif g1 == "poor" and g2 == "poor":
        case, text = "B", ("Even with verified boxes the recognizer's error on these pages is in the 'poor' range, and it is poor with found boxes as well. "
                           "On this evidence the recognizer's reading of this print limits the result; better boxes alone would not bring it into the 'good' range.")
    elif g1 == "good" and g2 == "good":
        case, text = "C", ("With verified boxes and with found boxes the result is in the 'good' range. On this evidence the present pipeline is viable on these pages.")
    else:
        case, text = "undecided", (f"E1 is '{g1}' and E2 is '{g2}'. The rules name a case only for good/poor, poor/poor and good/good; this combination does not decide. "
                                   "Read the two reports side by side.")
    if g3 == "strong":
        text += " The box-perturbation result is strong: a few pixels of margin change the error noticeably, so the exactness of the box matters (case D applies as well)."
    elif g3 == "not strong":
        text += " The box-perturbation result is not strong: within the margins tested, the exact box edge changes the error little."
    return {"case": case, "case_D_box_precision_matters": g3 == "strong", "grades": {"E1": g1, "E2": g2, "E3_sensitivity": g3},
            "numbers": {"E1_cer": m1["cer"] if m1 else None, "E1_scope": scope1, "E1_regions": m1["n"] if m1 else None,
                        "E2_cer": m2["cer"] if m2 else None, "E2_scope": scope2, "E2_cer_excess_over_E1": (m2["cer"] - m1["cer"]) if m1 and m2 else None,
                        "E2_box_recall": recall, "E3": d3},
            "statement": text,
            "caution": "This is not a finding about causes. The pages are development pages that the tools were built on; the reference covers part of each page, was made by one "
                       "reviewer who saw the AI annotation, and the thresholds are a proposal fixed in advance, not a registered hypothesis. The decision says where to look next."}


def _not_assessed(e1_metrics, e2_metrics, e2_boxes, e3_common, rules) -> dict:
    """The metrics, side by side, with no threshold applied. Nothing here is a grade."""
    m1, scope1, m2, scope2 = _pick_pair(e1_metrics, e2_metrics, rules["minimum_regions"])
    recall = (e2_boxes or {}).get("pooled", {}).get("recall")
    common = (e3_common or {}).get("common_set", {})
    cers = {int(k): v["cer"] for k, v in common.items()} if common else None
    e3 = None if not cers else {"cer_by_margin": cers, "spread": max(cers.values()) - min(cers.values()),
                                "cer_at_0px_minus_best": (cers[0] - min(cers.values())) if 0 in cers else None,
                                "best_margin_px": min(cers, key=lambda k: (cers[k], abs(k)))}
    run = lambda x: NOT_ASSESSED if x else "not run"
    return {"case": NOT_ASSESSED, "case_D_box_precision_matters": None, "verdict": NOT_ASSESSED,
            "grades": {"E1": run(e1_metrics), "E2": run(e2_metrics), "E3_sensitivity": run(e3_common)},
            "numbers": {"E1_cer": m1["cer"] if m1 else None, "E1_scope": scope1, "E1_regions": m1["n"] if m1 else None,
                        "E2_cer": m2["cer"] if m2 else None, "E2_scope": scope2, "E2_cer_excess_over_E1": (m2["cer"] - m1["cer"]) if m1 and m2 else None,
                        "E2_box_recall": recall, "E3": e3},
            "statement": (NOT_ASSESSED + ". The metrics are reported above and in the three run reports; no threshold was applied to them. "
                          "The owner has not stated an operational requirement (D-02 deferred) and has not approved the rules (approval.approved is false)."),
            "caution": "These are development pages that the tools were built on; the reference covers part of each page and was made by one reviewer who saw the AI annotation."}


def main():
    if "--write-rules" in sys.argv:
        if RULES_PATH.exists():
            sys.exit(f"{RULES_PATH} exists already; it is not overwritten.")
        core.write_new(RULES_PATH, core.dump(DEFAULT_RULES))
        print("written", RULES_PATH, core.sha_file(RULES_PATH))
        return
    rules, sha, src = load_rules()
    print(json.dumps({"rules_from": src, "rules_sha256": sha, "rules": rules}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
