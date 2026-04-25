from __future__ import annotations

BEST_ID = "099"
BEST_SCORE = 38.20618
FAILED_ID = "129"
FAILED_SCORE = 37.89482
FAILED_SUBMISSION_ID = "manual_129_score_37_89482"
ANCHOR_ID = "087"
ANCHOR_SCORE = 37.35945
MAPPING = {"040": "010", "036": "011", "016": "008", "024": "023", "019": "009"}
RISK_LABELS = ["safe_external_generalization", "medium_public_behavior_mapping", "high_risk_public_reconstruction"]
FUSION_LABEL = "three_tier_fusion_with_high_risk_reconstruction"


def output_rows() -> list[tuple[str, dict, str]]:
    return [
        ("counter129_005", {"mode": "outputs", "best": 1.05, "failed": -0.05, "anchor": 0.0}, "Move five percent opposite the failed 129 vector."),
        ("counter129_010", {"mode": "outputs", "best": 1.10, "failed": -0.10, "anchor": 0.0}, "Move ten percent opposite the failed 129 vector."),
        ("counter129_020", {"mode": "outputs", "best": 1.20, "failed": -0.20, "anchor": 0.0}, "Move twenty percent opposite the failed 129 vector."),
        ("counter129_035", {"mode": "outputs", "best": 1.35, "failed": -0.35, "anchor": 0.0}, "Stress-test the counter-129 direction."),
        ("damped129_005", {"mode": "outputs", "best": 0.95, "failed": 0.05, "anchor": 0.0}, "Keep only five percent of the failed 129 displacement."),
        ("damped129_010", {"mode": "outputs", "best": 0.90, "failed": 0.10, "anchor": 0.0}, "Keep ten percent of the failed 129 displacement."),
        ("damped129_015", {"mode": "outputs", "best": 0.85, "failed": 0.15, "anchor": 0.0}, "Keep fifteen percent of the failed 129 displacement."),
        ("counter_prior_mix", {"mode": "outputs", "best": 1.25, "failed": -0.10, "anchor": -0.15}, "Blend counter-129 movement with the successful 087 to 099 direction."),
        ("prior_success_025", {"mode": "outputs", "best": 1.25, "failed": 0.0, "anchor": -0.25}, "Extend the successful 087 to 099 direction by twenty-five percent."),
        ("prior_success_050", {"mode": "outputs", "best": 1.50, "failed": 0.0, "anchor": -0.50}, "Extend the successful 087 to 099 direction by fifty percent."),
        ("prior_success_075", {"mode": "outputs", "best": 1.75, "failed": 0.0, "anchor": -0.75}, "Extend the successful 087 to 099 direction by seventy-five percent."),
        ("prior_success_100", {"mode": "outputs", "best": 2.00, "failed": 0.0, "anchor": -1.00}, "Extend the successful 087 to 099 direction by one full step."),
        ("prior_counter_balanced", {"mode": "outputs", "best": 1.40, "failed": -0.15, "anchor": -0.25}, "Combine a moderate counter-129 vector with a smaller prior-success vector."),
        ("prior_counter_strong", {"mode": "outputs", "best": 1.65, "failed": -0.25, "anchor": -0.40}, "Stronger rebound from the failed direction plus prior-success extension."),
    ]


def high_rows() -> list[tuple[str, dict, str]]:
    rows = []
    for variant, weight in [("base", .12), ("base", .20), ("cal_direct", .12), ("cal_direct", .20), ("item_mean", .12), ("merged_avg", .16)]:
        rows.append((f"zmatch_{variant}_{int(weight*100):02d}", {"mode": "component", "component": "high", "variant": variant, "weight": weight, "mapping": MAPPING}, f"Mix distribution-matched high-risk {variant} evidence at weight {weight:.2f}."))
    return rows


def stack_rows() -> list[tuple[str, dict, str]]:
    rows = []
    for model, weight in [("ridge_log", .10), ("ridge_log", .18), ("ridge_raw", .15), ("huber", .15), ("trees", .12), ("trees", .20), ("ridge_log", .12), ("ridge_raw", .22), ("huber", .20)]:
        extra = {"counter": .08} if len(rows) >= 6 else {}
        rows.append((f"stack_{model}_{int(weight*100):02d}_{len(rows)+1:02d}", {"mode": "component", "component": "stack", "model": model, "weight": weight, "mapping": MAPPING, **extra}, f"Train {model} and add only a distribution-matched residual at weight {weight:.2f}."))
    return rows


def champion_row() -> tuple[str, dict, str]:
    config = {"mode": "champion", "counter": .28, "prior": .80, "stack_weight": .16, "high_weight": .12, "model": "ridge_log", "variant": "cal_direct", "mapping": MAPPING}
    return "champion_counter_prior_stack", config, "Champion: keep 099 as base, reverse the failed 129 direction, add small prior-success extension, and inject distribution-matched trained/high-risk residuals."


def specs(start: int) -> list[dict]:
    rows = output_rows() + high_rows() + stack_rows() + [champion_row()]
    final_id = f"{start + len(rows) - 1:03d}"
    out = []
    for offset, (name, config, hypothesis) in enumerate(rows):
        candidate_id = f"{start + offset:03d}"
        groups = ["safe_surface_via_best_fusion", "medium_public_behavior_via_best_fusion", "high_public_reconstruction_via_best_fusion"]
        if config.get("component") == "stack" or config.get("mode") == "champion":
            groups.append("trained_residual")
        if config.get("component") == "high" or config.get("mode") == "champion":
            groups.append("distribution_matched_high_risk")
        out.append({"candidate_id": candidate_id, "name": f"rebound_{name}", "config": config, "family": "failure_rebound_feedback", "risk_label": FUSION_LABEL, "risk_labels": RISK_LABELS, "fused_evidence_groups": groups, "next_manual_upload_target": candidate_id == final_id, "hypothesis": hypothesis})
    return out
