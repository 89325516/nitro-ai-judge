from __future__ import annotations

ANCHOR_ID = "087"
ANCHOR_SCORE = 37.35945
BEST_ID = "099"
BEST_SCORE = 38.20618
BEST_SUBMISSION_ID = "manual_099_score_38_20618"
ANCHOR_MAPPING = {"040": "010", "036": "011", "016": "008", "024": "023", "019": "009"}
SAFE_LABEL = "safe_external_generalization"
MEDIUM_LABEL = "medium_public_behavior_mapping"
HIGH_LABEL = "high_risk_public_reconstruction"
FUSION_LABEL = "three_tier_fusion_with_high_risk_reconstruction"
RISK_LABELS = [SAFE_LABEL, MEDIUM_LABEL, HIGH_LABEL]


def direction_rows() -> list[tuple[str, dict, str]]:
    rows = [
        ("direction_gamma150", {"mode": "direction", "gamma": 1.5, "scale": 1.00}, "Extend the verified 087 to 099 improvement direction by gamma 1.5."),
        ("direction_gamma200", {"mode": "direction", "gamma": 2.0, "scale": 1.00}, "Extend the verified official direction by gamma 2.0."),
        ("direction_gamma300", {"mode": "direction", "gamma": 3.0, "scale": 1.00}, "Make a large official-direction move instead of another tiny blend."),
        ("direction_gamma400", {"mode": "direction", "gamma": 4.0, "scale": 1.00}, "Stress-test the official direction with gamma 4.0."),
        ("direction_gamma600", {"mode": "direction", "gamma": 6.0, "scale": 1.00}, "Aggressive official-direction extrapolation at gamma 6.0."),
        ("direction_gamma800", {"mode": "direction", "gamma": 8.0, "scale": 1.00}, "Maximum official-direction extrapolation in the first block."),
        ("direction_gamma300_scale104", {"mode": "direction", "gamma": 3.0, "scale": 1.04}, "Gamma 3.0 plus positive scale shift."),
        ("direction_gamma300_zero01", {"mode": "direction", "gamma": 3.0, "scale": 1.00, "zero_quantile": 0.01}, "Gamma 3.0 with the lowest one percent set to zero."),
        ("direction_gamma400_floor02", {"mode": "direction", "gamma": 4.0, "scale": 1.00, "floor_quantile": 0.02}, "Gamma 4.0 with low-tail floor handling."),
        ("direction_gamma600_cap995", {"mode": "direction", "gamma": 6.0, "scale": 1.00, "cap_quantile": 0.995}, "Gamma 6.0 with high-tail cap control."),
    ]
    return rows


def trained_rows() -> list[tuple[str, dict, str]]:
    base = {"mode": "trained", "high_variant": "cal_direct", "mapping": ANCHOR_MAPPING}
    rows = [
        ("train_ridge_raw_balanced", {**base, "model": "ridge_raw", "alpha": 80.0, "weights": {"safe": .12, "medium": .28, "high": .20, "stack": .40}}, "Train a raw Ridge stacker on public behavior and surface features."),
        ("train_ridge_log_balanced", {**base, "model": "ridge_log", "alpha": 120.0, "weights": {"safe": .10, "medium": .30, "high": .20, "stack": .40}}, "Train a log-target Ridge stacker to change scale and tail behavior."),
        ("train_ridge_medium_heavy", {**base, "model": "ridge_raw", "alpha": 40.0, "weights": {"safe": .08, "medium": .42, "high": .18, "stack": .32}}, "Favor public item and text behavior inside a trained stack."),
        ("train_ridge_high_heavy", {**base, "model": "ridge_log", "alpha": 60.0, "weights": {"safe": .08, "medium": .25, "high": .35, "stack": .32}}, "Favor mapped public TRT while keeping trained correction active."),
        ("train_huber_balanced", {**base, "model": "huber", "alpha": 0.0001, "weights": {"safe": .10, "medium": .32, "high": .20, "stack": .38}}, "Use a robust Huber-style trained stacker for outlier resistance."),
        ("train_huber_high_push", {**base, "model": "huber", "alpha": 0.00005, "weights": {"safe": .07, "medium": .25, "high": .33, "stack": .35}}, "Robust stacker with stronger direct reconstruction signal."),
        ("train_tree_medium", {**base, "model": "trees", "trees": 48, "leaf": 6, "weights": {"safe": .08, "medium": .40, "high": .17, "stack": .35}}, "ExtraTrees-style stacker focused on public medium behavior."),
        ("train_tree_high", {**base, "model": "trees", "trees": 64, "leaf": 4, "weights": {"safe": .06, "medium": .25, "high": .34, "stack": .35}}, "ExtraTrees-style stacker with a high-risk reconstruction push."),
        ("train_ridge_zero_tail", {**base, "model": "ridge_log", "alpha": 90.0, "weights": {"safe": .10, "medium": .34, "high": .18, "stack": .38}, "zero_quantile": .015}, "Trained stacker plus explicit low-tail zero handling."),
        ("train_ridge_direction_blend", {**base, "model": "ridge_raw", "alpha": 60.0, "gamma": 3.0, "direction_weight": .22, "weights": {"safe": .08, "medium": .28, "high": .18, "stack": .46}}, "Trained stacker blended with the verified official direction."),
    ]
    return rows


def high_rows() -> list[tuple[str, dict, str]]:
    rows = [
        ("risk_base75", {"mode": "blend", "high_variant": "base", "weights": {"safe": .05, "medium": .20, "high": .75}, "scale": 1.02, "mapping": ANCHOR_MAPPING}, "Let the public TRT reconstruction base dominate the fused output."),
        ("risk_raw_direct75", {"mode": "blend", "high_variant": "raw_direct", "weights": {"safe": .05, "medium": .20, "high": .75}, "scale": 1.00, "mapping": ANCHOR_MAPPING}, "Use mapped raw public subject TRT as the dominant signal."),
        ("risk_cal_direct75", {"mode": "blend", "high_variant": "cal_direct", "weights": {"safe": .05, "medium": .20, "high": .75}, "scale": 1.00, "mapping": ANCHOR_MAPPING}, "Use calibrated mapped public subject TRT as the dominant signal."),
        ("risk_item_mean80", {"mode": "blend", "high_variant": "item_mean", "weights": {"safe": .04, "medium": .16, "high": .80}, "scale": 1.03, "mapping": ANCHOR_MAPPING}, "Push exact public item-level TRT recovery."),
        ("risk_merged_avg80", {"mode": "blend", "high_variant": "merged_avg", "weights": {"safe": .04, "medium": .16, "high": .80}, "scale": 1.03, "mapping": ANCHOR_MAPPING}, "Use public merged average TRT as a high-risk recovery source."),
        ("risk_raw_direct90", {"mode": "blend", "high_variant": "raw_direct", "weights": {"safe": .02, "medium": .08, "high": .90}, "scale": 1.00, "mapping": ANCHOR_MAPPING}, "Extreme raw public-subject reconstruction push."),
        ("risk_cal_direct90", {"mode": "blend", "high_variant": "cal_direct", "weights": {"safe": .02, "medium": .08, "high": .90}, "scale": .98, "mapping": ANCHOR_MAPPING}, "Extreme calibrated public-subject reconstruction push."),
        ("risk_item_zero", {"mode": "blend", "high_variant": "item_mean", "weights": {"safe": .04, "medium": .16, "high": .80}, "scale": 1.00, "zero_quantile": .02, "mapping": ANCHOR_MAPPING}, "Item-level recovery with aggressive zero-tail structure."),
        ("risk_direction_gamma4", {"mode": "blend", "high_variant": "cal_direct", "gamma": 4.0, "direction_weight": .30, "weights": {"safe": .04, "medium": .18, "high": .78}, "scale": 1.00, "mapping": ANCHOR_MAPPING}, "Calibrated reconstruction fused with a large official-direction move."),
    ]
    return rows


def champion_row() -> tuple[str, dict, str]:
    config = {"mode": "champion", "model": "ridge_log", "alpha": 70.0, "high_variant": "cal_direct", "gamma": 3.0, "direction_weight": .28, "weights": {"safe": .08, "medium": .30, "high": .24, "stack": .38}, "scale": 1.04, "mapping": ANCHOR_MAPPING}
    return "champion_trained_direction_risk", config, "Champion: trained stacker plus medium behavior, calibrated reconstruction, and gamma 3 official-direction extrapolation."


def specs(start: int) -> list[dict]:
    rows = direction_rows() + trained_rows() + high_rows() + [champion_row()]
    output = []
    final_id = f"{start + len(rows) - 1:03d}"
    for offset, (name, config, hypothesis) in enumerate(rows):
        candidate_id = f"{start + offset:03d}"
        groups = ["safe_surface", "medium_public_behavior", "high_public_reconstruction"]
        if config.get("mode") in {"trained", "champion"}:
            groups.append("trained_stacker")
        if config.get("mode") in {"direction", "champion"} or config.get("direction_weight"):
            groups.append("official_direction")
        output.append({"candidate_id": candidate_id, "name": f"aggressive_{name}", "config": config, "family": "aggressive_feedback_push", "risk_label": FUSION_LABEL, "risk_labels": RISK_LABELS, "trained_feature_groups": groups, "next_manual_upload_target": candidate_id == final_id, "hypothesis": hypothesis})
    return output
