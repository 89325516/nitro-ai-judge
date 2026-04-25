from __future__ import annotations

BEST_ID = "159"
BEST_SCORE = 38.87076
BEST_SUBMISSION_ID = "manual_159_score_38_87076"
PREVIOUS_BEST_ID = "099"
PREVIOUS_BEST_SCORE = 38.20618
FUSION_LABEL = "noise_aware_scaled_external_denoising"
RISK_LABELS = ["clean_external_aggregate_prior", "scaled_mac_model", "train_label_denoising"]


def external_rows() -> list[tuple[str, dict, str]]:
    rows = []
    for idx, (target, weight, calib) in enumerate([
        ("median", .06, "ridge"), ("trim10", .08, "ridge"), ("winsor10", .10, "ridge"), ("zeroaware", .10, "ridge"), ("positive", .08, "ridge"),
        ("median", .12, "direct"), ("trim20", .12, "ridge"), ("winsor20", .14, "ridge"), ("zeroaware", .14, "direct"), ("positive", .12, "ridge"),
    ], start=1):
        rows.append((f"external_clean_{idx:02d}", {"mode": "external", "target": target, "weight": weight, "calibration": calib}, f"Blend clean public aggregate TRT prior calibrated to {target}."))
    return rows


def scaled_rows() -> list[tuple[str, dict, str]]:
    rows = []
    for idx, (model, target, weight, trees) in enumerate([
        ("ridge", "median", .08, 0), ("ridge", "zeroaware", .10, 0), ("hgb", "trim10", .10, 0), ("hgb", "winsor10", .12, 0), ("trees", "median", .10, 96),
        ("trees", "trim10", .12, 128), ("trees", "zeroaware", .14, 160), ("hgb", "positive", .12, 0), ("trees", "winsor20", .16, 192), ("ridge", "positive", .12, 0),
    ], start=1):
        rows.append((f"scaled_{model}_{idx:02d}", {"mode": "scaled", "model": model, "target": target, "weight": weight, "trees": trees}, f"Mac-scaled {model} trained on denoised {target} labels with clean external features."))
    return rows


def eb_rows() -> list[tuple[str, dict, str]]:
    rows = []
    for idx, (prior, strength, weight, model) in enumerate([
        ("external", 8.0, .08, "ridge"), ("text", 8.0, .10, "ridge"), ("page", 10.0, .10, "ridge"), ("hybrid", 12.0, .12, "hgb"), ("external", 18.0, .12, "hgb"),
        ("text", 18.0, .14, "hgb"), ("page", 20.0, .14, "trees"), ("hybrid", 22.0, .16, "trees"), ("external", 30.0, .18, "trees"), ("hybrid", 30.0, .18, "hgb"),
    ], start=1):
        rows.append((f"eb_external_{idx:02d}", {"mode": "eb", "prior": prior, "strength": strength, "weight": weight, "model": model, "trees": 128}, f"Empirical-Bayes denoising toward {prior} prior with {model}."))
    return rows


def hurdle_rows() -> list[tuple[str, dict, str]]:
    rows = []
    for idx, (positive, zero_weight, weight, model) in enumerate([
        ("log", .50, .08, "ridge"), ("log", .70, .10, "hgb"), ("median", .60, .10, "ridge"), ("trim10", .65, .12, "hgb"), ("winsor10", .70, .12, "trees"),
        ("positive", .75, .14, "hgb"), ("external", .65, .14, "ridge"), ("hybrid", .70, .16, "hgb"), ("hybrid", .80, .16, "trees"),
    ], start=1):
        rows.append((f"hurdle_scaled_{idx:02d}", {"mode": "hurdle", "positive": positive, "zero_weight": zero_weight, "weight": weight, "model": model, "trees": 128}, f"Zero-inflated hurdle using {positive} positive target and {model}."))
    return rows


def champion_row() -> tuple[str, dict, str]:
    config = {"mode": "champion", "external_weight": .06, "scaled_weight": .08, "eb_weight": .08, "hurdle_weight": .06, "scale": 1.0}
    return "champion_scaled_external_denoising", config, "Champion: Mac-scaled model plus clean external aggregate prior and train-label denoising residuals."


def specs(start: int) -> list[dict]:
    rows = external_rows() + scaled_rows() + eb_rows() + hurdle_rows() + [champion_row()]
    final_id = f"{start + len(rows) - 1:03d}"
    out = []
    for offset, (name, config, hypothesis) in enumerate(rows):
        cid = f"{start + offset:03d}"
        groups = ["official_best_base", "train_label_noise_profile", "clean_external_aggregate_prior"]
        if config["mode"] in {"scaled", "eb", "hurdle", "champion"}: groups.append("mac_scaled_model")
        if config["mode"] in {"eb", "champion"}: groups.append("empirical_bayes_shrinkage")
        if config["mode"] in {"hurdle", "champion"}: groups.append("zero_hurdle")
        out.append({"candidate_id": cid, "name": f"noise_{name}", "config": config, "family": "noise_aware_scaled_external", "risk_label": FUSION_LABEL, "risk_labels": RISK_LABELS, "noise_evidence_groups": groups, "next_manual_upload_target": cid == final_id, "hypothesis": hypothesis})
    return out
