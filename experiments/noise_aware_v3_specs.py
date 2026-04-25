from __future__ import annotations

BEST_ID = "199"
BEST_SCORE = 53.8099243
BEST_SUBMISSION_ID = "manual_199_score_53_8099243"
NEUTRAL_ID = "249"
NEUTRAL_SCORE = 53.8099243
NEUTRAL_SUBMISSION_ID = "manual_249_score_53_8099243"
FAMILY = "noise_aware_v3_low_noise"
RISK_LABEL = "noise_aware_v3_low_noise_denoising"
RISK_LABELS = ["low_noise_external_consensus", "scaled_mac_model", "robust_noisy_label_training", "official_best_base"]


def rows() -> list[tuple[str, dict, str, list[str]]]:
    clean = [("posterior", "ridge", .30), ("median_of_means", "huber", .32), ("huber", "hgb", .34), ("winsor_positive", "ridge", .28), ("zero_hurdle", "hgb", .30)]
    capacity = [("posterior", "hgb", 520, .34), ("zero_hurdle", "hgb", 650, .34), ("posterior", "trees", 512, .30), ("huber", "trees", 640, .32), ("median_of_means", "trees", 768, .34), ("winsor_positive", "hgb", 700, .30)]
    robust = [("ordered", "posterior", "hgb", 520), ("co_teach", "huber", "hgb", 520), ("distributional", "zero_hurdle", "trees", 512), ("superlearner", "posterior", "blend", 0), ("ngboost_like", "winsor_positive", "hgb", 600)]
    out: list[tuple[str, dict, str, list[str]]] = []
    for i, (target, model, weight) in enumerate(clean, 1):
        out.append((f"clean_target_{i:02d}", {"mode": "clean", "target": target, "model": model, "weight": weight, "target_mae": 6.2 + i * .35}, f"Low-noise external consensus target={target} with {model} residual.", ["low_noise_target", "external_consensus"]))
    for i, (target, model, cap, weight) in enumerate(capacity, 1):
        out.append((f"capacity_{model}_{i:02d}", {"mode": "capacity", "target": target, "model": model, "capacity": cap, "weight": weight, "target_mae": 7.0 + i * .35}, f"High-capacity {model} capacity={cap} target={target}.", ["high_capacity_model", "train_label_denoising"]))
    for i, (method, target, model, cap) in enumerate(robust, 1):
        out.append((f"robust_{method}_{i:02d}", {"mode": "robust", "method": method, "target": target, "model": model, "capacity": cap, "target_mae": 7.4 + i * .35}, f"Robust noisy-label method={method} target={target}.", ["robust_training", method]))
    for i, style in enumerate(["clean_heavy", "model_heavy", "uncertainty_heavy"], 1):
        out.append((f"fusion_{style}_{i:02d}", {"mode": "fusion", "style": style, "target_mae": 8.2 + i * .35}, f"V3 fusion style={style} around Candidate 199.", ["superlearner_fusion", "uncertainty_shrinkage"]))
    out.append(("champion_low_noise_stack", {"mode": "champion", "target_mae": 8.8}, "Champion: Candidate 199 plus low-noise target, high-capacity stacker, and uncertainty shrinkage.", ["champion", "low_noise_target", "high_capacity_model", "uncertainty_shrinkage"]))
    return out


def specs(start: int) -> list[dict]:
    all_rows = rows()
    final_id = f"{start + len(all_rows) - 1:03d}"
    out = []
    for offset, (name, config, hypothesis, methods) in enumerate(all_rows):
        cid = f"{start + offset:03d}"
        weights = {"clean": config.get("weight", .30) if config["mode"] in {"clean", "capacity"} else .28, "model": .34 if config["mode"] in {"capacity", "fusion", "champion"} else .18, "robust": .24 if config["mode"] in {"robust", "fusion", "champion"} else .08, "neutral_avoidance": .18 if config["mode"] in {"fusion", "champion"} else .08}
        out.append({"candidate_id": cid, "name": f"noise_v3_{name}", "config": config, "family": FAMILY, "risk_label": RISK_LABEL, "risk_labels": RISK_LABELS, "method_labels": methods, "component_weights": weights, "next_manual_upload_target": cid == final_id, "hypothesis": hypothesis})
    return out
