from __future__ import annotations

ANCHOR_ID = "159"
ANCHOR_SCORE = 38.87076
BEST_ID = "199"
BEST_SCORE = 53.8099243
BEST_SUBMISSION_ID = "manual_199_score_53_8099243"
FAMILY = "noise_aware_v2_directional"
RISK_LABEL = "noise_aware_v2_directional_denoising"
RISK_LABELS = ["official_winning_direction", "clean_external_aggregate_prior", "scaled_mac_model", "train_label_denoising"]


def direction_rows() -> list[tuple[str, dict, str]]:
    rows = []
    pairs = [(.20, "raw"), (.35, "winsor"), (.50, "shrink"), (.65, "raw"), (.80, "winsor"), (1.00, "shrink"), (1.20, "raw"), (1.50, "winsor"), (.35, "zboost"), (.65, "zboost"), (1.00, "winsor"), (1.20, "shrink")]
    for idx, (gamma, style) in enumerate(pairs, start=1):
        rows.append((f"direction_{idx:02d}", {"mode": "direction", "gamma": gamma, "style": style}, f"Official 159-to-199 direction extrapolation gamma={gamma} style={style}."))
    return rows


def scaled_rows() -> list[tuple[str, dict, str]]:
    rows = []
    items = [("hgb", "trim10", .06, 260), ("hgb", "winsor10", .08, 300), ("hgb", "zeroaware", .10, 320), ("trees", "median", .06, 256), ("trees", "trim10", .08, 320), ("trees", "winsor20", .10, 384), ("ridge", "median", .08, 0), ("ridge", "positive", .10, 0), ("huber", "trim10", .08, 0), ("huber", "zeroaware", .10, 0), ("hgb", "positive", .08, 280), ("trees", "zeroaware", .08, 320)]
    for idx, (model, target, weight, scale) in enumerate(items, start=1):
        rows.append((f"scaled_{model}_{idx:02d}", {"mode": "scaled", "model": model, "target": target, "weight": weight, "scale_param": scale, "gamma": .35}, f"Mac-scaled {model} denoised stacker target={target}."))
    return rows


def external_rows() -> list[tuple[str, dict, str]]:
    rows = []
    items = [("external", 10.0, .06, "ridge"), ("hybrid", 14.0, .08, "ridge"), ("text", 18.0, .08, "hgb"), ("page", 18.0, .08, "hgb"), ("external", 24.0, .10, "hgb"), ("hybrid", 24.0, .10, "trees"), ("family", 20.0, .08, "ridge"), ("external", 32.0, .12, "trees"), ("hybrid", 32.0, .12, "hgb"), ("page", 28.0, .10, "trees"), ("text", 28.0, .10, "ridge"), ("hybrid", 40.0, .12, "trees")]
    for idx, (prior, strength, weight, model) in enumerate(items, start=1):
        rows.append((f"external_consensus_{idx:02d}", {"mode": "external", "prior": prior, "strength": strength, "weight": weight, "model": model, "gamma": .35}, f"Low-noise external consensus prior={prior} strength={strength}."))
    return rows


def fused_rows() -> list[tuple[str, dict, str]]:
    rows = []
    gammas = [.35, .50, .65, .80, 1.00, 1.20, .50, .65, .80, 1.00, 1.20, 1.50]
    for idx, gamma in enumerate(gammas, start=1):
        rows.append((f"fused_{idx:02d}", {"mode": "fused", "gamma": gamma, "scaled_weight": .04 + .005 * (idx % 3), "external_weight": .035 + .005 * (idx % 4), "eb_weight": .035, "hurdle_weight": .025, "style": "winsor" if idx % 2 else "raw"}, f"Direction-aware fusion gamma={gamma} with denoised residual components."))
    return rows


def specs(start: int) -> list[dict]:
    rows = [("anchor_reproduction", {"mode": "anchor"}, "Reproduce Candidate 199 official champion output.")]
    rows += direction_rows() + scaled_rows() + external_rows() + fused_rows()
    rows.append(("champion_directional_denoising", {"mode": "champion", "fallback_gammas": [.65, .50, .80, 1.00, .35], "scaled_weight": .045, "external_weight": .04, "eb_weight": .035, "hurdle_weight": .025, "style": "winsor"}, "Champion: fixed 199 plus validated 159-to-199 direction and denoised residual fusion."))
    final_id = f"{start + len(rows) - 1:03d}"
    out = []
    for offset, (name, config, hypothesis) in enumerate(rows):
        cid = f"{start + offset:03d}"
        groups = ["official_best_base", "official_winning_direction"]
        if config["mode"] in {"scaled", "external", "fused", "champion"}: groups += ["train_label_denoising", "clean_external_aggregate_prior"]
        if config["mode"] in {"scaled", "fused", "champion"}: groups.append("mac_scaled_model")
        if config["mode"] in {"external", "fused", "champion"}: groups.append("empirical_bayes_shrinkage")
        if config["mode"] in {"fused", "champion"}: groups.append("zero_hurdle")
        component_weights = {"direction_gamma": config.get("gamma", config.get("fallback_gammas", [0.0])[0] if config.get("mode") == "champion" else 0.0), "scaled": config.get("scaled_weight", config.get("weight", 0.0) if config.get("mode") == "scaled" else 0.0), "external": config.get("external_weight", config.get("weight", 0.0) if config.get("mode") == "external" else 0.0), "eb": config.get("eb_weight", 0.0), "hurdle": config.get("hurdle_weight", 0.0)}
        out.append({"candidate_id": cid, "name": f"noise_v2_{name}", "config": config, "family": FAMILY, "risk_label": RISK_LABEL, "risk_labels": RISK_LABELS, "noise_evidence_groups": groups, "component_weights": component_weights, "next_manual_upload_target": cid == final_id, "hypothesis": hypothesis})
    return out
