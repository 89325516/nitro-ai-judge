from __future__ import annotations

from experiments.high_risk_reconstruction_v4_components import mixture_from_ranked, rank_mappings
from experiments.high_risk_reconstruction_v4_contracts import CandidateSpec, RuntimeContext
from experiments.high_risk_reconstruction_v4_data import first_seen_participants

FAMILY = "high_risk_participant_reconstruction_v4"


def simple_config(source: str, mode: str, **values) -> dict:
    config = {"mode": mode, "source": source, "zero_strategy": "none", "movement_weight": 1.0}
    config.update(values)
    return config


def make_spec(cid: int, name: str, config: dict, labels: tuple[str, ...], mapping_mode: str, zero: str, scale: str, hypothesis: str, final_id: int) -> CandidateSpec:
    return CandidateSpec(str(cid), name, config, FAMILY, hypothesis, labels, mapping_mode, zero, scale, cid == final_id)


def build_specs(ctx: RuntimeContext, start: int, limit: int) -> list[CandidateSpec]:
    final_id = start + limit - 1
    ranked = rank_mappings(ctx)
    participants = first_seen_participants(ctx.test_rows)
    specs: list[CandidateSpec] = []
    cid = start
    for index, row in enumerate(ranked[:20], start=1):
        cfg = simple_config("hard", "single", mapping=row["mapping"], movement_weight=0.85 + 0.03 * (index % 6), target_std=155 + 5 * (index % 7))
        specs.append(make_spec(cid, f"v4_hard_perm_{index:02d}", cfg, ("hard_mapping", "row_trt"), "hard", "none", "zmatch_scale", f"Hard subject permutation rank {index}.", final_id)); cid += 1
    soft_rows = [(k, s) for k in (3, 5, 8, 12, 20) for s in (0.45, 0.60, 0.75, 0.90)]
    for index, (top_k, strength) in enumerate(soft_rows[:20], start=1):
        weights = mixture_from_ranked(ranked, participants, top_k, strength)
        cfg = simple_config("soft", "single", weights=weights, movement_weight=0.95, target_std=165 + 4 * (index % 5))
        specs.append(make_spec(cid, f"v4_soft_mix_{index:02d}", cfg, ("soft_mixture", "row_trt"), "soft", "none", "zmatch_scale", f"Soft participant mixture top={top_k} strength={strength}.", final_id)); cid += 1
    for index, row in enumerate(ranked[:5], start=1):
        for threshold in (0.20, 0.40, 0.60):
            cfg = simple_config("hard", "single", mapping=row["mapping"], movement_weight=1.0, target_std=180, zero_strategy="hybrid", zero_threshold=threshold)
            specs.append(make_spec(cid, f"v4_zero_hybrid_{index}_{int(threshold*100):02d}", cfg, ("hard_mapping", "zero_reconstruction"), "hard", "hybrid", "zmatch_scale", f"Hybrid zero reconstruction threshold={threshold}.", final_id)); cid += 1
    soft_ref = mixture_from_ranked(ranked, participants, 8, 0.75)
    tail_sources = [("anchor", {}), ("hard", {"mapping": ranked[0]["mapping"]}), ("soft", {"weights": soft_ref})]
    for target_std in (175, 195, 215, 235, 255):
        for source, extra in tail_sources:
            cfg = simple_config(source, "single", movement_weight=1.0, target_std=target_std, tail_boost=0.22, tail_quantile=0.88, **extra)
            specs.append(make_spec(cid, f"v4_tail_{source}_{target_std}", cfg, ("tail_restore", source), "none" if source == "anchor" else source, "none", "tail_scale", f"Restore tail with target std {target_std} from {source} source.", final_id)); cid += 1
    hard = simple_config("hard", "single", mapping=ranked[0]["mapping"], movement_weight=1.05, target_std=190)
    soft = simple_config("soft", "single", weights=soft_ref, movement_weight=1.05, target_std=190)
    zero = simple_config("hard", "single", mapping=ranked[1]["mapping"], movement_weight=1.0, target_std=190, zero_strategy="hybrid", zero_threshold=0.40)
    tail = simple_config("soft", "single", weights=soft_ref, movement_weight=1.0, target_std=245, tail_boost=0.28, tail_quantile=0.86)
    fusion_weights = [(0.35, 0.25, 0.15, 0.25), (0.25, 0.35, 0.20, 0.20), (0.25, 0.20, 0.30, 0.25), (0.20, 0.25, 0.20, 0.35), (0.30, 0.25, 0.20, 0.25), (0.20, 0.35, 0.15, 0.30), (0.30, 0.20, 0.30, 0.20), (0.18, 0.32, 0.20, 0.30), (0.22, 0.28, 0.25, 0.25), (0.20, 0.30, 0.20, 0.30)]
    for index, weights in enumerate(fusion_weights, start=1):
        cfg = {"mode": "fusion", "source": "fusion", "parts": [hard, soft, zero, tail], "weights": weights, "target_std": 200 + 5 * index}
        specs.append(make_spec(cid, f"v4_champion_fusion_{index:02d}", cfg, ("hard_mapping", "soft_mixture", "zero_reconstruction", "tail_restore"), "fusion", "hybrid", "fusion_scale", f"Fusion champion rank {index} combining reconstruction components.", final_id)); cid += 1
    return specs[:limit]
