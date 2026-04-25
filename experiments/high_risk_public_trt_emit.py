from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    from experiments.high_risk_public_trt_data import (
        BASE_OUTPUT_URL,
        OUTPUT_LIMIT,
        PINNED_COMMIT,
        RAW_BASE,
        RISK_LABEL,
        SOURCE_LIMIT,
        SUBJECTS,
        base_predictions,
        candidate_values,
        describe,
        item_values,
        load_external,
        ranked_permutations,
        read_csv,
        write_output,
    )
except ModuleNotFoundError:
    from high_risk_public_trt_data import (
        BASE_OUTPUT_URL,
        OUTPUT_LIMIT,
        PINNED_COMMIT,
        RAW_BASE,
        RISK_LABEL,
        SOURCE_LIMIT,
        SUBJECTS,
        base_predictions,
        candidate_values,
        describe,
        item_values,
        load_external,
        ranked_permutations,
        read_csv,
        write_output,
    )


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def source_text(config: dict) -> str:
    return f"""from __future__ import annotations
import argparse,csv,math,ssl,urllib.request
from pathlib import Path
import numpy as np
PINNED_COMMIT={PINNED_COMMIT!r}
RAW_BASE={RAW_BASE!r}
BASE_OUTPUT_URL={BASE_OUTPUT_URL!r}
SUBJECTS={SUBJECTS!r}
CONFIG={config!r}
def rc(path):
    with open(path,newline='',encoding='utf-8-sig') as h: return list(csv.DictReader(h))
def remote_csv(url):
    with urllib.request.urlopen(url,context=ssl._create_unverified_context(),timeout=90) as r: data=r.read().decode('utf-8-sig').splitlines()
    return list(csv.DictReader(data))
def fetch(subject,cache):
    cache.mkdir(parents=True,exist_ok=True); path=cache/(f"words_dict_romanian_{{int(subject):03d}}.csv")
    if not path.exists():
        with urllib.request.urlopen(f"{{RAW_BASE}}/words_dict_romanian_{{int(subject):03d}}.csv",context=ssl._create_unverified_context(),timeout=90) as r: path.write_bytes(r.read())
    return path
def ext(cache):
    out={{}}
    for s in SUBJECTS: out[s]={{r['word_id']:float(r['fixations_TRT']) for r in rc(fetch(s,cache)) if r.get('word_id') and r.get('fixations_TRT') not in (None,'')}}
    return out
def vals(e,wid): return [v[wid] for v in e.values() if wid in v]
def med(v):
    if not v: return 0.0
    o=sorted(v); m=len(o)//2; return o[m] if len(o)%2 else (o[m-1]+o[m])/2.0
def ridge(x,y,alpha=500.0):
    x=np.asarray(x,float); y=np.asarray(y,float); means=x.mean(axis=0); scales=x.std(axis=0); scales[scales==0.0]=1.0
    z=(x-means)/scales; d=np.column_stack([np.ones(len(z)),z]); p=np.eye(d.shape[1])*alpha; p[0,0]=0.0
    c=np.linalg.solve(d.T@d+p,d.T@y); slopes=c[1:]/scales; intercept=c[0]-float(np.sum(c[1:]*means/scales)); return [float(a) for a in slopes],float(intercept)
def cal(train,e):
    ix=[]; iy=[]; dx=[]; dy=[]
    for r in train:
        v=vals(e,r['word_id'])
        if not v: continue
        l=float(len(r['word'])); ix.append([float(np.mean(v)),med(v),l,math.log1p(l)]); iy.append(float(r['answer']))
        for a in v: dx.append([float(a),l,math.log1p(l)]); dy.append(float(r['answer']))
    islp,iint=ridge(ix,iy); dslp,dint=ridge(dx,dy); return {{'item_slopes':islp,'item_intercept':iint,'direct_slopes':dslp,'direct_intercept':dint}}
def ip(row,e,c):
    v=vals(e,row['word_id'])
    if not v: return 0.0
    l=float(len(row['word'])); f=[float(np.mean(v)),med(v),l,math.log1p(l)]
    return float(c['item_intercept']+sum(a*b for a,b in zip(c['item_slopes'],f)))
def dp(row,e,m,c=None):
    v=vals(e,row['word_id']); fallback=float(np.mean(v)) if v else 0.0; raw=e.get(m[row['participant_id']],{{}}).get(row['word_id'],fallback)
    if c is None: return raw
    l=float(len(row['word'])); f=[float(raw),l,math.log1p(l)]
    return float(c['direct_intercept']+sum(a*b for a,b in zip(c['direct_slopes'],f)))
def base_preds(path,test):
    rows=rc(path) if path and Path(path).exists() else remote_csv(BASE_OUTPUT_URL); lookup={{r['datapointID']:float(r['answer']) for r in rows}}; return [lookup[r['datapointID']] for r in test]
def compute(train,test,e,base):
    c=cal(train,e); item=[ip(r,e,c) for r in test]; k=CONFIG['kind']
    if k=='item': out=item
    elif k=='blend': out=[CONFIG['external_weight']*a+(1-CONFIG['external_weight'])*b for a,b in zip(item,base)]
    elif k=='zero_aware':
        z=float(np.mean(np.asarray([float(r['answer']) for r in train])==0.0)); out=[0.0 if vals(e,r['word_id']) and np.mean(np.asarray(vals(e,r['word_id']),float)==0.0)>=z else p for r,p in zip(test,item)]
    elif k=='raw_perm': out=[dp(r,e,CONFIG['mapping'],None) for r in test]
    elif k=='cal_perm': out=[dp(r,e,CONFIG['mapping'],c) for r in test]
    elif k=='ensemble':
        raw=[dp(r,e,CONFIG['raw_mapping'],None) for r in test]; calp=[dp(r,e,CONFIG['cal_mapping'],c) for r in test]; out=[0.4*a+0.3*b+0.3*d for a,b,d in zip(item,raw,calp)]
    else: raise ValueError(k)
    return [min(10000.0,max(0.0,float(v))) for v in out]
def main():
    p=argparse.ArgumentParser(); p.add_argument('--train',type=Path,required=True); p.add_argument('--test',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--base-output',type=Path); a=p.parse_args()
    train=rc(a.train); test=rc(a.test); e=ext(Path('.cache/high_risk_public_trt')); base=base_preds(a.base_output,test); pred=compute(train,test,e,base)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w',newline='',encoding='utf-8') as h:
        w=csv.DictWriter(h,fieldnames=['subtaskID','datapointID','answer'],lineterminator='\\n'); w.writeheader()
        for r,v in zip(test,pred): w.writerow({{'subtaskID':'1','datapointID':r['datapointID'],'answer':f"{{v:.6f}}"}})
if __name__=='__main__': main()
"""


def probe(test_path: Path, report_path: Path | None) -> dict:
    test_rows = read_csv(test_path)
    external = load_external(Path(".cache/high_risk_public_trt"))
    unique_ids = sorted({row["word_id"] for row in test_rows})
    coverage = {subject: sum(word_id in external[subject] for word_id in unique_ids) for subject in SUBJECTS}
    item_coverage = sum(bool(item_values(external, word_id)) for word_id in unique_ids)
    result = {"mode": "probe", "risk_label": RISK_LABEL, "pinned_commit": PINNED_COMMIT, "test_unique_word_ids": len(unique_ids), "item_coverage": item_coverage, "item_coverage_rate": item_coverage / len(unique_ids), "subject_coverage": coverage, "source_urls": [f"{RAW_BASE}/words_dict_romanian_{int(subject):03d}.csv" for subject in SUBJECTS]}
    if report_path:
        write_json(report_path, result)
    return result


def validate_candidate(root: Path, candidate: dict, test_ids: list[str]) -> dict:
    output = read_csv(root / candidate["output_file"])
    values = [float(row["answer"]) for row in output]
    row = {**candidate, "source_bytes": (root / candidate["source_file"]).stat().st_size, "output_bytes": (root / candidate["output_file"]).stat().st_size, "row_count": len(output), "columns": list(output[0].keys()) if output else [], "ids_match_test_order": [row["datapointID"] for row in output] == test_ids, "answers_finite_non_negative": all(math.isfinite(value) and value >= 0.0 for value in values), "answer_mean": float(np.mean(values)), "answer_std": float(np.std(values)), "answer_zero_rate": float(np.mean(np.asarray(values) == 0.0))}
    row["upload_ready"] = row["source_bytes"] < SOURCE_LIMIT and row["output_bytes"] < OUTPUT_LIMIT and row["row_count"] == len(test_ids) and row["columns"] == ["subtaskID", "datapointID", "answer"] and row["ids_match_test_order"] and row["answers_finite_non_negative"]
    return row


def candidate_specs(train_rows: list[dict[str, str]], test_rows: list[dict[str, str]], external: dict[str, dict[str, float]], start: int, limit: int) -> tuple[list[dict], dict]:
    raw_rankings = ranked_permutations(train_rows, test_rows, external, calibrated=False)
    cal_rankings = ranked_permutations(train_rows, test_rows, external, calibrated=True)
    base_specs = [("public_trt_item_calibrated", {"kind": "item"}, "Calibrated external public TRT item statistics."), ("public_trt_item_blend25", {"kind": "blend", "external_weight": 0.25}, "Blend 25 percent calibrated public TRT item estimate with Candidate 023."), ("public_trt_item_blend50", {"kind": "blend", "external_weight": 0.50}, "Blend 50 percent calibrated public TRT item estimate with Candidate 023."), ("public_trt_item_blend75", {"kind": "blend", "external_weight": 0.75}, "Blend 75 percent calibrated public TRT item estimate with Candidate 023."), ("public_trt_zero_aware", {"kind": "zero_aware"}, "Use calibrated public TRT item estimate with direct external all-zero item suppression.")]
    specs = [{"candidate_id": f"{start + index:03d}", "name": name, "config": config, "hypothesis": hypothesis, "family": config["kind"]} for index, (name, config, hypothesis) in enumerate(base_specs)]
    next_id = start + len(specs)
    for family, kind, rankings in [("raw_permutation", "raw_perm", raw_rankings), ("calibrated_permutation", "cal_perm", cal_rankings)]:
        for index, row in enumerate(rankings[:8], start=1):
            specs.append({"candidate_id": f"{next_id:03d}", "name": f"public_trt_{kind.replace('_perm', '')}_perm_{index:02d}", "config": {"kind": kind, "mapping": row["mapping"]}, "hypothesis": f"{family.replace('_', ' ').title()} rank {index}.", "family": family, "permutation_rank": index, "permutation_distance": row["distance"]})
            next_id += 1
    specs.append({"candidate_id": f"{next_id:03d}", "name": "public_trt_ensemble_fallback", "config": {"kind": "ensemble", "raw_mapping": raw_rankings[0]["mapping"], "cal_mapping": cal_rankings[0]["mapping"]}, "hypothesis": "Blend calibrated item estimate with the top raw and calibrated public TRT permutations.", "family": "ensemble"})
    return specs[:limit], {"raw_permutation_rankings": raw_rankings, "calibrated_permutation_rankings": cal_rankings}


def materialize(root: Path, spec: dict, train_rows: list[dict[str, str]], test_rows: list[dict[str, str]], external: dict[str, dict[str, float]], base: list[float]) -> dict:
    source = root / "official_candidates" / f"{spec['candidate_id']}_{spec['name']}_source.py"
    output = root / "official_candidates" / f"{spec['candidate_id']}_{spec['name']}_output.csv"
    source.write_text(source_text(spec["config"]), encoding="utf-8")
    write_output(output, test_rows, candidate_values(spec["config"], train_rows, test_rows, external, base))
    return {key: value for key, value in spec.items() if key != "config"} | {"source_file": str(source.relative_to(root)), "output_file": str(output.relative_to(root)), "risk_label": RISK_LABEL, "score_type": "risk_labeled_reconstruction", "pinned_commit": PINNED_COMMIT, "source_urls": [f"{RAW_BASE}/words_dict_romanian_{int(subject):03d}.csv" for subject in SUBJECTS], "reproduction_command": f"python3 {source.relative_to(root)} --train data/train_data.csv --test data/test_data.csv --output {output.relative_to(root)} --base-output official_candidates/023_lexical_transformer_output.csv"}
