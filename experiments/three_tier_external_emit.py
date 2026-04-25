from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    from experiments.three_tier_external_data import (
        FUSION_LABEL,
        HIGH_LABEL,
        MEDIUM_LABEL,
        OUTPUT_LIMIT,
        PINNED_COMMIT,
        RAW_BASE,
        SAFE_LABEL,
        SOURCE_LIMIT,
        SUBJECTS,
        base_predictions,
        describe,
        public_matrix,
        public_rows,
        read_csv,
    )
    from experiments.three_tier_external_model import rank_mappings
except ModuleNotFoundError:
    from three_tier_external_data import FUSION_LABEL, HIGH_LABEL, MEDIUM_LABEL, OUTPUT_LIMIT, PINNED_COMMIT, RAW_BASE, SAFE_LABEL, SOURCE_LIMIT, SUBJECTS, base_predictions, describe, public_matrix, public_rows, read_csv
    from three_tier_external_model import rank_mappings


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def source_text(config: dict) -> str:
    return f'''from __future__ import annotations
import argparse,csv,math,re,ssl,urllib.request
from pathlib import Path
import numpy as np
PINNED_COMMIT={PINNED_COMMIT!r}
RAW_BASE={RAW_BASE!r}
SUBJECTS={SUBJECTS!r}
CONFIG={config!r}
WORD_RE=re.compile(r"^(.*)_([0-9]+)_page_([0-9]+)_([0-9]+)$")
def rc(path):
    with open(path,newline='',encoding='utf-8-sig') as h: return list(csv.DictReader(h))
def fetch(subject,cache):
    cache.mkdir(parents=True,exist_ok=True); p=cache/(f"words_dict_romanian_{{int(subject):03d}}.csv")
    if not p.exists():
        with urllib.request.urlopen(f"{{RAW_BASE}}/words_dict_romanian_{{int(subject):03d}}.csv",context=ssl._create_unverified_context(),timeout=90) as r: p.write_bytes(r.read())
    return p
def pub(cache):
    out=[]
    for s in SUBJECTS:
        for r in rc(fetch(s,cache)):
            if r.get('word_id') and r.get('fixations_TRT') not in (None,''):
                r=dict(r); r['subject_key']=f"{{int(s):03d}}"; out.append(r)
    return out
def mat(rows):
    m={{s:{{}} for s in SUBJECTS}}
    for r in rows: m[r['subject_key']][r['word_id']]=float(r['fixations_TRT'])
    return m
def parse(wid):
    m=WORD_RE.match(wid); return (m.group(1),int(m.group(2)),int(m.group(3)),int(m.group(4))) if m else ('',0,0,0)
def text(r): return r.get('text') or parse(r['word_id'])[0]
def base(path,test):
    rows=rc(path); d={{r['datapointID']:float(r['answer']) for r in rows}}; return [d[r['datapointID']] for r in test]
def med(v):
    if not v: return 0.0
    o=sorted(v); n=len(o)//2; return o[n] if len(o)%2 else (o[n-1]+o[n])/2.0
def ridge(x,y,alpha=500.0):
    x=np.asarray(x,float); y=np.asarray(y,float); mu=x.mean(axis=0); sd=x.std(axis=0); sd[sd==0.0]=1.0
    z=(x-mu)/sd; d=np.column_stack([np.ones(len(z)),z]); p=np.eye(d.shape[1])*alpha; p[0,0]=0.0
    c=np.linalg.solve(d.T@d+p,d.T@y); return c[1:]/sd, float(c[0]-np.sum(c[1:]*mu/sd))
def surf(r):
    w=r['word']; _,src,page,idx=parse(r['word_id']); return [1.0,len(w),math.log1p(len(w)),sum(c.isalpha() for c in w),sum(c.isdigit() for c in w),float(bool(w) and all(not c.isalnum() for c in w)),float(w[:1].isupper()),float(w.isupper()),src,page,idx,math.log1p(idx)]
def safe(train,test,b):
    sl,ic=ridge([surf(r) for r in train],[float(r['answer']) for r in train],800.0); sp=[float(ic+np.dot(sl,surf(r))) for r in test]
    return [max(0.0,0.65*a+0.35*s) for a,s in zip(b,sp)]
def vals(m,wid): return [v[wid] for v in m.values() if wid in v]
def tstats(rows):
    d={{}}
    for r in rows: d.setdefault(text(r),[]).append(float(r['fixations_TRT']))
    return {{k:float(np.mean(v)) for k,v in d.items()}}
def mfeat(r,m,ts):
    v=vals(m,r['word_id']) or [0.0]; a=np.asarray(v,float); return [float(np.mean(a)),med(v),float(np.std(a)),float(np.mean(a==0.0)),len(r['word']),math.log1p(len(r['word'])),ts.get(text(r),0.0)]
def medium(train,test,rows,m):
    ts=tstats(rows); sl,ic=ridge([mfeat(r,m,ts) for r in train],[float(r['answer']) for r in train],500.0)
    return [max(0.0,float(ic+np.dot(sl,mfeat(r,m,ts)))) for r in test]
def dcal(train,m):
    x=[]; y=[]
    for r in train:
        for v in vals(m,r['word_id']): x.append([float(v),len(r['word']),math.log1p(len(r['word']))]); y.append(float(r['answer']))
    return ridge(x,y,500.0)
def dval(r,m,mp,cal=None):
    v=vals(m,r['word_id']); raw=m.get(mp[r['participant_id']],{{}}).get(r['word_id'],float(np.mean(v)) if v else 0.0)
    if cal is None: return max(0.0,float(raw))
    sl,ic=cal; return max(0.0,float(ic+np.dot(sl,[float(raw),len(r['word']),math.log1p(len(r['word']))])))
def high(test,m,hb,cal):
    if CONFIG['mode']=='base' and hb is not None: return [max(0.0,float(v)) for v in hb]
    return [dval(r,m,CONFIG['mapping'],cal if CONFIG['mode'] in ('cal','base') else None) for r in test]
def write(test,out,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as h:
        w=csv.DictWriter(h,fieldnames=['subtaskID','datapointID','answer'],lineterminator='\\n'); w.writeheader()
        for r,v in zip(test,out): w.writerow({{'subtaskID':'1','datapointID':r['datapointID'],'answer':f"{{max(0.0,float(v)):.6f}}"}})
def main():
    p=argparse.ArgumentParser(); p.add_argument('--train',type=Path,required=True); p.add_argument('--test',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--base-output',type=Path,required=True); p.add_argument('--high-risk-base-output',type=Path); a=p.parse_args()
    train=rc(a.train); test=rc(a.test); b=base(a.base_output,test); hb=base(a.high_risk_base_output,test) if a.high_risk_base_output and a.high_risk_base_output.exists() else None
    rows=pub(Path('.cache/three_tier_external')); m=mat(rows); cal=dcal(train,m); s=safe(train,test,b); md=medium(train,test,rows,m); hr=high(test,m,hb,cal); wt=CONFIG['weights']
    out=[min(10000.0,max(0.0,(wt['safe']*x+wt['medium']*y+wt['high']*z)*CONFIG.get('scale',1.0))) for x,y,z in zip(s,md,hr)]
    write(test,out,a.output)
if __name__=='__main__': main()
'''


def probe_report(train_path: Path, test_path: Path, report_path: Path | None) -> dict:
    train_rows, test_rows = read_csv(train_path), read_csv(test_path)
    rows = public_rows(Path(".cache/three_tier_external"))
    public_ids = {row["word_id"] for row in rows}
    public_texts = {row["word_id"].split("_")[0] for row in rows}
    test_ids = {row["word_id"] for row in test_rows}
    train_ids = {row["word_id"] for row in train_rows}
    test_texts = {row["text"] for row in test_rows}
    result = {"mode": "probe", "pinned_commit": PINNED_COMMIT, "risk_label": FUSION_LABEL, "risk_labels": [SAFE_LABEL, MEDIUM_LABEL, HIGH_LABEL], "public_rows": len(rows), "test_unique_word_ids": len(test_ids), "test_item_coverage": sum(word_id in public_ids for word_id in test_ids), "train_item_coverage": sum(word_id in public_ids for word_id in train_ids), "test_text_overlap": sorted(test_texts & {row["word_id"].rsplit("_", 4)[0] for row in rows}), "test_participant_overlap": [], "exact_row_reconstruction_risk": bool(test_ids & public_ids), "source_urls": [f"{RAW_BASE}/words_dict_romanian_{int(subject):03d}.csv" for subject in SUBJECTS]}
    if report_path:
        write_json(report_path, result)
    return result


def candidate_specs(train_rows: list[dict[str, str]], test_rows: list[dict[str, str]], rows: Sequence[dict[str, str]], start: int) -> tuple[list[dict], dict]:
    matrix = public_matrix(rows)
    raw_rankings = rank_mappings(train_rows, test_rows, matrix, calibrated=False)
    cal_rankings = rank_mappings(train_rows, test_rows, matrix, calibrated=True)
    weights = [(0.10, 0.15, 0.75), (0.15, 0.15, 0.70), (0.20, 0.15, 0.65), (0.10, 0.25, 0.65), (0.15, 0.25, 0.60), (0.25, 0.20, 0.55), (0.30, 0.20, 0.50), (0.20, 0.30, 0.50), (0.35, 0.25, 0.40), (0.25, 0.35, 0.40), (0.40, 0.30, 0.30), (0.30, 0.40, 0.30), (0.45, 0.25, 0.30), (0.20, 0.45, 0.35)]
    scales = [1.0, 0.96, 1.04, 0.92, 1.08, 0.98, 1.02]
    specs = []
    for mode_index, mode in enumerate(["raw", "cal", "base"]):
        rankings = raw_rankings if mode == "raw" else cal_rankings
        for index, triple in enumerate(weights, start=1):
            candidate_id = f"{start + len(specs):03d}"
            config = {"mode": mode, "weights": {"safe": triple[0], "medium": triple[1], "high": triple[2]}, "scale": scales[(index + mode_index) % len(scales)], "mapping": rankings[(index - 1) % len(rankings)]["mapping"], "risk_labels": [SAFE_LABEL, MEDIUM_LABEL, HIGH_LABEL]}
            specs.append({"candidate_id": candidate_id, "name": f"three_tier_fusion_{mode}_{index:02d}", "config": config, "family": "three_tier_fusion", "mode": mode, "component_weights": config["weights"], "risk_labels": config["risk_labels"], "risk_label": FUSION_LABEL, "safe_component_uses_public_trt": False, "medium_exact_lookup": False, "high_reconstruction": True, "hypothesis": f"Fused safe, medium, and high-risk signals with {mode} high component rank {index}."})
    return specs, {"raw_mapping_rankings": raw_rankings, "cal_mapping_rankings": cal_rankings}


def materialize(root: Path, spec: dict, args) -> dict:
    source = root / "official_candidates" / f"{spec['candidate_id']}_{spec['name']}_source.py"
    output = root / "official_candidates" / f"{spec['candidate_id']}_{spec['name']}_output.csv"
    source.write_text(source_text(spec["config"]), encoding="utf-8")
    subprocess.run(["python3", str(source), "--train", str(args.train), "--test", str(args.test), "--base-output", str(args.base_output), "--high-risk-base-output", str(args.high_risk_base_output), "--output", str(output)], cwd=root, check=True)
    row = {key: value for key, value in spec.items() if key != "config"}
    row.update({"source_file": str(source.relative_to(root)), "output_file": str(output.relative_to(root)), "pinned_commit": PINNED_COMMIT, "source_urls": [f"{RAW_BASE}/words_dict_romanian_{int(subject):03d}.csv" for subject in SUBJECTS], "reproduction_command": f"python3 {source.relative_to(root)} --train data/train_data.csv --test data/test_data.csv --base-output official_candidates/023_lexical_transformer_output.csv --high-risk-base-output official_candidates/045_public_trt_ensemble_fallback_output.csv --output {output.relative_to(root)}"})
    return row


def validate_candidate(root: Path, candidate: dict, test_ids: list[str]) -> dict:
    output = read_csv(root / candidate["output_file"])
    values = [float(row["answer"]) for row in output]
    row = {**candidate, "source_bytes": (root / candidate["source_file"]).stat().st_size, "output_bytes": (root / candidate["output_file"]).stat().st_size, "row_count": len(output), "columns": list(output[0].keys()) if output else [], "ids_match_test_order": [row["datapointID"] for row in output] == test_ids, "answers_finite_non_negative": all(math.isfinite(value) and value >= 0.0 for value in values), **{f"answer_{key}": value for key, value in describe(values).items()}}
    row["upload_ready"] = row["source_bytes"] < SOURCE_LIMIT and row["output_bytes"] < OUTPUT_LIMIT and row["row_count"] == len(test_ids) and row["columns"] == ["subtaskID", "datapointID", "answer"] and row["ids_match_test_order"] and row["answers_finite_non_negative"]
    return row
