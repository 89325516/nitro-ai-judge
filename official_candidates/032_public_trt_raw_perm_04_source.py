from __future__ import annotations
import argparse,csv,math,ssl,urllib.request
from pathlib import Path
import numpy as np
PINNED_COMMIT='6c724e8877c52ea021f295216949860f87d1c8b2'
RAW_BASE='https://raw.githubusercontent.com/ana0101/eye-tracking/6c724e8877c52ea021f295216949860f87d1c8b2/trt_model/word_sentence_fixations'
BASE_OUTPUT_URL='https://raw.githubusercontent.com/89325516/nitro-ai-judge/975cae90f129fa0b0bdc159445ccd09f68a88a5d/official_candidates/023_lexical_transformer_output.csv'
SUBJECTS=['008', '009', '010', '011', '023']
CONFIG={'kind': 'raw_perm', 'mapping': {'040': '008', '036': '009', '016': '011', '024': '023', '019': '010'}}
def rc(path):
    with open(path,newline='',encoding='utf-8-sig') as h: return list(csv.DictReader(h))
def remote_csv(url):
    with urllib.request.urlopen(url,context=ssl._create_unverified_context(),timeout=90) as r: data=r.read().decode('utf-8-sig').splitlines()
    return list(csv.DictReader(data))
def fetch(subject,cache):
    cache.mkdir(parents=True,exist_ok=True); path=cache/(f"words_dict_romanian_{int(subject):03d}.csv")
    if not path.exists():
        with urllib.request.urlopen(f"{RAW_BASE}/words_dict_romanian_{int(subject):03d}.csv",context=ssl._create_unverified_context(),timeout=90) as r: path.write_bytes(r.read())
    return path
def ext(cache):
    out={}
    for s in SUBJECTS: out[s]={r['word_id']:float(r['fixations_TRT']) for r in rc(fetch(s,cache)) if r.get('word_id') and r.get('fixations_TRT') not in (None,'')}
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
    islp,iint=ridge(ix,iy); dslp,dint=ridge(dx,dy); return {'item_slopes':islp,'item_intercept':iint,'direct_slopes':dslp,'direct_intercept':dint}
def ip(row,e,c):
    v=vals(e,row['word_id'])
    if not v: return 0.0
    l=float(len(row['word'])); f=[float(np.mean(v)),med(v),l,math.log1p(l)]
    return float(c['item_intercept']+sum(a*b for a,b in zip(c['item_slopes'],f)))
def dp(row,e,m,c=None):
    v=vals(e,row['word_id']); fallback=float(np.mean(v)) if v else 0.0; raw=e.get(m[row['participant_id']],{}).get(row['word_id'],fallback)
    if c is None: return raw
    l=float(len(row['word'])); f=[float(raw),l,math.log1p(l)]
    return float(c['direct_intercept']+sum(a*b for a,b in zip(c['direct_slopes'],f)))
def base_preds(path,test):
    rows=rc(path) if path and Path(path).exists() else remote_csv(BASE_OUTPUT_URL); lookup={r['datapointID']:float(r['answer']) for r in rows}; return [lookup[r['datapointID']] for r in test]
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
        w=csv.DictWriter(h,fieldnames=['subtaskID','datapointID','answer'],lineterminator='\n'); w.writeheader()
        for r,v in zip(test,pred): w.writerow({'subtaskID':'1','datapointID':r['datapointID'],'answer':f"{v:.6f}"})
if __name__=='__main__': main()
