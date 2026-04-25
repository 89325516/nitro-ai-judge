from __future__ import annotations
import argparse,csv,math,re,ssl,urllib.request
from pathlib import Path
import numpy as np
PINNED_COMMIT='6c724e8877c52ea021f295216949860f87d1c8b2'
RAW_BASE='https://raw.githubusercontent.com/ana0101/eye-tracking/6c724e8877c52ea021f295216949860f87d1c8b2/trt_model/word_sentence_fixations'
SUBJECTS=['008', '009', '010', '011', '023']
CONFIG={'mode': 'cal', 'weights': {'safe': 0.2, 'medium': 0.15, 'high': 0.65}, 'scale': 1.08, 'mapping': {'040': '008', '036': '010', '016': '011', '024': '023', '019': '009'}, 'risk_labels': ['safe_external_generalization', 'medium_public_behavior_mapping', 'high_risk_public_reconstruction']}
WORD_RE=re.compile(r"^(.*)_([0-9]+)_page_([0-9]+)_([0-9]+)$")
def rc(path):
    with open(path,newline='',encoding='utf-8-sig') as h: return list(csv.DictReader(h))
def fetch(subject,cache):
    cache.mkdir(parents=True,exist_ok=True); p=cache/(f"words_dict_romanian_{int(subject):03d}.csv")
    if not p.exists():
        with urllib.request.urlopen(f"{RAW_BASE}/words_dict_romanian_{int(subject):03d}.csv",context=ssl._create_unverified_context(),timeout=90) as r: p.write_bytes(r.read())
    return p
def pub(cache):
    out=[]
    for s in SUBJECTS:
        for r in rc(fetch(s,cache)):
            if r.get('word_id') and r.get('fixations_TRT') not in (None,''):
                r=dict(r); r['subject_key']=f"{int(s):03d}"; out.append(r)
    return out
def mat(rows):
    m={s:{} for s in SUBJECTS}
    for r in rows: m[r['subject_key']][r['word_id']]=float(r['fixations_TRT'])
    return m
def parse(wid):
    m=WORD_RE.match(wid); return (m.group(1),int(m.group(2)),int(m.group(3)),int(m.group(4))) if m else ('',0,0,0)
def text(r): return r.get('text') or parse(r['word_id'])[0]
def base(path,test):
    rows=rc(path); d={r['datapointID']:float(r['answer']) for r in rows}; return [d[r['datapointID']] for r in test]
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
    d={}
    for r in rows: d.setdefault(text(r),[]).append(float(r['fixations_TRT']))
    return {k:float(np.mean(v)) for k,v in d.items()}
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
    v=vals(m,r['word_id']); raw=m.get(mp[r['participant_id']],{}).get(r['word_id'],float(np.mean(v)) if v else 0.0)
    if cal is None: return max(0.0,float(raw))
    sl,ic=cal; return max(0.0,float(ic+np.dot(sl,[float(raw),len(r['word']),math.log1p(len(r['word']))])))
def high(test,m,hb,cal):
    if CONFIG['mode']=='base' and hb is not None: return [max(0.0,float(v)) for v in hb]
    return [dval(r,m,CONFIG['mapping'],cal if CONFIG['mode'] in ('cal','base') else None) for r in test]
def write(test,out,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as h:
        w=csv.DictWriter(h,fieldnames=['subtaskID','datapointID','answer'],lineterminator='\n'); w.writeheader()
        for r,v in zip(test,out): w.writerow({'subtaskID':'1','datapointID':r['datapointID'],'answer':f"{max(0.0,float(v)):.6f}"})
def main():
    p=argparse.ArgumentParser(); p.add_argument('--train',type=Path,required=True); p.add_argument('--test',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--base-output',type=Path,required=True); p.add_argument('--high-risk-base-output',type=Path); a=p.parse_args()
    train=rc(a.train); test=rc(a.test); b=base(a.base_output,test); hb=base(a.high_risk_base_output,test) if a.high_risk_base_output and a.high_risk_base_output.exists() else None
    rows=pub(Path('.cache/three_tier_external')); m=mat(rows); cal=dcal(train,m); s=safe(train,test,b); md=medium(train,test,rows,m); hr=high(test,m,hb,cal); wt=CONFIG['weights']
    out=[min(10000.0,max(0.0,(wt['safe']*x+wt['medium']*y+wt['high']*z)*CONFIG.get('scale',1.0))) for x,y,z in zip(s,md,hr)]
    write(test,out,a.output)
if __name__=='__main__': main()
