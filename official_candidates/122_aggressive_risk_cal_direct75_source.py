from __future__ import annotations
import argparse,csv,math,re,ssl,urllib.request
from pathlib import Path
import numpy as np
PINNED_COMMIT='6c724e8877c52ea021f295216949860f87d1c8b2'; RAW_BASE='https://raw.githubusercontent.com/ana0101/eye-tracking/6c724e8877c52ea021f295216949860f87d1c8b2/trt_model/word_sentence_fixations'; SUBJECTS=['008', '009', '010', '011', '023']; CONFIG={'mode': 'blend', 'high_variant': 'cal_direct', 'weights': {'safe': 0.05, 'medium': 0.2, 'high': 0.75}, 'scale': 1.0, 'mapping': {'040': '010', '036': '011', '016': '008', '024': '023', '019': '009'}}
WORD_RE=re.compile(r"^(.*)_([0-9]+)_page_([0-9]+)_([0-9]+)$"); FAMS=("arg","enc","ins","lit","popsci")
def rc(p):
    with open(p,newline='',encoding='utf-8-sig') as h: return list(csv.DictReader(h))
def fetch(name,cache):
    cache.mkdir(parents=True,exist_ok=True); p=cache/name
    if not p.exists():
        with urllib.request.urlopen(f"{RAW_BASE}/{name}",context=ssl._create_unverified_context(),timeout=90) as r: p.write_bytes(r.read())
    return p
def pub(cache):
    out=[]
    for s in SUBJECTS:
        for r in rc(fetch(f"words_dict_romanian_{int(s):03d}.csv",cache)):
            if r.get('word_id') and r.get('fixations_TRT') not in (None,''):
                q=dict(r); q['subject_key']=f"{int(s):03d}"; out.append(q)
    return out
def mat(rows):
    m={s:{} for s in SUBJECTS}
    for r in rows: m[r['subject_key']][r['word_id']]=float(r['fixations_TRT'])
    return m
def merged(cache):
    out={}
    for r in rc(fetch('words_dict_romanian_merged.csv',cache)):
        if r.get('word_id') and r.get('average_TRT') not in (None,''): out[r['word_id']]=float(r['average_TRT'])
    return out
def parse(wid):
    z=WORD_RE.match(wid); return (z.group(1),int(z.group(2)),int(z.group(3)),int(z.group(4))) if z else ('',0,0,0)
def fam(t): return (t.split('_',1)[0] if t else '')
def vals(m,wid): return [m[s][wid] for s in SUBJECTS if wid in m.get(s,{})]
def med(v):
    if not v: return 0.0
    o=sorted(v); n=len(o)//2; return float(o[n] if len(o)%2 else (o[n-1]+o[n])/2.0)
def outvals(path,test):
    d={r['datapointID']:float(r['answer']) for r in rc(path)}; return [d[r['datapointID']] for r in test]
def ridge(x,y,alpha=500.0):
    x=np.asarray(x,float); y=np.asarray(y,float); mu=x.mean(0); sd=x.std(0); sd[sd==0]=1.0; z=(x-mu)/sd
    d=np.column_stack([np.ones(len(z)),z]); p=np.eye(d.shape[1])*float(alpha); p[0,0]=0.0; c=np.linalg.solve(d.T@d+p,d.T@y)
    return c[1:]/sd, float(c[0]-np.sum(c[1:]*mu/sd))
def pred_ridge(xtr,y,xte,alpha,log=False):
    target=np.log1p(np.asarray(y,float)) if log else np.asarray(y,float); sl,ic=ridge(xtr,target,alpha); p=ic+np.asarray(xte,float)@sl
    return np.expm1(p) if log else p
def surf(r):
    w=r['word']; _,src,page,idx=parse(r['word_id']); a=sum(c.isalpha() for c in w); d=sum(c.isdigit() for c in w); pu=float(bool(w) and all(not c.isalnum() for c in w)); f=fam(r.get('text',''))
    return [1.0,len(w),math.log1p(len(w)),a,d,pu,float(w[:1].isupper()),float(w.isupper()),src,page,idx,math.log1p(idx)]+[float(f==x) for x in FAMS]
def safe(train,test,base):
    y=[float(r['answer']) for r in train]; sp=pred_ridge([surf(r) for r in train],y,[surf(r) for r in test],800.0,False)
    return np.maximum(0.0,0.62*np.asarray(base,float)+0.38*sp)
def tstats(rows):
    b={}
    for r in rows: b.setdefault(parse(r['word_id'])[0],[]).append(float(r['fixations_TRT']))
    return {k:float(np.mean(v)) for k,v in b.items()}
def mfeat(r,m,mm,ts):
    v=vals(m,r['word_id']) or [0.0]; a=np.asarray(v,float); text,src,page,idx=parse(r['word_id'])
    return [float(a.mean()),med(v),float(a.std()),float(a.min()),float(a.max()),float(np.mean(a==0.0)),mm.get(r['word_id'],0.0),ts.get(text,0.0),len(r['word']),math.log1p(len(r['word'])),src,page,idx]
def medium(train,test,rows,m,mm):
    ts=tstats(rows); y=[float(r['answer']) for r in train]
    return np.maximum(0.0,pred_ridge([mfeat(r,m,mm,ts) for r in train],y,[mfeat(r,m,mm,ts) for r in test],350.0,False))
def direct_cal(train,m):
    x=[]; y=[]
    for r in train:
        for v in vals(m,r['word_id']): x.append([v,len(r['word']),math.log1p(len(r['word']))]); y.append(float(r['answer']))
    return ridge(x,y,350.0)
def raw_direct(r,m,mp):
    v=vals(m,r['word_id']); fb=float(np.mean(v)) if v else 0.0; return m.get(mp.get(r.get('participant_id',''),''),{}).get(r['word_id'],fb)
def high(test,m,mm,hb,cal):
    var=CONFIG.get('high_variant','base'); mp=CONFIG.get('mapping',{}); out=[]
    for r,base in zip(test,hb,strict=True):
        v=vals(m,r['word_id']); mean=float(np.mean(v)) if v else 0.0
        if var=='base': z=base
        elif var=='raw_direct': z=raw_direct(r,m,mp)
        elif var=='item_mean': z=mean
        elif var=='merged_avg': z=mm.get(r['word_id'],mean)
        else:
            sl,ic=cal; raw=raw_direct(r,m,mp); z=ic+float(np.dot(sl,[raw,len(r['word']),math.log1p(len(r['word']))]))
        out.append(max(0.0,float(z)))
    return np.asarray(out,float)
def sfeat(r,m,mm):
    v=[m[s].get(r['word_id'],0.0) for s in SUBJECTS]; nz=[x for x in v if x!=0.0]; a=np.asarray(nz or [0.0],float); text,src,page,idx=parse(r['word_id']); w=r['word']; f=fam(r.get('text','')); mp=CONFIG.get('mapping',{}); mapped=raw_direct(r,m,mp)
    return v+[float(a.mean()),med(list(a)),float(a.std()),float(a.min()),float(a.max()),float(np.mean(a==0.0)),mm.get(r['word_id'],0.0),mapped,len(w),math.log1p(len(w)),sum(c.isalpha() for c in w),float(w[:1].isupper()),src,page,idx,math.log1p(idx),float(r.get('participant_id','0'))]+[float(f==x) for x in FAMS]
def stack(train,test,m,mm):
    x=[sfeat(r,m,mm) for r in train]; z=[sfeat(r,m,mm) for r in test]; y=np.asarray([float(r['answer']) for r in train],float); model=CONFIG.get('model','ridge_raw')
    if model=='ridge_log': p=pred_ridge(x,y,z,CONFIG.get('alpha',100.0),True)
    elif model=='huber':
        from sklearn.linear_model import SGDRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        clf=make_pipeline(StandardScaler(),SGDRegressor(loss='huber',epsilon=1.35,alpha=CONFIG.get('alpha',0.0001),max_iter=1200,tol=1e-3,random_state=17))
        p=clf.fit(np.asarray(x,float),y).predict(np.asarray(z,float))
    elif model=='trees':
        from sklearn.ensemble import ExtraTreesRegressor
        clf=ExtraTreesRegressor(n_estimators=int(CONFIG.get('trees',48)),min_samples_leaf=int(CONFIG.get('leaf',5)),max_features=0.85,random_state=17,n_jobs=1)
        p=clf.fit(np.asarray(x,float),y).predict(np.asarray(z,float))
    else: p=pred_ridge(x,y,z,CONFIG.get('alpha',100.0),False)
    return np.maximum(0.0,np.asarray(p,float))
def apply_tail(out):
    out=np.asarray(out,float)*CONFIG.get('scale',1.0); out=np.clip(out,0.0,10000.0)
    if CONFIG.get('zero_quantile') is not None:
        q=float(np.quantile(out,CONFIG['zero_quantile'])); out=np.where(out<=q,0.0,out)
    if CONFIG.get('floor_quantile') is not None:
        q=float(np.quantile(out,CONFIG['floor_quantile'])); out=np.maximum(out,q)
    if CONFIG.get('cap_quantile') is not None:
        q=float(np.quantile(out,CONFIG['cap_quantile'])); out=np.minimum(out,q)
    return out
def write(test,out,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as h:
        w=csv.DictWriter(h,fieldnames=['subtaskID','datapointID','answer'],lineterminator='\n'); w.writeheader()
        for r,v in zip(test,out,strict=True): w.writerow({'subtaskID':'1','datapointID':r['datapointID'],'answer':f"{max(0.0,float(v)):.6f}"})
def main():
    p=argparse.ArgumentParser(); p.add_argument('--train',type=Path,required=True); p.add_argument('--test',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--base-output',type=Path,required=True); p.add_argument('--high-risk-base-output',type=Path,required=True); p.add_argument('--anchor-output',type=Path,required=True); p.add_argument('--best-output',type=Path,required=True); a=p.parse_args()
    train=rc(a.train); test=rc(a.test); cache=Path('.cache/aggressive_feedback_push'); rows=pub(cache); m=mat(rows); mm=merged(cache); b=outvals(a.base_output,test); hb=outvals(a.high_risk_base_output,test); an=np.asarray(outvals(a.anchor_output,test),float); be=np.asarray(outvals(a.best_output,test),float)
    direction=be+CONFIG.get('gamma',3.0)*(be-an); s=safe(train,test,b); md=medium(train,test,rows,m,mm); cal=direct_cal(train,m); hr=high(test,m,mm,hb,cal); mode=CONFIG.get('mode','blend')
    if mode=='direction': out=direction
    else:
        wt=CONFIG.get('weights',{'safe':.1,'medium':.3,'high':.3,'stack':.3}); base=wt.get('safe',0)*s+wt.get('medium',0)*md+wt.get('high',0)*hr
        if mode in ('trained','champion') or wt.get('stack',0)>0: base=base+wt.get('stack',0)*stack(train,test,m,mm)
        dw=CONFIG.get('direction_weight',0.0); out=(1.0-dw)*base+dw*direction
    write(test,apply_tail(out),a.output)
if __name__=='__main__': main()
