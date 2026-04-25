from __future__ import annotations
import argparse,csv,math,re,ssl,urllib.request
from pathlib import Path
import numpy as np
PINNED_COMMIT='6c724e8877c52ea021f295216949860f87d1c8b2'; RAW_BASE='https://raw.githubusercontent.com/ana0101/eye-tracking/6c724e8877c52ea021f295216949860f87d1c8b2/trt_model/word_sentence_fixations'; SUBJECTS=['008', '009', '010', '011', '023']; CONFIG={'mode': 'component', 'component': 'stack', 'model': 'trees', 'weight': 0.2, 'mapping': {'040': '010', '036': '011', '016': '008', '024': '023', '019': '009'}}
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
    d={}
    for r in rc(fetch('words_dict_romanian_merged.csv',cache)):
        if r.get('word_id') and r.get('average_TRT') not in (None,''): d[r['word_id']]=float(r['average_TRT'])
    return d
def parse(wid):
    z=WORD_RE.match(wid); return (z.group(1),int(z.group(2)),int(z.group(3)),int(z.group(4))) if z else ('',0,0,0)
def vals(m,wid): return [m[s][wid] for s in SUBJECTS if wid in m.get(s,{})]
def med(v):
    if not v: return 0.0
    o=sorted(v); n=len(o)//2; return float(o[n] if len(o)%2 else (o[n-1]+o[n])/2.0)
def outvals(path,test):
    d={r['datapointID']:float(r['answer']) for r in rc(path)}; return np.asarray([d[r['datapointID']] for r in test],float)
def zmatch(x,ref):
    x=np.asarray(x,float); ref=np.asarray(ref,float); sd=float(x.std()) or 1.0
    return (x-float(x.mean()))/sd*(float(ref.std()) or 1.0)+float(ref.mean())
def ridge(x,y,alpha=300.0,log=False):
    x=np.asarray(x,float); y=np.asarray(y,float); y=np.log1p(y) if log else y; mu=x.mean(0); sd=x.std(0); sd[sd==0]=1.0; z=(x-mu)/sd
    d=np.column_stack([np.ones(len(z)),z]); p=np.eye(d.shape[1])*alpha; p[0,0]=0.0; c=np.linalg.solve(d.T@d+p,d.T@y); sl=c[1:]/sd; ic=float(c[0]-np.sum(c[1:]*mu/sd)); return sl,ic,log
def rpred(model,x):
    sl,ic,log=model; p=ic+np.asarray(x,float)@sl; return np.expm1(p) if log else p
def raw(r,m):
    mp=CONFIG.get('mapping',{}); v=vals(m,r['word_id']); fb=float(np.mean(v)) if v else 0.0; return m.get(mp.get(r.get('participant_id',''),''),{}).get(r['word_id'],fb)
def dcal(train,m):
    x=[]; y=[]
    for r in train:
        for v in vals(m,r['word_id']): x.append([v,len(r['word']),math.log1p(len(r['word']))]); y.append(float(r['answer']))
    return ridge(x,y,300.0,False)
def high(test,m,mm,hb,cal,ref):
    var=CONFIG.get('variant','cal_direct'); sl,ic,_=cal; out=[]
    for r,b in zip(test,hb,strict=True):
        v=vals(m,r['word_id']); mean=float(np.mean(v)) if v else 0.0
        if var=='base': z=b
        elif var=='item_mean': z=mean
        elif var=='merged_avg': z=mm.get(r['word_id'],mean)
        else:
            rv=raw(r,m); z=ic+float(np.dot(sl,[rv,len(r['word']),math.log1p(len(r['word']))]))
        out.append(max(0.0,float(z)))
    return zmatch(out,ref)
def feat(r,m,mm):
    v=[m[s].get(r['word_id'],0.0) for s in SUBJECTS]; nz=[x for x in v if x!=0]; a=np.asarray(nz or [0.0],float); text,src,page,idx=parse(r['word_id']); w=r['word']; fam=(r.get('text','').split('_',1)[0] if r.get('text') else '')
    return v+[float(a.mean()),med(list(a)),float(a.std()),float(a.min()),float(a.max()),float(np.mean(a==0.0)),mm.get(r['word_id'],0.0),raw(r,m),len(w),math.log1p(len(w)),sum(c.isalpha() for c in w),float(w[:1].isupper()),src,page,idx,math.log1p(idx),float(r.get('participant_id','0'))]+[float(fam==x) for x in FAMS]
def stack(train,test,m,mm,ref):
    x=[feat(r,m,mm) for r in train]; z=[feat(r,m,mm) for r in test]; y=np.asarray([float(r['answer']) for r in train],float); model=CONFIG.get('model','ridge_log')
    if model=='huber':
        from sklearn.linear_model import SGDRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        clf=make_pipeline(StandardScaler(),SGDRegressor(loss='huber',epsilon=1.35,alpha=.0001,max_iter=1200,tol=1e-3,random_state=29)); p=clf.fit(np.asarray(x,float),y).predict(np.asarray(z,float))
    elif model=='trees':
        from sklearn.ensemble import ExtraTreesRegressor
        clf=ExtraTreesRegressor(n_estimators=56,min_samples_leaf=5,max_features=.85,random_state=29,n_jobs=1); p=clf.fit(np.asarray(x,float),y).predict(np.asarray(z,float))
    else: p=rpred(ridge(x,y,120.0,model=='ridge_log'),z)
    return zmatch(np.maximum(0.0,p),ref)
def write(test,out,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as h:
        w=csv.DictWriter(h,fieldnames=['subtaskID','datapointID','answer'],lineterminator='\n'); w.writeheader()
        for r,v in zip(test,np.clip(out,0.0,10000.0),strict=True): w.writerow({'subtaskID':'1','datapointID':r['datapointID'],'answer':f"{float(v):.6f}"})
def main():
    p=argparse.ArgumentParser(); p.add_argument('--train',type=Path,required=True); p.add_argument('--test',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--best-output',type=Path,required=True); p.add_argument('--failed-output',type=Path,required=True); p.add_argument('--anchor-output',type=Path,required=True); p.add_argument('--high-risk-base-output',type=Path,required=True); a=p.parse_args()
    train=rc(a.train); test=rc(a.test); best=outvals(a.best_output,test); failed=outvals(a.failed_output,test); anchor=outvals(a.anchor_output,test); hb=outvals(a.high_risk_base_output,test); cache=Path('.cache/failure_rebound'); rows=pub(cache); m=mat(rows); mm=merged(cache); cal=dcal(train,m); mode=CONFIG.get('mode','outputs')
    if mode=='outputs': out=CONFIG.get('best',1)*best+CONFIG.get('failed',0)*failed+CONFIG.get('anchor',0)*anchor
    elif mode=='component':
        comp=stack(train,test,m,mm,best) if CONFIG.get('component')=='stack' else high(test,m,mm,hb,cal,best); out=(1-CONFIG.get('weight',.1))*best+CONFIG.get('weight',.1)*comp+CONFIG.get('counter',0)*(best-failed)
    else:
        comp=stack(train,test,m,mm,best); hv=high(test,m,mm,hb,cal,best); out=best+CONFIG.get('counter',.15)*(best-failed)+CONFIG.get('prior',.3)*(best-anchor)+CONFIG.get('stack_weight',.1)*(comp-best)+CONFIG.get('high_weight',.1)*(hv-best)
    write(test,out,a.output)
if __name__=='__main__': main()
