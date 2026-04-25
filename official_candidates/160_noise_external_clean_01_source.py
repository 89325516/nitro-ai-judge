from __future__ import annotations
import argparse,csv,math,re,ssl,urllib.request
from collections import defaultdict
from pathlib import Path
import numpy as np
CONFIG={'mode': 'external', 'target': 'median', 'weight': 0.06, 'calibration': 'ridge'}; RAW_BASE='https://raw.githubusercontent.com/ana0101/eye-tracking/6c724e8877c52ea021f295216949860f87d1c8b2/trt_model/word_sentence_fixations'; WORD_RE=re.compile(r"^(.*)_([0-9]+)_page_([0-9]+)_([0-9]+)$"); FAMS=("arg","enc","ins","lit","popsci")
def rc(p):
    with open(p,newline='',encoding='utf-8-sig') as h: return list(csv.DictReader(h))
def fetch(name):
    c=Path('.cache/noise_aware_external'); c.mkdir(parents=True,exist_ok=True); p=c/name
    if not p.exists():
        with urllib.request.urlopen(f"{RAW_BASE}/{name}",context=ssl._create_unverified_context(),timeout=90) as r: p.write_bytes(r.read())
    return p
def ext():
    d={}; comp={}
    for name in ['words_dict_romanian_merged.csv','words_dict_romanian_merged_008_009_010_011.csv']:
        for r in rc(fetch(name)):
            if r.get('word_id') and r.get('average_TRT') not in (None,''):
                d[r['word_id']]=float(r['average_TRT'])
                try: comp[r['word_id']]=float(r.get('complexity') or 0.0)
                except ValueError: comp[r['word_id']]=0.0
    return d,comp
def parse(wid):
    z=WORD_RE.match(wid); return (z.group(1),int(z.group(2)),int(z.group(3)),int(z.group(4))) if z else ('',0,0,0)
def fam(t): return t.split('_',1)[0] if t else ''
def stats(train):
    bw=defaultdict(list); bt=defaultdict(list); bp=defaultdict(list); bf=defaultdict(list); bg=defaultdict(list); arr=[]
    for r in train:
        v=float(r['answer']); arr.append(v); bw[r['word_id']].append(v); bt[r['text']].append(v); bp[(r['text'],parse(r['word_id'])[2])].append(v); bf[fam(r['text'])].append(v); bg[r['participant_id']].append(v)
    a=np.asarray(arr,float); return bw,{k:float(np.mean(v)) for k,v in bt.items()},{k:float(np.mean(v)) for k,v in bp.items()},{k:float(np.mean(v)) for k,v in bf.items()},float(a.mean()),float(np.mean(a==0.0))
def feat(r,ed,cp):
    w=r['word']; text,src,page,idx=parse(r['word_id']); f=fam(r.get('text','')); ev=ed.get(r['word_id'],0.0); cv=cp.get(r['word_id'],0.0)
    return [1.0,len(w),math.log1p(len(w)),sum(c.isalpha() for c in w),sum(c.isdigit() for c in w),float(bool(w) and all(not c.isalnum() for c in w)),float(w[:1].isupper()),float(w.isupper()),src,page,idx,math.log1p(idx),ev,math.log1p(max(ev,0.0)),float(ev==0.0),cv]+[float(f==x) for x in FAMS]
def outvals(path,test):
    d={r['datapointID']:float(r['answer']) for r in rc(path)}; return np.asarray([d[r['datapointID']] for r in test],float)
def agg(vals,kind):
    a=np.asarray(vals,float); pos=a[a>0]; zr=float(np.mean(a==0.0))
    if kind in ('external','hybrid'): return float(a.mean())
    if kind=='median': return float(np.median(a))
    if kind in ('positive','log'): return float(pos.mean()) if len(pos) else 0.0
    if kind=='zeroaware': return (float(pos.mean()) if len(pos) else float(a.mean()))*(1.0-zr)
    if kind.startswith('trim'):
        q=float(kind[4:])/100.0; lo,hi=np.quantile(a,[q,1-q]); b=a[(a>=lo)&(a<=hi)]; return float(b.mean()) if len(b) else float(a.mean())
    if kind.startswith('winsor'):
        q=float(kind[6:])/100.0; lo,hi=np.quantile(a,[q,1-q]); return float(np.clip(a,lo,hi).mean())
    return float(a.mean())
def ridge(x,y,w=None,alpha=200.0,log=False):
    x=np.asarray(x,float); y=np.asarray(y,float); y=np.log1p(np.maximum(0,y)) if log else y; mu=x.mean(0); sd=x.std(0); sd[sd==0]=1.0; z=(x-mu)/sd; d=np.column_stack([np.ones(len(z)),z])
    if w is not None: sw=np.sqrt(np.asarray(w,float)); d=d*sw[:,None]; y=y*sw
    p=np.eye(d.shape[1])*alpha; p[0,0]=0.0; c=np.linalg.solve(d.T@d+p,d.T@y); sl=c[1:]/sd; ic=float(c[0]-np.sum(c[1:]*mu/sd)); return sl,ic,log
def pred(model,x):
    sl,ic,log=model; y=ic+np.asarray(x,float)@sl; return np.expm1(y) if log else y
def zmatch(x,ref):
    x=np.asarray(x,float); ref=np.asarray(ref,float); return (x-x.mean())/(x.std() or 1.0)*(ref.std() or 1.0)+ref.mean()
def fit_predict(train,test,y,model,ed,cp,trees=96):
    x=[feat(r,ed,cp) for r in train]; z=[feat(r,ed,cp) for r in test]
    if model=='trees':
        from sklearn.ensemble import ExtraTreesRegressor
        m=ExtraTreesRegressor(n_estimators=int(trees or 96),min_samples_leaf=5,max_features=.85,random_state=37,n_jobs=1); return m.fit(np.asarray(x,float),np.asarray(y,float)).predict(np.asarray(z,float))
    if model=='hgb':
        from sklearn.ensemble import HistGradientBoostingRegressor
        m=HistGradientBoostingRegressor(max_iter=140,learning_rate=.045,l2_regularization=.08,max_leaf_nodes=31,random_state=37); return m.fit(np.asarray(x,float),np.asarray(y,float)).predict(np.asarray(z,float))
    return pred(ridge(x,y,alpha=240.0,log=False),z)
def component(train,test,best,bw,txt,pages,fams,glob,zr,ed,cp,cfg):
    mode=cfg.get('mode','external'); kind=cfg.get('target','median')
    if mode=='external':
        raw=np.asarray([ed.get(r['word_id'],0.0) for r in test],float)
        if cfg.get('calibration')=='direct': return zmatch(raw,best)
        y=[agg(bw[r['word_id']],kind) for r in train]; return zmatch(np.maximum(0,fit_predict(train,test,y,'ridge',ed,cp)),best)
    if mode=='scaled':
        y=[agg(bw[r['word_id']],kind) for r in train]; return zmatch(np.maximum(0,fit_predict(train,test,y,cfg.get('model','ridge'),ed,cp,cfg.get('trees',96))),best)
    if mode=='eb':
        y=[]; strength=float(cfg.get('strength',12.0)); prior=cfg.get('prior','external')
        for r in train:
            text,src,page,idx=parse(r['word_id']); vals=bw[r['word_id']]; base=float(np.mean(vals)); ev=ed.get(r['word_id'],base)
            pr=ev if prior=='external' else txt.get(r['text'],glob) if prior=='text' else pages.get((r['text'],page),glob) if prior=='page' else fams.get(fam(r['text']),glob) if prior=='family' else .5*ev+.25*txt.get(r['text'],glob)+.25*pages.get((r['text'],page),glob)
            y.append((len(vals)*base+strength*pr)/(len(vals)+strength))
        return zmatch(np.maximum(0,fit_predict(train,test,y,cfg.get('model','ridge'),ed,cp,cfg.get('trees',96))),best)
    if mode=='hurdle':
        zy=[1.0 if float(r['answer'])==0.0 else 0.0 for r in train]; zm=np.clip(fit_predict(train,test,zy,'ridge',ed,cp),0.02,0.95); pos=cfg.get('positive','log')
        y=[ed.get(r['word_id'],agg(bw[r['word_id']],'median')) if pos=='external' else .5*ed.get(r['word_id'],0.0)+.5*agg(bw[r['word_id']],'zeroaware') if pos=='hybrid' else agg(bw[r['word_id']],pos) for r in train]
        pm=np.maximum(0,fit_predict(train,test,y,cfg.get('model','ridge'),ed,cp,cfg.get('trees',96))); return zmatch((1.0-cfg.get('zero_weight',.6)*zm)*pm,best)
    ex=component(train,test,best,bw,txt,pages,fams,glob,zr,ed,cp,{'mode':'external','target':'winsor10','calibration':'ridge'})
    sc=component(train,test,best,bw,txt,pages,fams,glob,zr,ed,cp,{'mode':'scaled','target':'trim10','model':'hgb'})
    eb=component(train,test,best,bw,txt,pages,fams,glob,zr,ed,cp,{'mode':'eb','prior':'hybrid','strength':18.0,'model':'hgb'})
    hu=component(train,test,best,bw,txt,pages,fams,glob,zr,ed,cp,{'mode':'hurdle','positive':'hybrid','zero_weight':.7,'model':'hgb'})
    return best+cfg.get('external_weight',.06)*(ex-best)+cfg.get('scaled_weight',.08)*(sc-best)+cfg.get('eb_weight',.08)*(eb-best)+cfg.get('hurdle_weight',.06)*(hu-best)
def write(test,out,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as h:
        w=csv.DictWriter(h,fieldnames=['subtaskID','datapointID','answer'],lineterminator='\n'); w.writeheader()
        for r,v in zip(test,np.clip(out*CONFIG.get('scale',1.0),0.0,10000.0),strict=True): w.writerow({'subtaskID':'1','datapointID':r['datapointID'],'answer':f"{float(v):.6f}"})
def main():
    p=argparse.ArgumentParser(); p.add_argument('--train',type=Path,required=True); p.add_argument('--test',type=Path,required=True); p.add_argument('--best-output',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    train=rc(a.train); test=rc(a.test); best=outvals(a.best_output,test); ed,cp=ext(); bw,txt,pages,fams,glob,zr=stats(train); out=component(train,test,best,bw,txt,pages,fams,glob,zr,ed,cp,CONFIG); write(test,out,a.output)
if __name__=='__main__': main()
