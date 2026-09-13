#!/usr/bin/env python3
"""Paired XOR negative-slope dose-response experiment (NumPy, deterministic)."""
import json, math
from pathlib import Path
import numpy as np

X=np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
Y=np.array([[0.],[1.],[1.],[0.]])
SEEDS=list(range(200)); SLOPES=[0.,1e-4,1e-3,1e-2,1e-1]
CHECKPOINTS=[0,25,50,100,200,500,1000,2000]; STEPS=2000; LR=.01

def sigmoid(z): return 1/(1+np.exp(-np.clip(z,-60,60)))
def loss_fn(z): return float(np.mean(np.maximum(z,0)-z*Y+np.log1p(np.exp(-np.abs(z)))))
def init(seed,scheme):
    r=np.random.RandomState(seed)
    if scheme=='he': s1,s2=math.sqrt(2/2),math.sqrt(2/4)
    else: s1,s2=math.sqrt(2/6),math.sqrt(2/5)
    return [r.randn(2,4)*s1,np.zeros((1,4)),r.randn(4,1)*s2,np.zeros((1,1))]
def forward(p,alpha):
    z=X@p[0]+p[1]; h=np.where(z>0,z,alpha*z); q=h@p[2]+p[3]; pr=sigmoid(q).reshape(-1)
    return z,h,q,pr,loss_fn(q),float(np.mean((pr>=.5)==Y.reshape(-1)))
def gradients(p,alpha):
    z,h,q,pr,lo,ac=forward(p,alpha); dh=np.where(z>0,1.,alpha)
    dl=(pr.reshape(-1,1)-Y)/4; d=(dl@p[2].T)*dh
    return [X.T@d,np.sum(d,0,keepdims=True),h.T@dl,np.sum(dl,0,keepdims=True)]
def diagnostics(p,initial,alpha,gn,un):
    z,h,q,pr,lo,ac=forward(p,alpha); bits=(z>0).astype(int)
    signatures=[''.join(map(str,row)) for row in bits]
    d={'loss':lo,'accuracy':ac,'all_negative_unit_count':int(np.sum(np.all(bits==0,0))),
       'distinct_signature_count':len(set(signatures)),'hidden_output_variance_mean':float(np.mean(np.var(h,0))),
       'first_layer_gradient_norm':float(gn),'first_layer_update_norm':float(un),
       'W1_displacement':float(np.linalg.norm(p[0]-initial[0]))}
    if alpha==0.: d['dead_unit_count']=d['all_negative_unit_count']; del d['all_negative_unit_count']
    return d
def run(seed,scheme,alpha):
    p=init(seed,scheme); initial=[a.copy() for a in p]; m=[np.zeros_like(a) for a in p]; v=[np.zeros_like(a) for a in p]
    cps={}; g=gradients(p,alpha); cps['0']=diagnostics(p,initial,alpha,float(np.linalg.norm(g[0])),0.); first=0 if cps['0']['accuracy']==1 else None
    b1,b2,e=.9,.999,1e-8
    for step in range(1,STEPS+1):
        g=gradients(p,alpha); update_norm=0.
        for i,x in enumerate(g):
            m[i]=b1*m[i]+(1-b1)*x; v[i]=b2*v[i]+(1-b2)*x*x
            mh=m[i]/(1-b1**step); vh=v[i]/(1-b2**step); delta=LR*mh/(np.sqrt(vh)+e); p[i]-=delta
            if i==0: update_norm=float(np.linalg.norm(delta))
        post=forward(p,alpha)
        if first is None and post[5]==1: first=step
        if step in CHECKPOINTS:
            gg=gradients(p,alpha); cps[str(step)]=diagnostics(p,initial,alpha,float(np.linalg.norm(gg[0])),update_norm)
    final=forward(p,alpha)
    return {'seed':seed,'initialization':scheme,'alpha':alpha,'first_success_step':first,
            'ended_at_100':bool(cps['2000']['accuracy']==1.),'checkpoints':cps,'final_loss':final[4],'final_accuracy':final[5]}
def binom_cdf(k,n): return sum(math.comb(n,i) for i in range(k+1))/(2**n)
def exact_test(b,c):
    n=b+c; p=1. if n==0 else min(1.,2*binom_cdf(min(b,c),n))
    return {'discordant_total':n,'two_sided_p':p,'definition':'exact two-sided binomial test on discordant pairs, null p=0.5'}
def aggregate(rs,alpha):
    out={}
    for cp in CHECKPOINTS:
        vs=[r['checkpoints'][str(cp)] for r in rs]; success=[x['accuracy']==1 for x in vs]
        d={'success_count':sum(success),'success_fraction':sum(success)/200.,'mean_accuracy':float(np.mean([x['accuracy'] for x in vs])),'mean_loss':float(np.mean([x['loss'] for x in vs])),'median_loss':float(np.median([x['loss'] for x in vs])),'mean_signature_diversity':float(np.mean([x['distinct_signature_count'] for x in vs])),'median_first_layer_gradient_norm':float(np.median([x['first_layer_gradient_norm'] for x in vs])),'median_first_layer_update_norm':float(np.median([x['first_layer_update_norm'] for x in vs])),'median_W1_displacement':float(np.median([x['W1_displacement'] for x in vs]))}
        key='dead_unit_count' if alpha==0 else 'all_negative_unit_count'; d.update({'mean_'+key:float(np.mean([x[key] for x in vs])),'median_'+key:float(np.median([x[key] for x in vs])),'fraction_any_'+key:float(np.mean([x[key]>0 for x in vs]))}); out[str(cp)]=d
    succ=[r['first_success_step'] for r in rs if r['first_success_step'] is not None]
    out['final_summary']={'mean_final_loss':float(np.mean([r['final_loss'] for r in rs])),'median_final_loss':float(np.median([r['final_loss'] for r in rs])),'mean_final_accuracy':float(np.mean([r['final_accuracy'] for r in rs])),'median_first_success_step':float(np.median(succ)) if succ else None,'successful_runs':len(succ),'never_successful_by_2000':200-len(succ)}
    return out
def main():
    rows=[run(seed,scheme,alpha) for scheme in ('he','xavier') for seed in SEEDS for alpha in SLOPES]
    summaries={}; paired={}; rescue={}; dead2={}
    for scheme in ('he','xavier'):
        sr=[r for r in rows if r['initialization']==scheme]
        summaries[scheme]={str(a):aggregate([r for r in sr if r['alpha']==a],a) for a in SLOPES}
        control={r['seed']:r for r in sr if r['alpha']==0.}
        paired[scheme]={}; rescue[scheme]={}; dead2[scheme]={}
        control_fail={s for s,r in control.items() if not r['ended_at_100']}
        early_dead={s for s,r in control.items() if r['checkpoints']['25']['dead_unit_count']>=2}
        for a in SLOPES[1:]:
            cur={r['seed']:r for r in sr if r['alpha']==a}; only_s=sum(cur[s]['ended_at_100'] and not control[s]['ended_at_100'] for s in SEEDS); only_r=sum(control[s]['ended_at_100'] and not cur[s]['ended_at_100'] for s in SEEDS); both=sum(control[s]['ended_at_100'] and cur[s]['ended_at_100'] for s in SEEDS); neither=200-only_s-only_r-both
            paired[scheme][str(a)]={'only_slope':only_s,'only_relu':only_r,'both':both,'neither':neither,'success_rate_difference_vs_relu':(only_s-only_r)/200.,'mcnemar_exact':exact_test(only_s,only_r)}
            rescued={s for s in control_fail if cur[s]['ended_at_100']}; rescue[scheme][str(a)]={'prior_relu_failure_count':len(control_fail),'rescued_count':len(rescued),'rescue_fraction':len(rescued)/len(control_fail) if control_fail else None,'non_rescued_failure_count':len(control_fail-rescued)}
            rescued_early={s for s in early_dead if cur[s]['ended_at_100']}; dead2[scheme][str(a)]={'control_early_dead_ge2_count':len(early_dead),'rescued_from_early_dead_ge2':len(rescued_early),'rescue_fraction':len(rescued_early)/len(early_dead) if early_dead else None}
    result={'experiment':'XOR LeakyReLU negative-slope dose response','runtime':'numpy_fallback_no_pytorch','dataset':{'inputs':X.tolist(),'labels':Y.reshape(-1).astype(int).tolist()},'architecture':'2->4->1','optimizer':{'name':'Adam','learning_rate':LR,'steps':STEPS,'full_batch':True},'slopes':SLOPES,'seeds':SEEDS,'checkpoints':CHECKPOINTS,'summaries':summaries,'paired_tests_vs_relu':paired,'rescue_analysis_prior_relu_failures':rescue,'early_dead_ge2_rescue':dead2,'runs':rows}
    Path(__file__).with_name('metrics.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'rows':len(rows),'slopes':SLOPES},indent=2))
if __name__=='__main__': main()
