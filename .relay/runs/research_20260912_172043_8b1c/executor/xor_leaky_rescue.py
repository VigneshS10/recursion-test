#!/usr/bin/env python3
"""Paired ReLU/LeakyReLU(0.01) XOR rescue experiment."""
import json, math
from pathlib import Path
import numpy as np

X=np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
Y=np.array([[0.],[1.],[1.],[0.]])
SEEDS=list(range(200)); CHECKPOINTS=[0,25,50,100,200,500,1000,2000]; STEPS=2000; LR=.01; SLOPE=.01

def sigmoid(z): return 1/(1+np.exp(-np.clip(z,-60,60)))
def loss_fn(z): return float(np.mean(np.maximum(z,0)-z*Y+np.log1p(np.exp(-np.abs(z)))))
def init(seed,scheme):
    r=np.random.RandomState(seed)
    if scheme=='he': s1,s2=math.sqrt(2/2),math.sqrt(2/4)
    else: s1,s2=math.sqrt(2/6),math.sqrt(2/5)
    return [r.randn(2,4)*s1,np.zeros((1,4)),r.randn(4,1)*s2,np.zeros((1,1))]
def fwd(p,act):
    z=X@p[0]+p[1]; h=np.maximum(z,0) if act=='relu' else np.where(z>0,z,SLOPE*z)
    q=h@p[2]+p[3]; pr=sigmoid(q).reshape(-1); return z,h,q,pr,loss_fn(q),float(np.mean((pr>=.5)==Y.reshape(-1)))
def grads(p,act):
    z,h,q,pr,lo,ac=fwd(p,act); dh=(z>0).astype(float) if act=='relu' else np.where(z>0,1.,SLOPE)
    dl=(pr.reshape(-1,1)-Y)/4; d=(dl@p[2].T)*dh
    return [X.T@d,np.sum(d,0,keepdims=True),h.T@dl,np.sum(dl,0,keepdims=True)]
def diag(p,initial,act,gn,un,step):
    z,h,q,pr,lo,ac=fwd(p,act); bits=(z>0).astype(int); sig=[''.join(map(str,row)) for row in bits]
    d={'loss':lo,'accuracy':ac,'pre_activations':z.tolist(),'hidden_outputs':h.tolist(),'signatures':sig,'distinct_signature_count':len(set(sig)),'hidden_output_variance_per_unit':[float(x) for x in np.var(h,0)],'hidden_output_variance_mean':float(np.mean(np.var(h,0))),'gradient_norms':dict(zip(['W1','b1','W2','b2'],gn)),'update_norms':dict(zip(['W1','b1','W2','b2'],un)),'W1_displacement':float(np.linalg.norm(p[0]-initial[0])),'W2_displacement':float(np.linalg.norm(p[2]-initial[2]))}
    if act=='relu': d['dead_unit_count']=int(np.sum(np.all(bits==0,0)))
    else: d['all_negative_unit_count']=int(np.sum(np.all(bits==0,0)))
    return d
def run(seed,scheme,act):
    p=init(seed,scheme); initial=[a.copy() for a in p]; m=[np.zeros_like(a) for a in p]; v=[np.zeros_like(a) for a in p]; cps={}
    g=grads(p,act); cps['0']=diag(p,initial,act,[float(np.linalg.norm(x)) for x in g],[0.,0.,0.,0.],0); first=0 if cps['0']['accuracy']==1 else None; b1,b2,e=.9,.999,1e-8
    for step in range(1,STEPS+1):
        g=grads(p,act); un=[]
        for i,x in enumerate(g):
            m[i]=b1*m[i]+(1-b1)*x; v[i]=b2*v[i]+(1-b2)*x*x; mh=m[i]/(1-b1**step); vh=v[i]/(1-b2**step); delta=LR*mh/(np.sqrt(vh)+e); p[i]-=delta; un.append(float(np.linalg.norm(delta)))
        post=fwd(p,act)
        if first is None and post[5]==1: first=step
        if step in CHECKPOINTS:
            gg=grads(p,act); cps[str(step)]=diag(p,initial,act,[float(np.linalg.norm(x)) for x in gg],un,step)
    return {'seed':seed,'initialization':scheme,'activation':act,'first_success_step':first,'ended_at_100':cps['2000']['accuracy']==1.,'checkpoints':cps,'final_probabilities':[float(x) for x in fwd(p,act)[3]]}
def binom_cdf(k,n): return sum(math.comb(n,i) for i in range(k+1))/(2**n)
def exact_test(b,c):
    n=b+c; return {'discordant_total':n,'two_sided_p':1. if n==0 else min(1.,2*binom_cdf(min(b,c),n)),'definition':'exact two-sided binomial test on discordant pairs, null p=0.5'}
def aggregate(rs,act):
    out={}
    for cp in CHECKPOINTS:
        vs=[r['checkpoints'][str(cp)] for r in rs]; d={'success_count':sum(x['accuracy']==1 for x in vs),'success_fraction':sum(x['accuracy']==1 for x in vs)/200.,'mean_accuracy':float(np.mean([x['accuracy'] for x in vs])),'mean_loss':float(np.mean([x['loss'] for x in vs])),'median_loss':float(np.median([x['loss'] for x in vs])),'mean_variance':float(np.mean([x['hidden_output_variance_mean'] for x in vs])),'median_W1_gradient':float(np.median([x['gradient_norms']['W1'] for x in vs])),'median_W1_update':float(np.median([x['update_norms']['W1'] for x in vs])),'median_W1_displacement':float(np.median([x['W1_displacement'] for x in vs])),'median_W2_displacement':float(np.median([x['W2_displacement'] for x in vs])),'final_accuracy_distribution':{str(a):sum(x['accuracy']==a for x in vs) for a in (.5,.75,1.)}}
        if act=='relu': d.update({'mean_dead':float(np.mean([x['dead_unit_count'] for x in vs])),'median_dead':float(np.median([x['dead_unit_count'] for x in vs])),'fraction_any_dead':float(np.mean([x['dead_unit_count']>0 for x in vs]))})
        else: d.update({'mean_all_negative':float(np.mean([x['all_negative_unit_count'] for x in vs])),'median_all_negative':float(np.median([x['all_negative_unit_count'] for x in vs])),'fraction_any_all_negative':float(np.mean([x['all_negative_unit_count']>0 for x in vs]))})
        out[str(cp)]=d
    return out
def main():
    rows=[run(seed,scheme,act) for scheme in ('he','xavier') for seed in SEEDS for act in ('relu','leaky_relu')]
    summaries={}; paired={}; rescue={}
    for scheme in ('he','xavier'):
        rr=[r for r in rows if r['initialization']==scheme and r['activation']=='relu']; ll=[r for r in rows if r['initialization']==scheme and r['activation']=='leaky_relu']; rd={r['seed']:r for r in rr}; ld={r['seed']:r for r in ll}
        good={r['seed'] for r in rr if r['ended_at_100']}; bad=set(SEEDS)-good
        summaries[scheme]={'relu':aggregate(rr,'relu'),'leaky_relu':aggregate(ll,'leaky_relu'),'relu_success_n':len(good),'relu_failure_n':len(bad)}
        pt={}
        for cp in CHECKPOINTS:
            a=sum(ld[i]['checkpoints'][str(cp)]['accuracy']==1 for i in SEEDS); b=sum(rd[i]['checkpoints'][str(cp)]['accuracy']==1 for i in SEEDS); only_l=sum((not rd[i]['checkpoints'][str(cp)]['accuracy']==1) and ld[i]['checkpoints'][str(cp)]['accuracy']==1 for i in SEEDS); only_r=sum(rd[i]['checkpoints'][str(cp)]['accuracy']==1 and (not ld[i]['checkpoints'][str(cp)]['accuracy']==1) for i in SEEDS); both=sum(rd[i]['checkpoints'][str(cp)]['accuracy']==1 and ld[i]['checkpoints'][str(cp)]['accuracy']==1 for i in SEEDS); neither=200-only_l-only_r-both
            pt[str(cp)]={'only_leaky_relu':only_l,'only_relu':only_r,'both':both,'neither':neither,'success_rate_difference_leaky_minus_relu':(a-b)/200.,'mcnemar_exact':exact_test(only_l,only_r)}
        paired[scheme]=pt
        rescued=sorted(i for i in bad if ld[i]['ended_at_100']); notrescued=sorted(bad-set(rescued));
        def early_ge2(ids): return sum(rd[i]['checkpoints']['25']['dead_unit_count']>=2 for i in ids)
        rescue[scheme]={'prior_relu_failure_count':len(bad),'leaky_rescued_count':len(rescued),'rescue_fraction':len(rescued)/len(bad) if bad else None,'rescued_seeds':rescued,'non_rescued_failure_count':len(notrescued),'early_dead_ge2_rescued':early_ge2(rescued),'early_dead_ge2_non_rescued':early_ge2(notrescued),'fraction_early_dead_ge2_rescued':early_ge2(rescued)/len(rescued) if rescued else None,'fraction_early_dead_ge2_non_rescued':early_ge2(notrescued)/len(notrescued) if notrescued else None}
    result={'experiment':'XOR causal ReLU dead-unit rescue with LeakyReLU','runtime':'numpy_fallback_no_pytorch','dataset':{'inputs':X.tolist(),'labels':Y.reshape(-1).astype(int).tolist()},'architecture':'2->4->1','optimizer':{'name':'Adam','learning_rate':LR,'steps':STEPS,'full_batch':True},'leaky_relu_negative_slope':SLOPE,'seeds':SEEDS,'checkpoints':CHECKPOINTS,'summaries':summaries,'paired_tests':paired,'rescue_analysis':rescue,'runs':rows}
    Path(__file__).with_name('metrics.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8'); print(json.dumps({'rows':len(rows),'rescue':{s:rescue[s]['rescue_fraction'] for s in rescue}},indent=2))
if __name__=='__main__': main()
