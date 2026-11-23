"""Read-only benchmark analysis; never edits the manuscript or source CSVs.

Fixed 19-method, 3-dataset, 8-metric benchmark. Bootstrap image IDs within
dataset, sharing each draw across all methods and metrics. Recompute means,
ranks and normalized summaries in every replicate. Permutation exchanges two
methods' entire 8-metric vectors within each image pair, separately by dataset,
then recomputes the pair's ranks relative to the other 17 fixed methods.
"""
from pathlib import Path
from itertools import combinations
import csv,json,hashlib,time
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
SUP=HERE.parent
OUT=Path.cwd()/'rank_results'
SOURCE=SUP/'table_4_means.csv'
METHODS=['DenseFuse','SAGE','TGFuse','MaeFuse','MetaFusion','SwinFuse','DCMIF','DRFusion','CNLFusion','ControlFusion','DPOFusion','DiTFuse','IF-FILM','LDFusion','TeRF','Text-DiFuse','Text-IF','TextFusion','VGDCFusion']
DATASETS=['TNO','LLVIP','RoadScene']
METRICS=['EN','SD','SF','AG','MI','SSIM','CC','SCD']
N_BOOT=10000;N_PERM=10000;SEED=20261027

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def holm(ps):
    ps=np.asarray(ps);ix=np.argsort(ps);out=np.empty(len(ps));run=0.
    for k,i in enumerate(ix):run=max(run,(len(ps)-k)*ps[i]);out[i]=min(run,1.)
    return out
def boot_ci(x):return np.percentile(x,[2.5,97.5],axis=0)
def rank_desc(a):
    return 1+(a[:,:,None,:]>a[:,None,:,:]).sum(axis=1)+.5*((a[:,:,None,:]==a[:,None,:,:]).sum(axis=1)-1)
def spearman(a,b):
    ar=pd.Series(a).rank(method='average').to_numpy();br=pd.Series(b).rank(method='average').to_numpy()
    return float(np.corrcoef(ar,br)[0,1])

def main():
    OUT.mkdir(exist_ok=True)
    start=time.monotonic();source_sha=sha(SOURCE)
    X=[];ids={};means=[];files={}
    for ds in DATASETS:
        frames=[]
        for method in METHODS:
            name='Densefuse' if method=='DenseFuse' else method
            p=SUP/'per_image'/name/f'{ds}.csv'
            # Copied supplementary package may use normalized directory names.
            if not p.exists():p=SUP/'per_image'/method/f'{ds}.csv'
            files[p.relative_to(SUP).as_posix()]=sha(p)
            f=pd.read_csv(p);f['id']=f['Image'].map(lambda v:Path(str(v)).stem)
            assert not f['id'].duplicated().any();frames.append(f.set_index('id'))
        keys=sorted(frames[0].index)
        assert all(set(f.index)==set(keys) for f in frames)
        a=np.stack([f.loc[keys,METRICS].to_numpy(float) for f in frames],axis=1)
        assert np.isfinite(a).all();X.append(a);means.append(a.mean(axis=0));ids[ds]=keys
    means=np.stack(means)
    ranks=rank_desc(means)
    avg_r=ranks.mean(axis=(0,2));score=1-(avg_r-1)/18
    baseline_win=(means[:,8:,:,None]>means[:,:8,:].transpose(0,2,1)[:,None,:,:]).mean(axis=(0,2,3))
    fullwin=np.full(19,np.nan);fullwin[8:]=baseline_win
    # Min-max sensitivity is illustrative, not a replacement metric definition.
    minmax=(means-means.min(axis=1,keepdims=True))/(means.max(axis=1,keepdims=True)-means.min(axis=1,keepdims=True))
    minmaxscore=minmax.mean(axis=(0,2))
    boot_r=[];boot_cells=[];boot_win=[]
    signs=[]
    rng=np.random.default_rng(SEED)
    for di,(ds,a) in enumerate(zip(DATASETS,X)):
        n=len(a)
        weights=rng.multinomial(n,np.full(n,1/n),size=N_BOOT).astype(float)/n
        bm=(weights@a.reshape(n,-1)).reshape(N_BOOT,19,8)
        br=rank_desc(bm)
        boot_r.append(br.mean(axis=2));boot_cells.append(br)
        bw=(bm[:,8:,:,None]>bm[:,:8,:].transpose(0,2,1)[:,None,:,:]).mean(axis=(2,3))
        boot_win.append(bw)
        signs.append(np.random.default_rng(SEED+1000+di).choice([-1.,1.],size=(N_PERM,n)))
        print('bootstrap',ds,n,flush=True)
    brank=np.mean(boot_r,axis=0);bscore=1-(brank-1)/18;bwin=np.mean(boot_win,axis=0)
    rc=boot_ci(brank);sc=boot_ci(bscore);wc=boot_ci(bwin)
    best_all=np.isclose(brank,brank.min(axis=1,keepdims=True),rtol=0,atol=1e-12)
    first_share_all=(best_all/best_all.sum(axis=1,keepdims=True)).mean(axis=0)
    best_vlm=np.isclose(brank[:,8:],brank[:,8:].min(axis=1,keepdims=True),rtol=0,atol=1e-12)
    first_share_vlm=(best_vlm/best_vlm.sum(axis=1,keepdims=True)).mean(axis=0)
    final_ranks=pd.DataFrame(brank).rank(axis=1,method='average').to_numpy()
    out=[]
    for i,m in enumerate(METHODS):
        out.append({'method':m,'average_rank':avg_r[i],'normalized_score':score[i],
          'score_ci_low':sc[0,i],'score_ci_high':sc[1,i],'average_rank_ci_low':rc[0,i],'average_rank_ci_high':rc[1,i],
          'resample_first_share_all':first_share_all[i],'resample_first_share_vlm':first_share_vlm[i-8] if i>=8 else np.nan,
          'prob_top3_overall':np.mean(final_ranks[:,i]<=3),'baseline_winrate':fullwin[i],
          'baseline_win_ci_low':wc[0,i-8] if i>=8 else np.nan,'baseline_win_ci_high':wc[1,i-8] if i>=8 else np.nan,
          'minmax_aggregate':minmaxscore[i]})
    summary=pd.DataFrame(out).sort_values('average_rank');summary.to_csv(OUT/'normalized_bootstrap_summary.csv',index=False)
    cells=[]
    for di,ds in enumerate(DATASETS):
        ci=boot_ci(1-(boot_cells[di]-1)/18)
        for mi,m in enumerate(METHODS):
            for ki,k in enumerate(METRICS):cells.append({'dataset':ds,'method':m,'metric':k,'mean':means[di,mi,ki],'rank':ranks[di,mi,ki],'s':1-(ranks[di,mi,ki]-1)/18,'s_ci_low':ci[0,mi,ki],'s_ci_high':ci[1,mi,ki]})
    pd.DataFrame(cells).to_csv(OUT/'heatmap_cell_bootstrap.csv',index=False)
    # All 171 method pairs: aggregate-score effect, paired CI, paired null test.
    contrasts=[]
    for pairnum,(ia,ib) in enumerate(combinations(range(19),2)):
        obs=score[ia]-score[ib];bd=bscore[:,ia]-bscore[:,ib];ci=boot_ci(bd)
        perm=np.zeros(N_PERM)
        other=[j for j in range(19) if j not in [ia,ib]]
        for di,a in enumerate(X):
            middle=(means[di,ia]+means[di,ib])/2
            half=(signs[di]@(a[:,ia,:]-a[:,ib,:]))/(2*len(a))
            pa=middle+half;pb=middle-half;fixed=means[di,other,:]
            ra=1+(fixed[None,:,:]>pa[:,None,:]).sum(axis=1)+.5*(fixed[None,:,:]==pa[:,None,:]).sum(axis=1)+(pb>pa)+.5*(pb==pa)
            rb=1+(fixed[None,:,:]>pb[:,None,:]).sum(axis=1)+.5*(fixed[None,:,:]==pb[:,None,:]).sum(axis=1)+(pa>pb)+.5*(pa==pb)
            perm+=((rb-ra)/18).mean(axis=1)/3
        p=(np.count_nonzero(np.abs(perm)>=abs(obs)-1e-12)+1)/(N_PERM+1)
        contrasts.append({'method_a':METHODS[ia],'method_b':METHODS[ib],'score_difference_a_minus_b':obs,
           'ci_low':ci[0],'ci_high':ci[1],'p_permutation':p,'bootstrap_prob_a_greater_b':np.mean(bd>0)})
        if pairnum%25==0:print('permutation pairs',pairnum+1,'/171',flush=True)
    pc=pd.DataFrame(contrasts);pc['p_holm_171']=holm(pc['p_permutation']);pc.to_csv(OUT/'normalized_score_pairwise_171.csv',index=False)
    # Leave-one-dataset / metric sensitivity under the original rank convention.
    sensitivity=[]
    for typ,labels,axis in [('dataset',DATASETS,0),('metric',METRICS,2)]:
        for i,label in enumerate(labels):
            reduced=np.delete(ranks,i,axis=axis).mean(axis=(0,2))
            sensitivity.append({'omitted_type':typ,'omitted':label,'best_method':METHODS[int(np.argmin(reduced))],
               'best_vlm':METHODS[8+int(np.argmin(reduced[8:]))], 'rho_all19':spearman(avg_r,reduced),
               'LDFusion_rank':reduced[METHODS.index('LDFusion')],'IF-FILM_rank':reduced[METHODS.index('IF-FILM')]})
    pd.DataFrame(sensitivity).to_csv(OUT/'sensitivity_omissions.csv',index=False)
    report={'seed':SEED,'bootstrap_replicates':N_BOOT,'permutation_replicates':N_PERM,
      'datasets':{d:len(a) for d,a in zip(DATASETS,X)},'methods':19,'metrics':METRICS,
      'sampling':'Within each dataset resample paired image IDs; share draw across all 19 methods and all 8 metrics; datasets and metrics receive equal aggregate weight.',
      'permutation_null':'Within-image exchangeability of two methods entire metric vectors; swap jointly over metrics, independently by dataset. Recompute ranks relative to remaining fixed methods. Two-sided aggregate score statistic. Holm family all 171 method pairs.',
      'inference_scope':'Conditional on the fixed 3 datasets,19 methods,and single-run outputs. Image-pair bootstrap assumes independent sampling units; no scene clustering because a verified scene grouping is not supplied. Does not assess run-to-run seed variability or external perceptual validity.',
      'interval_interpretation':'Rank-based percentile intervals describe conditional resampling uncertainty. Ordinary bootstrap ranking intervals need not have exact nominal coverage near ties. Use original metric paired differences for substantive effect-size and significance claims.',
      'permutation_interpretation':'Aggregate-score label-exchange tests are exploratory diagnostics under the stronger exchangeability null; they are not the inverse of percentile intervals or tests validating the quality metric. Do not infer significance from overlap of marginal intervals.',
      'rho_avg_rank_vs_winrate_11':spearman(-avg_r[8:],baseline_win),
      'rho_rank_score_vs_minmax_score_19':spearman(score,minmaxscore),
      'identity_max_abs_error':float(np.max(np.abs((1-(ranks-1)/18).mean(axis=(0,2))-score))),
      'rank_to_score':'S = 1 - (Rbar - 1) / 18; exact affine transform, not effect-size preservation.',
      'table_4_csv_sha256':source_sha,'source_files_sha256':files,
      'elapsed_seconds':time.monotonic()-start}
    assert sha(SOURCE)==source_sha
    (OUT/'analysis_protocol_and_results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    np.savez_compressed(OUT/'bootstrap_replicates.npz',score=bscore,rank=brank,winrate=bwin,methods=np.array(METHODS))
    print(summary.head(8).to_string(index=False),flush=True)
    print(json.dumps({k:v for k,v in report.items() if k!='source_files_sha256'},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
