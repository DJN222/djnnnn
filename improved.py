"""Leakage-free 2022-23 training / 2024 holdout comparison.
Run: python improved.py --data PATH_TO_DATA --out results
Requires adjacent baseline.py and pandas, numpy, pyarrow, lightgbm, scikit-learn.
Missing generation is NEVER treated as zero; scores cover observed site-hours.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import baseline as b

def extra(d):
    d['site_code'] = d.site_id.str.replace('PV', '', regex=False).astype('float32')
    d['cloud_diff'] = d.wa_cloud - d.wb_cloud
    d['temp_diff'] = d.wa_temp - d.wb_temp
    d['dryness'] = d.wb_temp - d.wb_dew_point
    d['sun_cloud_a'] = d.cosz.clip(lower=0) * (1-d.wa_cloud/100)
    d['sun_cloud_b'] = d.cosz.clip(lower=0) * (1-d.wb_cloud/100)
    d['sun_uv'] = d.cosz.clip(lower=0) * d.wb_uv_idx
    d['wind_b_ms'] = d.wb_wind_speed / 3.6
    return b.FEATS + ['site_code','cloud_diff','temp_diff','dryness','sun_cloud_a','sun_cloud_b','sun_uv','wind_b_ms']

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=Path('results'))
    ap.add_argument('--min-capacity-kw',type=float,default=20000)
    ap.add_argument('--threads',type=int,default=4)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    sites=pd.read_csv(a.data/'sites.csv')
    sites['group']=sites.current_group.map({'A':'G1','B':'G2'})
    assert sites.site_id.nunique()==253 and sites.group.notna().all()
    caps=sites.groupby('group').capacity_kw.sum()
    assert len(caps)==2 and caps.min()>=a.min_capacity_kw, caps
    print('Group capacities:',caps.to_dict(),flush=True)
    sites[['site_id','group']].to_csv(a.out/'composition.csv',index=False)
    gen=pd.read_parquet(a.data/'train/generation.parquet')
    gen.ts=pd.to_datetime(gen.ts).dt.tz_convert('Asia/Seoul')
    # Right-closed hourly intervals: Jan 1 00:00 belongs to previous year.
    gen=gen[(gen.ts>pd.Timestamp('2022-01-01',tz='Asia/Seoul')) & (gen.ts<=pd.Timestamp('2025-01-01',tz='Asia/Seoul'))].reset_index(drop=True)
    assert not gen.duplicated(['site_id','ts']).any()
    year=(gen.ts-pd.Timedelta(minutes=30)).dt.year
    masks={'fit':year.eq(2022).to_numpy(),'val':year.eq(2023).to_numpy(),'train':year.le(2023).to_numpy(),'test':year.eq(2024).to_numpy()}
    prices=pd.read_csv(a.data/'train/smp.csv'); prices.ts=pd.to_datetime(prices.ts)
    meta=gen.merge(sites[['site_id','group','capacity_kw']],on='site_id',how='left',sort=False).merge(prices,on='ts',how='left',sort=False)
    assert meta[['site_id','ts']].equals(gen[['site_id','ts']])
    assert meta.smp.notna().all()
    meta['k']=np.where(meta.ts.dt.hour.between(11,15),3,1)
    def score(pred,mask,save=None):
        z=meta.loc[mask,['site_id','ts','group','energy','smp','k']].copy()
        z['prediction']=pred
        g=z.groupby(['group','ts']).agg(actual=('energy','sum'),prediction=('prediction','sum'),n_sites=('site_id','nunique'),smp=('smp','first'),k=('k','first')).reset_index()
        g['penalty']=.1*g.smp*g.k*abs(g.prediction-g.actual)
        expected=sites.groupby('group').size()
        complete=g.n_sites.eq(g.group.map(expected))
        if save: g.to_csv(a.out/save,index=False)
        return {'penalty_krw_observed':float(g.penalty.sum()),'site_mae_kwh':float(np.abs(z.prediction-z.energy).mean()),'group_mae_kwh_observed':float(abs(g.prediction-g.actual).mean()),'observed_site_hours':len(z),'group_hours':len(g),'complete_group_hours':int(complete.sum()),'complete_group_penalty_krw':float(g.loc[complete,'penalty'].sum()) if complete.any() else None,'min_sites_per_group_hour':int(g.n_sites.min())}
    fa=pd.read_parquet(a.data/'train/forecast_wa.parquet',columns=['wa_station','round','ts']+b.WA_COLS)
    fb=pd.read_parquet(a.data/'train/forecast_wb.parquet',columns=['wb_station','round','ts']+b.WB_COLS)
    for f in (fa,fb): f.ts=pd.to_datetime(f.ts)
    configurations=[dict(num_leaves=31,learning_rate=.05),dict(num_leaves=63,learning_rate=.03),dict(num_leaves=127,learning_rate=.03)]
    common=dict(min_child_samples=200,subsample=.8,subsample_freq=1,colsample_bytree=.8,verbosity=-1,n_jobs=a.threads,random_state=42,metric="l1")
    cache={}; val_predictions=[np.zeros(masks['val'].sum()) for _ in configurations]; iterations={}
    for r in b.ROUNDS:
        d=b.features(gen,sites,fa,fb,None,r)
        assert d[['site_id','ts']].equals(gen[['site_id','ts']])
        feats=extra(d); X=d[feats].astype('float32'); y=(d.energy/d.cap).clip(0,1.2).astype('float32'); day=d.elev.to_numpy()>0
        cache[r]=(X,y,day,d.cap.to_numpy())
        fit=masks['fit'] & day; valid=masks['val'] & day
        for i,p in enumerate(configurations):
            model=lgb.LGBMRegressor(n_estimators=1000,**common,**p)
            model.fit(X.loc[fit],y.loc[fit],eval_set=[(X.loc[valid],y.loc[valid])],eval_metric='l1',callbacks=[lgb.early_stopping(60,first_metric_only=True,verbose=False)])
            iterations[f'{r}_{i}']=int(model.best_iteration_)
            pred=np.maximum(model.predict(X.loc[masks['val']]),0)*d.cap.to_numpy()[masks['val']]
            pred[~day[masks['val']]]=0; val_predictions[i]+=pred/2
            print('validation',r,i,'trees',model.best_iteration_,flush=True)
    validation=[dict(params=p,**score(pred,masks['val'])) for p,pred in zip(configurations,val_predictions)]
    best=int(np.argmin([v['penalty_krw_observed'] for v in validation]))
    print('Selected configuration:',best,validation[best],flush=True)
    predictions={k:np.zeros(masks['test'].sum()) for k in ['baseline','improved']}
    for r,(X,y,day,cap) in cache.items():
        train=masks['train'] & day
        for name in predictions:
            feats=b.FEATS if name=='baseline' else X.columns.tolist()
            params=dict(n_estimators=600,num_leaves=127,learning_rate=.06) if name=='baseline' else dict(n_estimators=iterations[f'{r}_{best}'],**configurations[best])
            model=lgb.LGBMRegressor(**common,**params).fit(X.loc[train,feats],y.loc[train])
            model.booster_.save_model(str(a.out/f'{name}_round{r}.txt'))
            p=np.maximum(model.predict(X.loc[masks['test'],feats]),0)*cap[masks['test']]
            p[~day[masks['test']]]=0; predictions[name]+=p/2
            print('Final fitted:',name,r,flush=True)
    results={name:score(p,masks['test'],f'{name}_2024_groups.csv') for name,p in predictions.items()}
    before=results['baseline']['penalty_krw_observed']; after=results['improved']['penalty_krw_observed']
    report=dict(train_years=[2022,2023],validation_year=2023,validation_training_year=2022,test_year=2024,group_capacity_kw=caps.to_dict(),minimum_capacity_kw=a.min_capacity_kw,validation=validation,chosen_configuration=best,early_stopping_iterations=iterations,results=results,penalty_reduction_krw=before-after,penalty_reduction_percent=100*(before-after)/before if before else None,caveat='Observed site-hours only. Missing generation is excluded, never imputed as zero. This is not the full-fleet official contest score. Groups fixed to baseline. 2024 never used for tuning. Seed fixed to 42 for both models. Early stopping uses MAE; candidates selected using group monetary penalty.')
    (a.out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__': main()

