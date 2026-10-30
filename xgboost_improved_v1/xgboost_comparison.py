"""Leakage-free 2022-23 training / 2024 holdout comparison.
Run: python xgboost_comparison.py --data PATH_TO_DATA --out results
Requires pandas, numpy, pyarrow, lightgbm, scikit-learn.
Missing generation is NEVER treated as zero; scores cover observed site-hours.
"""
import argparse
import json
import hashlib
import importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import types
from xgboost import XGBRegressor
b=types.ModuleType("baseline"); b.__file__=__file__
exec('"""OIBC 2026 3회차 — 참가자용 기준선.\n\n참조 구성(sites.csv의 253개소 전부 + current_group)을 그대로 쓰고, 하루전 기상예보로\n발전소별 설비이용률을 LightGBM으로 학습해 예측한 뒤 집합 단위로 합산한다.\n라운드 10·17 예보로 각각 예측하고 그 **평균**을 시각당 하나의 입찰로 제출한다.\n학습 구간 데이터만 사용하므로 참가자가 그대로 재현할 수 있다.\n\n    python baseline.py            # composition.csv, forecast.csv 생성\n    python scoring.py --composition composition.csv --forecast forecast.csv\n\n--sites-out 를 주면 발전소별 예측도 저장한다. 구성을 직접 최적화하려면\n집합 합계가 아니라 발전소별 예측이 필요하다 (OIBC2026-문제정의서.md 7.2절).\n"""\nfrom __future__ import annotations\nimport argparse, os, time\nimport numpy as np\nimport pandas as pd\nimport lightgbm as lgb\n\nHERE = os.path.dirname(os.path.abspath(__file__))\nTRAIN_FROM = pd.Timestamp("2023-06-01 01:00", tz="Asia/Seoul")   # 최근 2년만 사용\nROUNDS = (10, 17)\nWA_COLS = ["wa_cloud", "wa_temp", "wa_humidity", "wa_rain",\n           "wa_wind_speed", "wa_ground_press"]\nWB_COLS = ["wb_cloud", "wb_temp", "wb_rel_hum", "wb_uv_idx", "wb_precip_prob",\n           "wb_total_liq", "wb_vis", "wb_wind_speed", "wb_dew_point"]\n\nt0 = time.time()\nlog = lambda m: print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)\n\n\ndef solar(ts: pd.DatetimeIndex, lat: np.ndarray, lon: np.ndarray):\n    """태양 고도·방위 (NOAA 근사, 도 단위). 외부 의존성 없이 계산."""\n    doy = ts.dayofyear.to_numpy()\n    hour = ts.hour.to_numpy() + ts.minute.to_numpy() / 60.0\n    g = 2 * np.pi / 365.0 * (doy - 1 + (hour - 12) / 24.0)\n    eqtime = 229.18 * (0.000075 + 0.001868*np.cos(g) - 0.032077*np.sin(g)\n                       - 0.014615*np.cos(2*g) - 0.040849*np.sin(2*g))     # 분\n    decl = (0.006918 - 0.399912*np.cos(g) + 0.070257*np.sin(g)\n            - 0.006758*np.cos(2*g) + 0.000907*np.sin(2*g)\n            - 0.002697*np.cos(3*g) + 0.00148*np.sin(3*g))                 # rad\n    # KST(UTC+9) 기준 진태양시. ts는 우측닫힘이므로 구간 중앙(-30분)을 쓴다.\n    tst = (hour - 0.5) * 60 + eqtime + 4 * (lon - 135.0)\n    ha = np.radians(tst / 4.0 - 180.0)\n    la = np.radians(lat)\n    cosz = np.sin(la)*np.sin(decl) + np.cos(la)*np.cos(decl)*np.cos(ha)\n    cosz = np.clip(cosz, -1, 1)\n    elev = 90.0 - np.degrees(np.arccos(cosz))\n    az = np.degrees(np.arctan2(-np.sin(ha),\n                               np.tan(decl)*np.cos(la) - np.sin(la)*np.cos(ha))) % 360\n    return elev.astype("float32"), az.astype("float32"), cosz.astype("float32")\n\n\ndef features(gen: pd.DataFrame | None, sites: pd.DataFrame, fa: pd.DataFrame,\n             fb: pd.DataFrame, index: pd.DatetimeIndex | None, r: int) -> pd.DataFrame:\n    """(site, ts) 격자에 예보·태양기하를 붙인 특징 행렬."""\n    if gen is not None:\n        base = gen[["site_id", "ts", "energy"]]\n    else:\n        base = pd.MultiIndex.from_product([sites.site_id, index],\n                                          names=["site_id", "ts"]).to_frame(index=False)\n    d = base.merge(sites, on="site_id", how="left")\n    d = d.merge(fa[fa["round"] == r].drop(columns="round"), on=["wa_station", "ts"], how="left")\n    d = d.merge(fb[fb["round"] == r].drop(columns="round"), on=["wb_station", "ts"], how="left")\n    ts = pd.DatetimeIndex(d.ts)\n    d["elev"], d["azim"], d["cosz"] = solar(ts, d.lat.to_numpy(), d.lon.to_numpy())\n    doy = ts.dayofyear.to_numpy()\n    d["doy_sin"] = np.sin(2*np.pi*doy/365.25).astype("float32")\n    d["doy_cos"] = np.cos(2*np.pi*doy/365.25).astype("float32")\n    d["hour"] = ts.hour.to_numpy().astype("int8")\n    d["cap"] = d.capacity_kw.astype("float32")\n    return d\n\n\nFEATS = WA_COLS + WB_COLS + ["elev", "azim", "cosz", "doy_sin", "doy_cos",\n                              "hour", "cap", "lat", "lon"]\n\n\n', b.__dict__)

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
    ap.add_argument('--out', type=Path, default=Path('annual_comparison_results'))
    ap.add_argument('--min-capacity-kw',type=float,default=20000)
    ap.add_argument('--threads',type=int,default=4)
    a=ap.parse_args()
    if (a.out/'xgboost_report.json').exists(): raise ValueError('Use a fresh output directory')
    a.out.mkdir(parents=True,exist_ok=True)
    originals=json.loads(Path(__file__).with_name('original_hashes.json').read_text()) if Path(__file__).with_name('original_hashes.json').exists() else {}
    (a.out/'source_hashes.json').write_text(json.dumps(originals,indent=2))
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
    gen=gen.dropna(subset=['energy']).reset_index(drop=True)
    # Generate ALL 2024 site-hours, even when observed truth is absent.
    cut=pd.Timestamp('2024-01-01',tz='Asia/Seoul')
    ix=pd.date_range(cut+pd.Timedelta(hours=1),pd.Timestamp('2025-01-01',tz='Asia/Seoul'),freq='h')
    grid=pd.MultiIndex.from_product([sites.site_id,ix],names=['site_id','ts']).to_frame(index=False)
    test=grid.merge(gen.loc[gen.ts>cut],on=['site_id','ts'],how='left',validate='one_to_one')
    gen=pd.concat([gen.loc[gen.ts<=cut],test],ignore_index=True)
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
        z=z.dropna(subset=['energy'])
        g=z.groupby(['group','ts']).agg(actual=('energy','sum'),prediction=('prediction','sum'),n_sites=('site_id','nunique'),smp=('smp','first'),k=('k','first')).reset_index()
        g['penalty']=.1*g.smp*g.k*abs(g.prediction-g.actual)
        expected=sites.groupby('group').size()
        complete=g.n_sites.eq(g.group.map(expected))
        if save: g.to_csv(a.out/save,index=False)
        return {'penalty_krw_observed':float(g.penalty.sum()),'site_rmse_kwh':float(np.sqrt(np.mean((z.prediction-z.energy)**2))),'group_rmse_kwh_observed':float(np.sqrt(np.mean((g.prediction-g.actual)**2))),'site_mae_kwh':float(np.abs(z.prediction-z.energy).mean()),'group_mae_kwh_observed':float(abs(g.prediction-g.actual).mean()),'observed_site_hours':len(z),'group_hours':len(g),'complete_group_hours':int(complete.sum()),'complete_group_penalty_krw':float(g.loc[complete,'penalty'].sum()) if complete.any() else None,'min_sites_per_group_hour':int(g.n_sites.min())}
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
        fit=masks['fit'] & day & (d.ts<=pd.Timestamp(f'2022-12-31 {r}:00',tz='Asia/Seoul')).to_numpy(); valid=masks['val'] & day
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
    pd.DataFrame(validation).to_csv(a.out/'validation.csv',index=False)
    (a.out/'frozen_selection.json').write_text(json.dumps(dict(best=best,params=configurations[best],iterations=iterations,seed=42,train_cutoff='2023-12-31T17:00:00+09:00',test_used_for_selection=False),indent=2))
    # XGBoost candidates are fixed before evaluating any 2024 predictions.
    xconfigs=[dict(max_depth=4,learning_rate=.05,subsample=.8,colsample_bytree=.8,min_child_weight=20,reg_lambda=10),dict(max_depth=6,learning_rate=.03,subsample=.8,colsample_bytree=.8,min_child_weight=50,reg_lambda=10),dict(max_depth=8,learning_rate=.03,subsample=.7,colsample_bytree=.9,min_child_weight=100,reg_lambda=20)]
    xv=[np.zeros(masks['val'].sum()) for _ in xconfigs]; xi={}
    for r,(X,y,day,cap) in cache.items():
        fit=masks['fit'] & day & (gen.ts<=pd.Timestamp(f'2022-12-31 {r}:00',tz='Asia/Seoul')).to_numpy()
        valid=masks['val'] & day
        for i,p in enumerate(xconfigs):
            model=XGBRegressor(n_estimators=1000,objective='reg:squarederror',tree_method='hist',n_jobs=a.threads,random_state=42,eval_metric='mae',early_stopping_rounds=60,**p)
            model.fit(X.loc[fit],y.loc[fit],eval_set=[(X.loc[valid],y.loc[valid])],verbose=False)
            xi[f'{r}_{i}']=model.best_iteration+1
            pred=np.maximum(model.predict(X.loc[masks['val']]),0)*cap[masks['val']]
            pred[~day[masks['val']]]=0; xv[i]+=pred/2
            print('XGB validation',r,i,xi[f'{r}_{i}'],flush=True)
    xvalidation=[dict(params=p,**score(pred,masks['val'])) for p,pred in zip(xconfigs,xv)]
    xb=int(np.argmin([v['penalty_krw_observed'] for v in xvalidation]))
    (a.out/'xgb_frozen_selection.json').write_text(json.dumps(dict(best=xb,validation=xvalidation,iterations=xi),indent=2))
    predictions={k:np.zeros(masks['test'].sum()) for k in ['baseline','improved','xgb_baseline','xgb_improved']}
    settings={}
    for r,(X,y,day,cap) in cache.items():
        train=masks['train'] & day & (gen.ts<=pd.Timestamp(f'2023-12-31 {r}:00',tz='Asia/Seoul')).to_numpy()
        for name in predictions:
            feats=b.FEATS if name in ['baseline','xgb_baseline'] else X.columns.tolist()
            if name.startswith('xgb'):
                params=dict(n_estimators=600,max_depth=6,learning_rate=.06,subsample=.8,colsample_bytree=.8) if name=='xgb_baseline' else dict(n_estimators=xi[f'{r}_{xb}'],**xconfigs[xb])
                model=XGBRegressor(objective='reg:squarederror',tree_method='hist',n_jobs=a.threads,random_state=42,**params).fit(X.loc[train,feats],y.loc[train])
                model.save_model(a.out/f'{name}_round{r}.json')
            else:
                params=dict(n_estimators=600,num_leaves=127,learning_rate=.06) if name=='baseline' else dict(n_estimators=iterations[f'{r}_{best}'],**configurations[best])
                model=lgb.LGBMRegressor(**common,**params).fit(X.loc[train,feats],y.loc[train])
                model.booster_.save_model(str(a.out/f'{name}_round{r}.txt'))
            settings[f'{name}_{r}']=dict(params=model.get_params(),features=feats,training_rows=int(train.sum()))
            p=np.maximum(model.predict(X.loc[masks['test'],feats]),0)*cap[masks['test']]
            p[~day[masks['test']]]=0; predictions[name]+=p/2
            print('Final fitted:',name,r,flush=True)
    results={name:score(p,masks['test'],f'{name}_2024_groups.csv') for name,p in predictions.items()}
    before=results['baseline']['penalty_krw_observed']; after=results['improved']['penalty_krw_observed']
    report=dict(train_years=[2022,2023],validation_year=2023,validation_training_year=2022,test_year=2024,group_capacity_kw=caps.to_dict(),minimum_capacity_kw=a.min_capacity_kw,validation=validation,chosen_configuration=best,early_stopping_iterations=iterations,results=results,penalty_reduction_krw=before-after,penalty_reduction_percent=100*(before-after)/before if before else None,caveat='Observed site-hours only. Missing generation is excluded, never imputed as zero. This is not the full-fleet official contest score. Groups fixed to baseline. 2024 never used for tuning. Seed fixed to 42 for both models. Early stopping uses MAE; candidates selected using group monetary penalty.')
    report.update(model_settings=settings,xgb_validation=xvalidation,xgb_chosen=xb,xgb_iterations=xi,official_annual_score=None)
    (a.out/'xgboost_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    rows=[]
    for name,pred in predictions.items():
        z=meta.loc[masks['test'],['site_id','ts','energy']].copy(); z['prediction_kwh']=pred
        z.to_parquet(a.out/f'{name}_2024_sites.parquet',index=False)
        g=pd.read_csv(a.out/f'{name}_2024_groups.csv')
        for group in ['G1','G2','TOTAL']:
            v=g if group=='TOTAL' else g[g.group==group]
            err=v.prediction-v.actual
            rows.append(dict(model=name,group=group,penalty_krw=v.penalty.sum(),group_mae_kwh=err.abs().mean(),group_rmse_kwh=np.sqrt((err**2).mean()),group_hours=len(v)))
    comparison=pd.DataFrame(rows)
    ref=comparison[comparison.model=='baseline'].set_index('group').penalty_krw
    comparison['penalty_reduction_pct']=100*(1-comparison.penalty_krw/comparison.group.map(ref))
    comparison.to_csv(a.out/'xgboost_comparison.csv',index=False)
    metric_rows=[]
    for name in predictions:
        z=pd.read_parquet(a.out/f'{name}_2024_sites.parquet').dropna(subset=['energy'])
        for site,v in z.groupby('site_id'):
            err=v.prediction_kwh-v.energy
            metric_rows.append(dict(model=name,level='site',id=site,n=len(v),mae_kwh=err.abs().mean(),rmse_kwh=np.sqrt((err**2).mean())))
        g=pd.read_csv(a.out/f'{name}_2024_groups.csv')
        for group,v in g.groupby('group'):
            err=v.prediction-v.actual
            metric_rows.append(dict(model=name,level='group',id=group,n=len(v),mae_kwh=err.abs().mean(),rmse_kwh=np.sqrt((err**2).mean())))
    pd.DataFrame(metric_rows).to_csv(a.out/'xgboost_metrics.csv',index=False)
    for path,h in originals.items(): assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h
    versions={n:importlib.metadata.version(n) for n in ['numpy','pandas','lightgbm','xgboost','scikit-learn','pyarrow']}
    (a.out/'versions.json').write_text(json.dumps(versions,indent=2))
    print(comparison.to_string(index=False),flush=True)
    print(json.dumps(results,indent=2),flush=True)

if __name__=='__main__': main()


