"""Leakage-free 2022-23 training / 2024 holdout comparison.
Run: python improved.py --data PATH_TO_DATA --out results
Requires adjacent baseline.py and pandas, numpy, pyarrow, lightgbm, scikit-learn.
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
    ap.add_argument('--out', type=Path, default=Path('annual_comparison_results'))
    ap.add_argument('--min-capacity-kw',type=float,default=20000)
    ap.add_argument('--threads',type=int,default=4)
    a=ap.parse_args()
    if (a.out/'report.json').exists(): raise ValueError('Use a fresh output directory')
    a.out.mkdir(parents=True,exist_ok=True)
    originals={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__).with_name('baseline.py'),Path(__file__).with_name('improved.py')]}
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
        fit=masks['fit'] & day & (d.ts<=pd.Timestamp('2022-12-31 17:00',tz='Asia/Seoul')).to_numpy(); valid=masks['val'] & day
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
    predictions={k:np.zeros(masks['test'].sum()) for k in ['baseline','improved']}
    for r,(X,y,day,cap) in cache.items():
        train=masks['train'] & day & (gen.ts<=pd.Timestamp('2023-12-31 17:00',tz='Asia/Seoul')).to_numpy()
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
    comparison.to_csv(a.out/'comparison.csv',index=False)
    pd.DataFrame(results).T.to_csv(a.out/'metrics.csv')
    for path,h in originals.items(): assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h
    versions={n:importlib.metadata.version(n) for n in ['numpy','pandas','lightgbm','scikit-learn','pyarrow']}
    (a.out/'versions.json').write_text(json.dumps(versions,indent=2))
    text='# 2022–2023 학습 / 2024 전체 평가\n\n'+comparison.to_string(index=False)+'\n\n'
    text+='2022년 학습, 2023년 검증으로만 설정 선택 후 2022~2023년 재학습. 2024년은 채점에만 사용. KST 우측닫힘 2024-01-01 01:00~2025-01-01 00:00, 8784시간. 첫 예측 전날 17시까지 학습. seed=42, 스레드=4.\n\n'
    text+='기본: 600회, 학습률 0.06, num_leaves=127. 개선: 최대 1000회/60회 조기종료, (31,0.05)/(63,0.03)/(127,0.03) 비교. MAE 조기종료 후 G1/G2 검증 벌금으로 선택. 두 모델 모두 L2 학습 목적함수.\n\n'
    text+='추가 변수: 발전소 숫자 ID, A/B 운량·기온 차이, 기온-이슬점, 태양고도 cos값×A/B 맑음 비율, 태양고도×UV, B풍속 m/s. 기존 improved.py 방식을 재현한다. ID의 순서 의미는 보장되지 않으며 풍속 단위변환은 새 정보가 아니다. 추가 탐색은 2024년 점수를 보고 수행하지 않았다.\n\n'
    text+='G1/G2 current_group 고정, 각각 20000kW 이상 확인. 벌금=0.10×Σ SMP×k×|그룹 예측합-실적합|. k는 종료시각 11~15시 3, 나머지 1. 실적이 없는 행은 예측·실적 합계에서 함께 제외. 관측 부분합의 가상 벌금이며 공식 전체 연간 점수가 아니다. 예측 파일에는 전체 253×8784행을 저장한다. 실제 실적은 학습 입력 변수에 포함하지 않는다.\n\n'
    text+='실행: python annual_comparison.py --data DATA_DIR --out NEW_RESULT_DIR (baseline.py를 코드와 같은 폴더에 두고 requirements.txt 설치). 원본 baseline.py/improved.py SHA256 보존 확인. 전체 2023년을 설정 선택에 사용하므로 팀원도 같은 구간과 관측 마스크, seed, 버전으로 맞춰야 한다.\n'
    text+='\n선택 설정: '+str(configurations[best])+'; 횟수: '+str(iterations)+'\n'
    text+='\n총벌금 감소율: '+str(report['penalty_reduction_percent'])+'%\n'
    (a.out/'README.md').write_text(text,encoding='utf-8')
    print(comparison.to_string(index=False),flush=True)
    print(json.dumps(results,indent=2),flush=True)

if __name__=='__main__': main()


