"""Identical local 2024 scoring contract for all three team members."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd

def score(data,composition,forecast):
    s=pd.read_csv(data/'sites.csv'); c=pd.read_csv(composition); f=pd.read_csv(forecast)
    if list(c.columns)!=['site_id','group'] or list(f.columns)!=['group','ts','forecast_kwh']: raise ValueError('Exact columns required')
    if len(c)!=253 or not c.site_id.is_unique or set(c.site_id)!=set(s.site_id) or set(c.group)!={'G1','G2'}: raise ValueError('Invalid composition')
    cap=c.merge(s[['site_id','capacity_kw']],on='site_id',validate='one_to_one').groupby('group').capacity_kw.sum()
    if not (cap>=20000).all(): raise ValueError('Both groups require 20 MW')
    if not f.ts.str.endswith('+09:00').all(): raise ValueError('KST +09:00 required')
    f['ts']=pd.to_datetime(f.ts); ix=pd.date_range('2024-01-01 01:00','2025-01-01 00:00',freq='h',tz='Asia/Seoul'); expected=pd.MultiIndex.from_product([['G1','G2'],ix]); keys=pd.MultiIndex.from_frame(f[['group','ts']])
    if len(f)!=17568 or not keys.is_unique or len(expected.difference(keys)): raise ValueError('Require exactly all 2024 group-hours')
    if not np.isfinite(f.forecast_kwh).all() or (f.forecast_kwh<0).any(): raise ValueError('Invalid prediction')
    truth=pd.read_parquet(data/'train/generation.parquet'); truth=truth[truth.ts.isin(ix)]
    if truth.duplicated(['site_id','ts']).any() or not np.isfinite(truth.energy).all(): raise ValueError('Invalid truth')
    panel=truth.pivot(index='ts',columns='site_id',values='energy').reindex(columns=s.site_id).dropna(); mapping=c.set_index('site_id').group
    prices=pd.read_csv(data/'train/smp.csv'); prices['ts']=pd.to_datetime(prices.ts); prices=prices.set_index('ts').reindex(panel.index).smp.to_numpy(); k=np.where(panel.index.hour.isin([11,12,13,14,15]),3,1)
    cost=0.; revenue=float((panel.sum(axis=1).to_numpy()*prices).sum()); groups={}
    for g in ['G1','G2']:
        actual=panel.loc[:,[i for i in panel.columns if mapping[i]==g]].sum(axis=1).to_numpy(); pred=f[f.group==g].set_index('ts').reindex(panel.index).forecast_kwh.to_numpy(); charge=float(.1*np.sum(prices*k*abs(actual-pred))); cost+=charge; groups[g]=charge
    return {'complete_hours':len(panel),'requested_hours':len(ix),'penalty_krw':cost,'settlement_krw':revenue-cost,'by_group_penalty':groups,'capacity_kw':cap.to_dict(),'forecast_sha256':hashlib.sha256(Path(forecast).read_bytes()).hexdigest(),'limitation':'2024 complete fleet hours only, not annual or official score. Different groupings permitted; predictions must match their own composition.'}
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,required=True); p.add_argument('--composition',type=Path,required=True); p.add_argument('--forecast',type=Path,required=True); p.add_argument('--out',type=Path,required=True); a=p.parse_args(); r=score(a.data,a.composition,a.forecast); a.out.write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf8'); print(json.dumps(r,ensure_ascii=False,indent=2))
