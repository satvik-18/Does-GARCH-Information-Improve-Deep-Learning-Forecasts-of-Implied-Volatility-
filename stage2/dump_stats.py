import json, numpy as np, pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from arch import arch_model

df = pd.read_csv("data/raw/stage2_final.csv", index_col=0, parse_dates=True)
r, v, g = df["ret"], df["VIX"], df["garch_vol_ann"]
res = arch_model(r, mean="Constant", vol="GARCH", p=1, q=1, dist="normal").fit(disp="off")
rt  = arch_model(r, mean="Constant", vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
w,a,b = res.params["omega"], res.params["alpha[1]"], res.params["beta[1]"]
av, ar_, aq = acf(v,nlags=250,fft=True), acf(r,nlags=40,fft=True), acf(r**2,nlags=40,fft=True)
band = 1.96/np.sqrt(len(df))
adf_v = adfuller(v, regression="c", autolag="AIC")
adf_r = adfuller(r, regression="c", autolag="AIC")
lb_r  = acorr_ljungbox(r,   lags=[10], return_df=True)
lb_q  = acorr_ljungbox(r**2,lags=[10], return_df=True)
lm5   = het_arch(r, nlags=5)
dg, dv = g.diff(), v.diff()
N = {
 "n": len(df), "start": str(df.index.min().date()), "end": str(df.index.max().date()),
 "vix": {k: float(f) for k,f in zip(
    ["mean","median","std","min","max","skew","kurt"],
    [v.mean(),v.median(),v.std(),v.min(),v.max(),stats.skew(v),stats.kurtosis(v,fisher=False)])},
 "ret": {k: float(f) for k,f in zip(
    ["mean","median","std","min","max","skew","kurt"],
    [r.mean(),r.median(),r.std(),r.min(),r.max(),stats.skew(r),stats.kurtosis(r,fisher=False)])},
 "jb_vix_p": float(stats.jarque_bera(v)[1]), "jb_ret_p": float(stats.jarque_bera(r)[1]),
 "acf_vix": {str(L): float(av[L]) for L in [1,5,10,22,66,125,250]}, "band": float(band),
 "acf_ret": {str(L): float(ar_[L]) for L in [1,2,5,10,22]},
 "acf_sq":  {str(L): float(aq[L])  for L in [1,2,5,10,22]},
 "adf_vix": [float(adf_v[0]), float(adf_v[1])], "adf_ret": [float(adf_r[0]), float(adf_r[1])],
 "lb_ret": [float(lb_r.lb_stat.iloc[0]), float(lb_r.lb_pvalue.iloc[0])],
 "lb_sq":  [float(lb_q.lb_stat.iloc[0]), float(lb_q.lb_pvalue.iloc[0])],
 "arch_lm": [float(lm5[0]), float(lm5[1])],
 "garch": {"omega":float(w),"alpha":float(a),"beta":float(b),"sum":float(a+b),
   "se_omega":float(res.std_err["omega"]),"se_alpha":float(res.std_err["alpha[1]"]),
   "se_beta":float(res.std_err["beta[1]"]),
   "t_omega":float(res.tvalues["omega"]),"t_alpha":float(res.tvalues["alpha[1]"]),
   "t_beta":float(res.tvalues["beta[1]"]),
   "halflife":float(np.log(.5)/np.log(a+b)), "lr_ann":float(np.sqrt(w/(1-a-b))*np.sqrt(252)),
   "ll":float(res.loglikelihood),"bic":float(res.bic)},
 "garch_t": {"alpha":float(rt.params["alpha[1]"]),"beta":float(rt.params["beta[1]"]),
   "sum":float(rt.params["alpha[1]"]+rt.params["beta[1]"]),"nu":float(rt.params["nu"]),
   "bic":float(rt.bic)},
 "gvol": {"mean":float(g.mean()),"median":float(g.median()),"min":float(g.min()),"max":float(g.max())},
 "gap": {"mean":float((v-g).mean()),"pct":float(100*(v-g).mean()/g.mean()),
   "share_pos":float(100*((v-g)>0).mean()),"min":float((v-g).min()),
   "min_date":str((v-g).idxmin().date()),"gmax_date":str(g.idxmax().date()),
   "vix_at_gmax":float(v[g.idxmax()]),"worst_ret":float(r.min()),"worst_date":str(r.idxmin().date())},
 "corr": {"level":float(g.corr(v)),"spearman":float(g.corr(v,method="spearman")),
   "log":float(np.log(g).corr(np.log(v))),"diff":float(dg.corr(dv)),
   "lead1":float(dg.shift(-1).corr(dv))},
}
json.dump(N, open("numbers.json","w"), indent=1)
print(json.dumps(N, indent=1)[:2000])
