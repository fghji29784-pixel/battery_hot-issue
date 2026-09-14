"""데모: 15분 SDM 곡선으로 3일 ΔOCV를 예측하는 원리와, 무엇이 성패를 가르는지."""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr
from sdm_demo import *

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
line = lambda s: print("\n" + "="*76 + f"\n {s}\n" + "="*76)

cells = make_cells(); y = make_target(cells); cells["dOCV_3d"] = y
tr_ids = np.arange(N_TRAY); RNG.shuffle(tr_ids)
m_tr = cells.tray.isin(tr_ids[:int(N_TRAY*.7)]).values
m_te = ~m_tr

def fit_eval(F, cols=None, model=None):
    X = F.fillna(F.median()).replace([np.inf,-np.inf], 0)
    cols = cols or list(X.columns)
    m = model or HistGradientBoostingRegressor(random_state=0)
    m.fit(X.loc[m_tr, cols], y[m_tr]); p = m.predict(X.loc[m_te, cols])
    return p, np.sqrt(np.mean((p-y[m_te])**2)), pearsonr(p, y[m_te])[0]

line("1. 물리 — 왜 예측이 가능한가")
print("""  SDM과 3일 ΔOCV는 같은 물리량(내부 누설전류 I_sd)을 다르게 잰 것이다.

     3일 ΔOCV  = (I_sd x 72h / 용량) x dOCV/dSOC     ... 느리게, 적분해서
     SDM 정착전류 = I_sd                              ... 빠르게, 직접

  따라서 I_sd를 알아내면 ΔOCV를 계산할 수 있다. 예측의 근거는 이것뿐이다.""")
print(f"\n  확인: 진짜 I_sd 와 3일 ΔOCV 의 상관 r = "
      f"{pearsonr(cells.I_sd_eff, y)[0]:+.3f}   (정의상 거의 1)")

line("2. 문제 — 15분 시점의 전류는 I_sd가 아니다")
print(f"""  측정되는 것:  I(t) = I_sd + A1·exp(-t/T1) + A2·exp(-t/T2) + 노이즈
  긴 시정수 T2 중앙값 = {cells.tau2.median()/3600:.1f} 시간  (Rout=1옴 기준, Keysight)
  15분 시점에 과도 성분이 {np.exp(-15*60/cells.tau2.median())*100:.0f}% 남아 있음""")
t, I = make_curves(cells); F = extract_features(t, I)
for n_, v in [("15분 시점 전류", F.y_end), ("지수 피팅 외삽값", F.fit_Iinf)]:
    m = np.isfinite(v)
    print(f"    {n_:16s} vs 3일 ΔOCV  →  r = {pearsonr(v[m], y[m])[0]:+.3f}")
print("""
  → 측정값이 과도 성분에 묻혀 있다. 그리고 15분으로 4시간짜리 감쇠를
     피팅하는 것은 수학적으로 불안정해서, 지수 외삽도 잘 듣지 않는다.""")

line("3. ★ 핵심 — 출력저항 Rout를 낮추면 어떻게 되는가")
print("""  Keysight:  T = (Rout + Rser) x Ceff
             Rout를 1옴에서 0.1~0.2옴으로 낮추면 시정수가 수배 짧아진다.
  같은 15분 측정인데 T2만 바뀌면 예측 성능이 어떻게 달라지는가?\n""")
print(f"  {'Rout(추정)':>12} {'T2':>9} {'15분/T2':>9} {'단일값 r':>10} {'다변량 r':>10} {'RMSE[mV]':>10}")
print("  " + "-"*64)
rout_rows = []
for rout, sc in [(3.0, 2.75), (1.0, 1.0), (0.5, 0.55), (0.2, 0.25), (0.1, 0.14)]:
    tt, II = make_curves(cells, tau_scale=sc)
    FF = extract_features(tt, II)
    r1 = pearsonr(FF.y_end, y)[0]
    _, rmse, rM = fit_eval(FF)
    tau2h = cells.tau2.median()*sc/3600
    rout_rows.append((rout, tau2h, rmse, rM, r1))
    print(f"  {rout:>10.1f}옴 {tau2h:>8.1f}h {15/60/tau2h:>9.2f} "
          f"{r1:>10.3f} {rM:>10.3f} {rmse:>10.4f}")
print("""
  → 15분이 시정수보다 충분히 길어야 예측이 산다.
     Rout를 낮추는 것이 모델을 바꾸는 것보다 효과가 크다.
     ★ 그래서 AI 이전에 장비 설정부터 확인해야 한다.""")

line("4. 모델 비교 (Rout=0.2옴 조건, 트레이 단위 분할)")
tt, II = make_curves(cells, tau_scale=0.25); FF = extract_features(tt, II)
for n_, cols, mdl in [("① 15분 시점 전류 1개",  ["y_end"],    Ridge()),
                      ("② 지수 피팅 외삽 1개",  ["fit_Iinf"], Ridge()),
                      ("③ 곡선 전체 피처",      None,         None)]:
    p, rmse, r = fit_eval(FF, cols, mdl)
    print(f"    {n_:22s}  RMSE {rmse:.4f} mV   r = {r:+.3f}")
    if cols is None: pred = p
print("\n  → 단일 값보다 곡선 전체 피처가 낫다 (Liu 2022와 같은 결론)")

line("5. 트레이 μ+3σ 판정 — 현행 규칙 그대로 적용")
te = pd.DataFrame(dict(tray=cells.tray[m_te].values, true=y[m_te],
                       pred=pred, defect=cells.is_defect[m_te].values))
def flag(c):
    g = te.groupby("tray")[c]
    return (te[c] > g.transform("mean") + 3*g.transform("std")).values
ft, fp_ = flag("true"), flag("pred")
tp, fn, fp = (ft&fp_).sum(), (ft&~fp_).sum(), (~ft&fp_).sum()
print(f"    3일 실측 기준 불량 : {ft.sum()}개      15분 예측 기준 불량 : {fp_.sum()}개")
print(f"    검출 {tp} / 미검 {fn} / 과검 {fp}       일치율 {tp/max(ft.sum(),1)*100:.0f}%")
k = max(int(len(te)*.01), 1); top = np.argsort(-te.pred.values)[:k]
print(f"    예측 상위 1%({k}셀) 안의 실제 불량 {te.defect.values[top].sum()}개 "
      f"(무작위 기대 {te.defect.sum()*k/len(te):.2f}개)")

line("6. 몇 분이면 충분한가  (Rout=0.2옴 조건)")
curve = []
for mins in [1, 3, 5, 7, 10, 15]:
    tt, II = make_curves(cells, minutes=mins, tau_scale=0.25)
    _, rmse, r = fit_eval(extract_features(tt, II))
    curve.append((mins, rmse, r)); print(f"    {mins:2d}분 → RMSE {rmse:.4f} mV,  r = {r:+.3f}")

# ── 그림 ──────────────────────────────────────────────────────────────
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .3})
fig, ax = plt.subplots(1, 4, figsize=(18, 3.9))
for i in RNG.choice(len(I), 30, replace=False):
    ax[0].plot(t/60, I[i], lw=.7, alpha=.55,
               color="crimson" if cells.is_defect[i] else "steelblue")
ax[0].set(xlabel="time [min]", ylabel="SDM current [uA]",
          title="(1) 15-min SDM curves\n(red = defect)")
ro, th, rm, rM, r1 = zip(*rout_rows)
ax[1].plot(th, rM, "o-", label="multivariate"); ax[1].plot(th, r1, "s--", label="single point")
ax[1].set(xlabel="time constant T2 [h]", ylabel="correlation r",
          title="(2) Lower Rout -> shorter T2\n-> better prediction"); ax[1].legend()
ax[2].scatter(te.pred, te.true, s=5, alpha=.3)
lim = [min(te.true.min(), te.pred.min()), max(te.true.max(), te.pred.max())]
ax[2].plot(lim, lim, "k--", lw=.8)
ax[2].set(xlabel="predicted dOCV [mV]", ylabel="measured dOCV [mV]",
          title=f"(3) 15-min prediction vs 3-day\nr = {pearsonr(te.pred, te.true)[0]:.3f}")
mm, rr, _ = zip(*curve)
ax[3].plot(mm, rr, "o-"); ax[3].set(xlabel="measurement window [min]",
          ylabel="RMSE [mV]", title="(4) How many minutes\nare enough?")
plt.tight_layout(); plt.savefig(f"{OUT}/demo.png", dpi=130)
print(f"\n  그림 저장: {OUT}/demo.png")
print("\n※ 모두 합성 데이터입니다. 숫자 자체가 아니라 '무엇이 성패를 가르는지'를 보세요.")
