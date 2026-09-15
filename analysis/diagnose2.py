# -*- coding: utf-8 -*-
"""
2차 진단 — 1차에서 순위 상관이 낮게 나온 원인을 특정한다.

  python diagnose2.py "파일.CSV"

확인하는 가설
  H1) 셀마다 시정수 τ가 달라 순위가 뒤집힌다
  H2) 소수의 이상 곡선이 상관을 깨고 있다  (Pearson << Spearman 의 원인)
  H3) 형상 피처(비율)를 쓰면 τ 차이를 보정할 수 있다
  H4) 단일 피처가 아니라 다변량이면 회복된다

원본/셀단위 값은 출력하지 않는다.
"""
import sys, numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr
from scipy.optimize import curve_fit
from diagnose import read_any, to_matrix

def f1(t, Iinf, A, tau): return Iinf - A*np.exp(-t/tau)

def main(path, win_s=900):
    df, enc, sep, hdr = read_any(path)
    t, I, ids, fmt = to_matrix(df)
    o = np.argsort(t); t = t[o] - t[o][0]; I = I[:, o]
    ok = np.isfinite(I).all(axis=1); I = I[ok]
    n, T = I.shape
    Ifin = I[:, -max(T//50,3):].mean(axis=1)
    j = int(np.argmin(np.abs(t-win_s)))
    print("="*74); print(f" 2차 진단   셀 {n}개 / 길이 {t[-1]/3600:.1f}시간 / 창 {win_s//60}분")
    print("="*74)

    # ── H1) 셀별 τ 분포  (전체 곡선으로 개별 피팅) ─────────────────────
    print("\n[H1] 셀마다 시정수가 다른가  — 전체 곡선 개별 피팅")
    print("-"*74)
    taus, iinfs = [], []
    for y in I:
        try:
            p,_ = curve_fit(f1, t, y, p0=[y[-1], y[-1]-y[0], t[-1]/3],
                            bounds=([0,0,60],[np.inf,np.inf,1e7]), maxfev=40000)
            taus.append(p[2]); iinfs.append(p[0])
        except Exception:
            taus.append(np.nan); iinfs.append(np.nan)
    taus = np.array(taus)/3600; iinfs = np.array(iinfs)
    m = np.isfinite(taus)
    print(f"  τ  중앙값 {np.nanmedian(taus):.2f}h,  5~95% {np.nanpercentile(taus,5):.2f}~"
          f"{np.nanpercentile(taus,95):.2f}h,  변동계수 CV = {np.nanstd(taus)/np.nanmedian(taus):.2f}")
    print(f"  I∞ 중앙값 {np.nanmedian(iinfs):.0f}µA, 변동계수 CV = {np.nanstd(iinfs)/np.nanmedian(iinfs):.2f}")
    if m.sum() > 5:
        print(f"  τ 와 I∞ 의 상관  ρ = {spearmanr(taus[m], iinfs[m]).statistic:+.3f}")
    print("""  판정: τ의 CV가 0.3 이상이면 셀마다 시정수가 크게 다르다는 뜻.
        그 경우 이른 시점의 '값'만으로는 최종 순위가 보존되지 않는다.""")

    # ── H2) 이상 곡선이 상관을 깨는가 ──────────────────────────────────
    print("\n[H2] 소수 이상 곡선의 영향  — 잔차 큰 셀을 하나씩 제외")
    print("-"*74)
    v = I[:, j]
    print(f"  {'제외 셀 수':>10} {'Spearman':>10} {'Pearson':>10}")
    print("  " + "-"*34)
    # 표준화 후 선형관계에서 벗어난 정도로 순위
    zx = (v-v.mean())/v.std(); zy = (Ifin-Ifin.mean())/Ifin.std()
    dev = np.abs(zy-zx); order = np.argsort(-dev)
    for k in [0, 1, 2, 3, 5, max(n//10,1)]:
        keep = np.setdiff1d(np.arange(n), order[:k])
        if len(keep) < 8: break
        print(f"  {k:>10} {spearmanr(v[keep],Ifin[keep]).statistic:>10.3f}"
              f" {pearsonr(v[keep],Ifin[keep])[0]:>10.3f}")
    print("""  판정: 2~3개만 빼도 Pearson이 크게 오르면, 전체가 무관한 게 아니라
        소수 이상 곡선 문제. 그 셀들을 따로 확인해야 한다.""")

    # ── H3) 형상 피처 ─────────────────────────────────────────────────
    print("\n[H3] 형상 피처로 시정수 차이를 보정할 수 있는가")
    print("-"*74)
    seg_t, seg = t[:j+1], I[:, :j+1]
    idx = lambda s: int(np.argmin(np.abs(seg_t-s)))
    a, b, c = idx(win_s*0.2), idx(win_s*0.5), idx(win_s)
    eps = 1e-9
    feats = {
        "값 I(창끝)":            seg[:, c],
        "비율 I(끝)/I(20%)":     seg[:, c]/(seg[:, a]+eps),
        "비율 I(끝)/I(50%)":     seg[:, c]/(seg[:, b]+eps),
        "로그기울기 Δln I":       np.log(np.abs(seg[:, c])+eps)-np.log(np.abs(seg[:, a])+eps),
        "값 x 비율":             seg[:, c]*(seg[:, c]/(seg[:, a]+eps)),
        "창내 개별피팅 I∞":       np.nan*np.ones(n),
    }
    fit_i = []
    for y in seg:
        try:
            p,_ = curve_fit(f1, seg_t, y, p0=[y[-1]*3, y[-1], seg_t[-1]*20],
                            bounds=([0,0,60],[np.inf,np.inf,1e7]), maxfev=40000)
            fit_i.append(p[0])
        except Exception: fit_i.append(np.nan)
    feats["창내 개별피팅 I∞"] = np.array(fit_i)
    for nm, x in feats.items():
        g = np.isfinite(x)
        if g.sum() < 8: print(f"    {nm:<18} 계산 실패"); continue
        print(f"    {nm:<18} ρ = {spearmanr(x[g],Ifin[g]).statistic:+.3f}"
              f"   r = {pearsonr(x[g],Ifin[g])[0]:+.3f}")
    print("""  판정: 비율·로그기울기가 '값'보다 높으면, τ 보정이 먹힌다는 뜻.""")

    # ── H4) 다변량 ────────────────────────────────────────────────────
    print("\n[H4] 다변량으로 회복되는가  — leave-one-out 교차검증")
    print("-"*74)
    X = np.column_stack([x for x in feats.values()])
    g = np.isfinite(X).all(axis=1) & np.isfinite(Ifin)
    X, yv = X[g], Ifin[g]
    if len(yv) >= 12:
        from sklearn.linear_model import RidgeCV
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        from sklearn.model_selection import LeaveOneOut, cross_val_predict
        for nm, Xs in [("값 1개만", X[:, [0]]), ("형상 피처 전부", X)]:
            p = cross_val_predict(make_pipeline(StandardScaler(), RidgeCV()),
                                  Xs, yv, cv=LeaveOneOut())
            print(f"    {nm:<14} ρ = {spearmanr(p,yv).statistic:+.3f}"
                  f"   r = {pearsonr(p,yv)[0]:+.3f}"
                  f"   RMSE = {np.sqrt(np.mean((p-yv)**2)):.0f} µA")
    else:
        print("    셀 수가 적어 생략")
    print("""
  판정: '형상 피처 전부'가 '값 1개만'보다 뚜렷이 높으면 다변량이 답.
        둘 다 낮으면 이 창(window)으로는 정보가 부족하다는 뜻.""")
    print("\n" + "="*74)
    print(" 공유용: 각 [H] 블록의 숫자만. 셀 단위 값은 없습니다.")
    print("="*74)

if __name__ == "__main__":
    if len(sys.argv) < 2: print(__doc__)
    else: main(sys.argv[1], int(sys.argv[2])*60 if len(sys.argv) > 2 else 900)
