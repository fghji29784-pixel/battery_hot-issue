# -*- coding: utf-8 -*-
"""
예측 모델 — 15분 구간으로 48시간 최종 전류를 실제로 예측하고 오차를 잰다.

  python predict.py "파일.CSV"              # 기본 15분
  python predict.py "파일.CSV" 30           # 30분 구간
  python predict.py "파일.CSV" 15 --plot    # 그림 저장

1차/2차 진단이 '상관이 있는가'를 봤다면, 이것은 '실제로 예측하고
얼마나 틀리는가'를 잰다. 표본이 적으므로 leave-one-out 교차검증을 쓴다.
(모든 예측값은 그 셀을 학습에서 제외한 상태에서 만들어진 값이다)

타깃에 대한 주의
  현재 타깃은 '48시간 SDM 최종 전류'이며, 최종 목표인 '3일 전압강하'의
  대리 타깃이다. 두 값의 연결은 아직 검증되지 않았다.
"""
import sys, numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from diagnose import read_any, to_matrix
from sweep import find_outliers, feats


def main(path, win_min=15, do_plot=False):
    df, *_ = read_any(path)
    t, I, ids, fmt = to_matrix(df)
    o = np.argsort(t); t = t[o]-t[o][0]; I = I[:, o]
    I = I[np.isfinite(I).all(axis=1)]
    flags, reasons = find_outliers(t, I)
    I = I[~flags]; n, T = I.shape
    y = I[:, -max(T//50,3):].mean(axis=1)          # 타깃: 48시간 최종 전류
    w = win_min*60
    F = feats(t, I, w)
    X = np.column_stack(list(F.values())); names = list(F.keys())

    print("="*76)
    print(f" 예측 모델 — {win_min}분 구간 → 48시간 최종 전류")
    print("="*76)
    print(f"  셀 {n}개 (이상 {flags.sum()}개 제외) / 피처 {X.shape[1]}개")
    print(f"  타깃 분포: 중앙값 {np.median(y):.1f} µA, 범위 {y.min():.0f}~{y.max():.0f},"
          f" 표준편차 {y.std():.1f}")

    print("\n" + "-"*76); print(" 예측 성능 (leave-one-out 교차검증)"); print("-"*76)
    print(f"  {'모델':<22} {'RMSE':>9} {'MAE':>9} {'R²':>8} {'ρ':>8}  {'평균오차율':>9}")
    print("  " + "-"*70)
    base = np.full_like(y, np.median(y))
    results = {}
    cands = [
        ("기준선 (중앙값)",      None,                                     None),
        ("단순: 값 1개",          make_pipeline(StandardScaler(), RidgeCV()), [names.index("값")]),
        ("단순: 비율 1개",        make_pipeline(StandardScaler(), RidgeCV()), [names.index("비율50")]),
        ("선형: 전체 피처",       make_pipeline(StandardScaler(), RidgeCV()), None),
        ("비선형: 랜덤포레스트",  RandomForestRegressor(n_estimators=400, min_samples_leaf=2,
                                                       random_state=0), None),
    ]
    for nm, mdl, cols in cands:
        if mdl is None:
            p = base
        else:
            Xs = X if cols is None else X[:, cols]
            g = np.isfinite(Xs).all(axis=1)
            p = np.full(n, np.nan)
            p[g] = cross_val_predict(mdl, Xs[g], y[g], cv=LeaveOneOut())
        m = np.isfinite(p)
        rmse = np.sqrt(np.mean((p[m]-y[m])**2)); mae = np.mean(np.abs(p[m]-y[m]))
        r2 = 1 - np.sum((p[m]-y[m])**2)/np.sum((y[m]-y[m].mean())**2)
        rho = spearmanr(p[m], y[m]).statistic if len(set(np.round(p[m],6))) > 1 else np.nan
        mape = np.mean(np.abs(p[m]-y[m])/np.abs(y[m]))*100
        results[nm] = p
        print(f"  {nm:<22} {rmse:>7.1f}µA {mae:>7.1f}µA {r2:>8.3f} {rho:>8.3f} {mape:>8.1f}%")
    print("""
  · RMSE/MAE 단위는 µA.  R²는 중앙값 예측 대비 설명력 (0이면 기준선과 동일)
  · 모든 예측은 해당 셀을 학습에서 뺀 상태에서 만든 값""")

    # 최적 모델로 판정 시뮬레이션
    best_nm = max([k for k in results if k != "기준선 (중앙값)"],
                  key=lambda k: -np.sqrt(np.nanmean((results[k]-y)**2)))
    p = results[best_nm]; m = np.isfinite(p)
    print("-"*76); print(f" 트레이 평균+3σ 판정 적용  (모델: {best_nm})"); print("-"*76)
    def flag(v):
        return v > v.mean() + 3*v.std()
    ft, fp = flag(y[m]), flag(p[m])
    tp = (ft & fp).sum(); fn = (ft & ~fp).sum(); fpos = (~ft & fp).sum()
    print(f"  48시간 실측 기준 불량 : {ft.sum()}개")
    print(f"  {win_min}분 예측 기준 불량 : {fp.sum()}개")
    print(f"    검출 {tp} / 미검 {fn} / 과검 {fpos}")
    if ft.sum(): print(f"    검출률 {tp/ft.sum()*100:.0f}%")
    for q in (0.05, 0.10, 0.20):
        k = max(int(round(len(p[m])*q)), 1)
        top_t = set(np.argsort(-y[m])[:k]); top_p = set(np.argsort(-p[m])[:k])
        print(f"  상위 {q*100:>2.0f}% ({k}셀) 재현율: {len(top_t&top_p)}/{k}"
              f" = {len(top_t&top_p)/k*100:.0f}%")
    print("""
  ※ 트레이가 하나뿐이면 위 판정은 참고용이다.
     실제로는 트레이별로 임계값을 따로 계산해야 한다.""")

    print("\n" + "-"*76); print(" 피처 기여도 (랜덤포레스트 기준)"); print("-"*76)
    g = np.isfinite(X).all(axis=1)
    rf = RandomForestRegressor(n_estimators=400, min_samples_leaf=2, random_state=0).fit(X[g], y[g])
    for nm, imp in sorted(zip(names, rf.feature_importances_), key=lambda x: -x[1]):
        print(f"    {nm:<10} {'█'*int(imp*40):<40} {imp*100:>5.1f}%")

    if do_plot:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        p2 = results[best_nm]; m2 = np.isfinite(p2)
        plt.figure(figsize=(5.2,5))
        plt.scatter(p2[m2], y[m2], s=26, alpha=.75, edgecolor="k", linewidth=.4)
        lo, hi = min(p2[m2].min(), y[m2].min()), max(p2[m2].max(), y[m2].max())
        pad = (hi-lo)*.06
        plt.plot([lo-pad,hi+pad],[lo-pad,hi+pad],"k--",lw=.9)
        plt.xlabel(f"predicted from {win_min} min [uA]")
        plt.ylabel("measured at 48 h [uA]")
        rmse = np.sqrt(np.mean((p2[m2]-y[m2])**2))
        plt.title(f"{win_min}-min prediction (LOO CV)\nRMSE {rmse:.0f} uA")
        plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(f"predict_{win_min}min.png", dpi=140)
        print(f"\n  그림 저장: predict_{win_min}min.png")

    print("\n" + "="*76)
    print(" 공유용: 성능 표 + 판정 결과 + 피처 기여도.  셀 단위 값은 없습니다.")
    print("="*76)


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    if not a: print(__doc__)
    else: main(a[0], int(a[1]) if len(a) > 1 else 15, "--plot" in sys.argv)
