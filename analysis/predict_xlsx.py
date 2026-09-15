# -*- coding: utf-8 -*-
"""
엑셀 기반 예측 — 15분 SDM 피처로 3일 ΔOCV 예측

  python predict_xlsx.py "데이터.xlsx" --inspect      # 구조·진단만
  python predict_xlsx.py "데이터.xlsx"                # 예측
  python predict_xlsx.py "데이터.xlsx" --plot
  python predict_xlsx.py "폴더"                       # 폴더 안 전부
  python predict_xlsx.py "데이터.xlsx" --use-opt      # 보정값/z_score 도 피처로 사용

처리하는 두 가지 함정
  ① 15분만 측정해도 30분까지 값이 채워지는 경우
     → 뒤쪽 값이 변하지 않으면 그 시점 이후는 무효로 보고,
       구간별 평가에 실제로 그 구간까지 측정된 셀만 쓴다.
  ② SDM 전류가 전압대·온도에 따라 음수도 나오고 트레이별 분포도 다름
     → 트레이 내 정규화 피처를 자동 생성하고,
       전압·온도·배선저항을 함께 피처로 넣는다. 비율·로그는 쓰지 않는다.

원본/셀단위 값은 출력하지 않는다.
"""
import sys, os, re, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr

I_PAT     = re.compile(r"^i[_\s]*(\d+)\s*min$", re.I)
SLOPE_PAT = re.compile(r"^slope[_\s]*0[_\s]*(\d+)$", re.I)
TARGET_PAT= re.compile(r"delta\s*ocv", re.I)
COND_PAT  = re.compile(r"^(rwiring|v_init|v_final|delta_v|t_init|t_final|delta_t|layer|dummy_l\d+)$", re.I)
OPT_PAT   = re.compile(r"(보정값|z_score)", re.I)
LEAK_PAT  = re.compile(r"(ocv_ocv|^ocv\d|end\s*voltage|charge_end|판정|grade|delta\s*ocv|^y$)", re.I)
TRAY_PAT  = re.compile(r"tray", re.I)
CELL_PAT  = re.compile(r"^(cell_no|cell\s*id|lot\s*id|device_no|channel_no)$", re.I)


def read_table(p):
    if str(p).lower().endswith((".xlsx",".xlsm",".xls")): return pd.read_excel(p)
    for e in ("utf-8-sig","cp949","euc-kr","utf-8"):
        try: return pd.read_csv(p, encoding=e)
        except UnicodeDecodeError: continue
    return pd.read_csv(p, encoding="latin-1")


def load(spec):
    if os.path.isdir(spec):
        fs = sorted(sum([glob.glob(os.path.join(spec,e))
                         for e in ("*.xlsx","*.xlsm","*.xls","*.csv","*.CSV")], []))
    else:
        fs = sorted(glob.glob(spec)) or [spec]
    df = pd.concat([read_table(f) for f in fs], ignore_index=True)
    df.columns = [str(c).strip() for c in df.columns]
    return df, len(fs)


def measured_upto(V, mins, rtol=1e-9):
    """셀별 실제 측정 종료 시점.

    장비가 측정을 조기 종료하면 이후 구간에 같은 값이 그대로 채워진다.
    뒤에서부터 값이 (사실상) 변하지 않는 구간을 잘라내 실제 종료 시점을 얻는다.
    """
    mins = np.asarray(mins)
    n, k = V.shape
    same = np.isclose(V[:, 1:], V[:, :-1], rtol=rtol, atol=0.0, equal_nan=True)
    # 뒤에서부터 연속으로 같은 개수
    run = np.zeros(n, dtype=int)
    for j in range(k-2, -1, -1):
        run = np.where(same[:, j], run+1, 0) if j == k-2 else \
              np.where(same[:, j] & (run == (k-2-j)), run+1, run)
    idx = np.clip(k-1-run, 0, k-1)
    return mins[idx]


def main(spec, do_plot=False, only_inspect=False, use_opt=False):
    df, nfile = load(spec)
    print("="*80); print(f" 데이터 {df.shape[0]:,}행 x {df.shape[1]}열  (파일 {nfile}개)"); print("="*80)

    icols  = sorted([c for c in df.columns if I_PAT.match(c)],     key=lambda c:int(I_PAT.match(c).group(1)))
    scols  = sorted([c for c in df.columns if SLOPE_PAT.match(c)], key=lambda c:int(SLOPE_PAT.match(c).group(1)))
    mins   = [int(I_PAT.match(c).group(1)) for c in icols]
    conds  = [c for c in df.columns if COND_PAT.match(c) and pd.api.types.is_numeric_dtype(df[c])]
    opts   = [c for c in df.columns if OPT_PAT.search(c) and pd.api.types.is_numeric_dtype(df[c])]
    tgts   = [c for c in df.columns if TARGET_PAT.search(c)]
    leaks  = [c for c in df.columns if LEAK_PAT.search(c)]
    tray   = next((c for c in df.columns if TRAY_PAT.search(c)), None)

    print(f"  전류 피처 {len(icols)}개  ({mins[0]}~{mins[-1]}분)")
    print(f"  기울기 피처 {len(scols)}개")
    print(f"  조건 피처 {len(conds)}개: {', '.join(conds)}")
    print(f"  타깃: {tgts[-1] if tgts else '못 찾음'}   트레이: {tray}")
    print(f"  보정값/z_score {len(opts)}개  →  {'피처로 사용' if use_opt else '기본 제외 (--use-opt 로 포함)'}")
    print(f"  누수 위험 제외 {len(leaks)}개: {', '.join(leaks[:6])}{' ...' if len(leaks)>6 else ''}")
    if not tgts or not icols:
        print("\n  !! 타깃 또는 전류 피처를 못 찾았습니다."); return
    tcol = tgts[-1]

    D = df.copy()
    D["_y"]    = pd.to_numeric(D[tcol], errors="coerce")
    D["_tray"] = D[tray].astype(str) if tray else "ALL"
    for c in icols+scols+conds+opts: D[c] = pd.to_numeric(D[c], errors="coerce")
    D = D.dropna(subset=["_y"]+icols)
    V = D[icols].values
    D["_upto"] = measured_upto(V, mins)

    # ── ① 실제 측정 길이 ──────────────────────────────────────────────
    print("\n" + "-"*80); print(" [진단 ①] 셀별 실제 측정 길이 (뒤쪽 동일값 반복 검출)"); print("-"*80)
    vc = D["_upto"].value_counts().sort_index()
    for m, c in vc.items():
        print(f"    {m:>3}분까지 측정: {c:>7,}셀  ({c/len(D)*100:>5.1f}%)")
    if len(vc) > 1:
        print(f"""
    → 측정 길이가 셀마다 다릅니다. 구간별 성능을 비교할 때는
      그 구간까지 실제로 측정된 셀만 써야 합니다. 아래 [2]에서 그렇게 합니다.""")
    else:
        print("    → 모든 셀이 같은 길이로 측정되었습니다.")

    # ── ② 음수·트레이 분포 ────────────────────────────────────────────
    print("\n" + "-"*80); print(" [진단 ②] 음수 비율과 트레이 간 분포 차이"); print("-"*80)
    print(f"    {'시점':>6} {'음수 비율':>10} {'중앙값':>12} {'트레이간 분산 비중':>18}")
    print("    " + "-"*52)
    for c, m in zip(icols, mins):
        if m not in (mins[0], 10, 15, 20, mins[-1]): continue
        v = D[c]
        bt = D.groupby("_tray")[c].transform("median")
        icc = np.var(bt) / max(np.var(v), 1e-30)
        print(f"    {m:>4}분 {np.mean(v<0)*100:>9.1f}% {np.median(v):>12.4g} {icc*100:>17.1f}%")
    print("""
    → 트레이간 분산 비중이 크면, 전류 절대값보다 트레이 내 상대값이 중요하다는 뜻.
      아래에서 트레이 정규화 피처를 자동 생성해 함께 평가합니다.""")

    if conds:
        print("\n    조건 변수와 전류의 상관 (마지막 시점 기준)")
        last = icols[-1]
        for c in conds:
            if D[c].nunique() < 3: continue
            print(f"      {c:<12} ρ = {spearmanr(D[c], D[last], nan_policy='omit').statistic:+.3f}")
        print("      → 전압·온도와 상관이 크면 보정이 필요하다는 근거")

    if only_inspect: return

    # ── 피처 구성 ─────────────────────────────────────────────────────
    def build(cols_i, cols_s):
        use = cols_i + cols_s + conds + (opts if use_opt else [])
        X = D[use].copy()
        for c in cols_i + cols_s:                    # 트레이 내 정규화본 추가
            X[c+"_tn"] = D[c] - D.groupby("_tray")[c].transform("median")
        return X

    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.linear_model import RidgeCV
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    def evaluate(X, y, g, tag, quiet=False):
        ntr = pd.Series(g).nunique()
        cv = GroupKFold(n_splits=min(5, max(ntr, 2)))
        out = {}
        for nm, mdl in [("기준선", None),
                        ("선형", make_pipeline(StandardScaler(), RidgeCV())),
                        ("부스팅", HistGradientBoostingRegressor(random_state=0))]:
            p = np.full(len(y), np.median(y)) if mdl is None else \
                cross_val_predict(mdl, X.values, y, cv=cv, groups=g)
            rmse = np.sqrt(np.mean((p-y)**2))
            r2 = 1-np.sum((p-y)**2)/np.sum((y-y.mean())**2)
            rho = spearmanr(p, y).statistic if np.ptp(p) > 0 else float("nan")
            out[nm] = (p, rmse, r2, rho)
            if not quiet:
                print(f"    {tag+' · '+nm:<26} {rmse:>9.4f} {r2:>8.3f} {rho:>8.3f}")
        return out

    print("\n" + "-"*80); print(" [1] 전체 피처 성능  (트레이 단위 교차검증)"); print("-"*80)
    print(f"    {'구성':<26} {'RMSE':>9} {'R²':>8} {'ρ':>8}")
    print("    " + "-"*54)
    full = D["_upto"] >= mins[-1]
    sub = D[full] if full.sum() >= 50 else D
    Xs = build(icols, scols).loc[sub.index]
    ys, gs = sub["_y"].values, sub["_tray"].values
    print(f"    (대상 {len(sub):,}셀 / 트레이 {sub['_tray'].nunique()}개)")
    res = evaluate(Xs, ys, gs, "절대")
    yn = (sub["_y"] - sub.groupby("_tray")["_y"].transform("median")).values
    print()
    evaluate(Xs, yn, gs, "트레이정규화")

    # ── ★ 몇 분이면 충분한가 ─────────────────────────────────────────
    print("\n" + "-"*80)
    print(" [2] ★ 몇 분이면 충분한가  (각 구간까지 실제 측정된 셀만 사용)")
    print("-"*80)
    print(f"    {'구간':>6} {'대상 셀':>9} {'트레이':>7} {'RMSE':>9} {'R²':>8} {'ρ':>8}")
    print("    " + "-"*54)
    for m in mins:
        if m % max(1, len(mins)//10) and m not in (mins[0], 10, 15, 20, mins[-1]): continue
        ok = D["_upto"] >= m
        if ok.sum() < 60: continue
        S = D[ok]
        ci = [c for c, mm in zip(icols, mins) if mm <= m]
        cs = [c for c in scols if int(SLOPE_PAT.match(c).group(1)) <= m]
        X = build(ci, cs).loc[S.index]
        r = evaluate(X, S["_y"].values, S["_tray"].values, "", quiet=True)
        _, rmse, r2, rho = r["부스팅"]
        print(f"    {m:>4}분 {ok.sum():>9,} {S['_tray'].nunique():>7} {rmse:>9.4f} {r2:>8.3f} {rho:>8.3f}")
    print("\n    → 성능이 포화되는 구간이 최소 필요 측정시간")

    # ── 판정 ─────────────────────────────────────────────────────────
    best = min([k for k in res if k != "기준선"], key=lambda k: res[k][1])
    p = res[best][0]
    print("\n" + "-"*80); print(f" [3] 트레이 평균+3σ 판정  (모델: {best})"); print("-"*80)
    S = sub.copy(); S["_p"] = p
    def fl(c):
        g = S.groupby("_tray")[c]
        return (S[c] > g.transform("mean") + 3*g.transform("std")).values
    ft, fp = fl("_y"), fl("_p")
    tp, ms, ov = (ft&fp).sum(), (ft&~fp).sum(), (~ft&fp).sum()
    print(f"  실측 불량 {ft.sum()}개 / 예측 불량 {fp.sum()}개"
          f"  →  검출 {tp} / 미검 {ms} / 과검 {ov}"
          + (f"  검출률 {tp/ft.sum()*100:.1f}%" if ft.sum() else ""))
    for q in (0.005, 0.01, 0.05):
        k = max(int(round(len(S)*q)), 1)
        a = set(np.argsort(-S["_y"].values)[:k]); b = set(np.argsort(-S["_p"].values)[:k])
        print(f"  상위 {q*100:>4.1f}% ({k}셀) 재현율 {len(a&b)}/{k}"
              f" = {len(a&b)/k*100:.0f}%  (무작위 기대 {k*k/len(S):.1f})")

    print("\n" + "-"*80); print(" [4] 피처 기여도 (상위 12개)"); print("-"*80)
    from sklearn.inspection import permutation_importance
    mdl = HistGradientBoostingRegressor(random_state=0).fit(Xs, ys)
    pi = permutation_importance(mdl, Xs, ys, n_repeats=5, random_state=0)
    s = max(pi.importances_mean.max(), 1e-12)
    for nm, v in sorted(zip(Xs.columns, pi.importances_mean), key=lambda x:-x[1])[:12]:
        print(f"    {nm:<20} {'█'*int(max(v,0)/s*30):<30} {v/s*100:>5.1f}%")

    if do_plot:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5.4,5.2))
        plt.scatter(S["_p"], S["_y"], s=5, alpha=.25, edgecolor="none")
        lo, hi = S["_y"].quantile(.001), S["_y"].quantile(.999)
        plt.plot([lo,hi],[lo,hi],"k--",lw=.9); plt.xlim(lo,hi); plt.ylim(lo,hi)
        plt.xlabel("predicted [mV]"); plt.ylabel("measured delta OCV [mV]")
        plt.title(f"n={len(S):,}  rho={spearmanr(S['_p'],S['_y']).statistic:.3f}")
        plt.grid(alpha=.3); plt.tight_layout(); plt.savefig("predict_xlsx.png", dpi=140)
        print("\n  그림 저장: predict_xlsx.png")

    print("\n" + "="*80); print(" 공유용: [진단①②] + [1]~[4].  셀 단위 값은 없습니다."); print("="*80)


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    if not a: print(__doc__)
    else: main(a[0], "--plot" in sys.argv, "--inspect" in sys.argv, "--use-opt" in sys.argv)
