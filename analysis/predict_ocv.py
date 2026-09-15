# -*- coding: utf-8 -*-
"""
본 예측 모델 — 15분 SDM 곡선으로 3일 전압강하(ΔOCV)를 예측한다.

  python predict_ocv.py curves.csv targets.csv
  python predict_ocv.py curves.csv targets.csv --inspect     # 형식 확인만
  python predict_ocv.py curves.csv targets.csv --plot

입력
  curves.csv  : 15분 SDM 곡선 (TIME + 셀별 컬럼, 또는 cell/time/current 형태)
  targets.csv : 셀ID + 3일 ΔOCV(mV).  있으면 함께 인식:
                트레이ID, 양품/불량 라벨, 장기보관 ΔOCV

핵심 설계
  · 검증 분할은 반드시 트레이 단위 (같은 트레이가 학습·검증에 섞이면 누수)
  · 타깃은 두 가지를 모두 평가
      (a) 절대 ΔOCV
      (b) 트레이 정규화 ΔOCV  = ΔOCV - 해당 트레이 중앙값
  · 판정은 현행 규칙(트레이 평균+3σ)을 예측값에 그대로 적용

원본/셀단위 값은 출력하지 않는다.
"""
import sys, re, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.stats import spearmanr, pearsonr
from diagnose import read_any, to_matrix
from sweep import find_outliers, feats

OCV_PAT  = re.compile(r"(docv|d_ocv|delta.?ocv|ocv.?drop|drop|전압강하|ocv)", re.I)
TRAY_PAT = re.compile(r"(tray|트레이|rack|lot|랏|batch)", re.I)
CELL_PAT = re.compile(r"(cell|셀|channel|ch|slot|serial|barcode|id)", re.I)
LBL_PAT  = re.compile(r"(label|판정|불량|defect|ng|grade|양품)", re.I)
LONG_PAT = re.compile(r"(long|장기|storage|보관)", re.I)


def pick(df, pat, exclude=()):
    for c in df.columns:
        if c in exclude: continue
        if pat.search(str(c)): return c
    return None


def load_targets(path):
    d, enc, sep, hdr = read_any(path)
    cid  = pick(d, CELL_PAT)
    tray = pick(d, TRAY_PAT, exclude={cid})
    lbl  = pick(d, LBL_PAT,  exclude={cid, tray})
    lng  = pick(d, LONG_PAT, exclude={cid, tray, lbl})
    ocv  = pick(d, OCV_PAT,  exclude={cid, tray, lbl, lng})
    print(f"  [타깃 파일] 인코딩 {enc} / {d.shape[0]:,}행 x {d.shape[1]}열")
    print(f"    셀ID={cid}  ΔOCV={ocv}  트레이={tray}  라벨={lbl}  장기보관={lng}")
    if cid is None or ocv is None:
        print("\n  !! 셀ID 또는 ΔOCV 컬럼을 못 찾았습니다. 컬럼 목록:")
        for c in d.columns[:40]: print("      ", c)
        sys.exit(1)
    out = pd.DataFrame({"cid": d[cid].astype(str).str.strip(),
                        "docv": pd.to_numeric(d[ocv], errors="coerce")})
    out["tray"]  = d[tray].astype(str) if tray else "ALL"
    out["label"] = d[lbl] if lbl else np.nan
    out["long"]  = pd.to_numeric(d[lng], errors="coerce") if lng else np.nan
    return out.dropna(subset=["docv"])


def main(curve_path, target_path, win_min=15, do_plot=False, inspect=False):
    print("="*78); print(" 15분 SDM → 3일 ΔOCV 예측"); print("="*78)
    tg = load_targets(target_path)
    df, enc, *_ = read_any(curve_path)
    t, I, ids, fmt = to_matrix(df)
    o = np.argsort(t); t = t[o]-t[o][0]; I = I[:, o]
    print(f"  [곡선 파일] 형식 {fmt} / 셀 {I.shape[0]:,}개 / 시점 {I.shape[1]}개"
          f" / 길이 {t[-1]/60:.1f}분 / 간격 {np.median(np.diff(t)):.0f}초")

    ids = [str(c).strip() for c in ids]
    join = len(set(ids) & set(tg.cid))
    print(f"  [조인] 셀ID 일치 {join:,}개")
    if inspect or join < 10:
        if join < 10:
            print("\n  !! 셀ID 매칭이 거의 안 됩니다. 양쪽 ID 예시를 비교하세요:")
            print("     곡선 파일 :", ids[:5])
            print("     타깃 파일 :", list(tg.cid[:5]))
        return

    good = np.isfinite(I).all(axis=1)
    flags, _ = find_outliers(t, I)
    keep = good & ~flags
    print(f"  [전처리] 결측 {(~good).sum()}개, 이상곡선 {flags.sum()}개 제외"
          f" → {keep.sum():,}개")
    I = I[keep]; ids = [c for c, k in zip(ids, keep) if k]

    F = feats(t, I, min(win_min*60, t[-1]))
    X = pd.DataFrame(F); X["cid"] = ids
    D = X.merge(tg, on="cid", how="inner").dropna(subset=list(F.keys())+["docv"])
    fn = list(F.keys())
    n, ntray = len(D), D.tray.nunique()
    print(f"\n  분석 대상 {n:,}셀 / 트레이 {ntray}개"
          f" / 트레이당 중앙값 {int(D.groupby('tray').size().median())}셀")
    print(f"  ΔOCV: 중앙값 {D.docv.median():.3f} mV,"
          f" 5~95% {D.docv.quantile(.05):.3f}~{D.docv.quantile(.95):.3f},"
          f" 최대 {D.docv.max():.3f}")

    # ── 물리 정합성 ────────────────────────────────────────────────────
    print("\n" + "-"*78); print(" [1] 물리 정합성 — SDM 전류와 ΔOCV가 맞는가"); print("-"*78)
    r_abs = spearmanr(D["값"], D.docv).statistic
    print(f"  15분 SDM 전류  vs  3일 ΔOCV     ρ = {r_abs:+.3f}  (n={n:,})")
    D["docv_n"] = D.docv - D.groupby("tray").docv.transform("median")
    for k in fn: D[k+"_n"] = D[k] - D.groupby("tray")[k].transform("median")
    r_nrm = spearmanr(D["값_n"], D.docv_n).statistic
    print(f"  트레이 정규화 후                ρ = {r_nrm:+.3f}"
          f"   ({'개선' if r_nrm > r_abs else '개선 없음'})")
    print("""  → 정규화 후 상관이 오르면 트레이 공통 교란이 실재한다는 증거""")

    # ── 예측 ──────────────────────────────────────────────────────────
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.linear_model import RidgeCV
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    cv = GroupKFold(n_splits=min(5, ntray)) if ntray >= 2 else 5
    groups = D.tray.values if ntray >= 2 else None

    for tag, ycol, xs in [("절대 ΔOCV", "docv", fn), ("트레이 정규화 ΔOCV", "docv_n", [c+"_n" for c in fn])]:
        y = D[ycol].values
        print("\n" + "-"*78); print(f" [2] 예측 성능 — {tag}  (트레이 단위 {getattr(cv,'n_splits',5)}-fold)")
        print("-"*78)
        print(f"  {'모델':<20} {'RMSE':>9} {'MAE':>9} {'R²':>8} {'ρ':>8}")
        print("  " + "-"*58)
        base = np.full(n, np.median(y))
        cands = [("기준선(중앙값)", None, None),
                 ("단일: 값",       make_pipeline(StandardScaler(), RidgeCV()), [xs[fn.index("값")]]),
                 ("단일: 비율50",   make_pipeline(StandardScaler(), RidgeCV()), [xs[fn.index("비율50")]]),
                 ("선형: 전체",     make_pipeline(StandardScaler(), RidgeCV()), xs),
                 ("부스팅: 전체",   HistGradientBoostingRegressor(random_state=0), xs)]
        preds = {}
        for nm, mdl, cols in cands:
            p = base if mdl is None else cross_val_predict(
                mdl, D[cols].values, y, cv=cv, groups=groups)
            preds[nm] = p
            rmse = np.sqrt(np.mean((p-y)**2)); mae = np.mean(np.abs(p-y))
            r2 = 1-np.sum((p-y)**2)/np.sum((y-y.mean())**2)
            rho = spearmanr(p, y).statistic if np.ptp(p) > 0 else float('nan')
            print(f"  {nm:<20} {rmse:>7.4f}mV {mae:>7.4f}mV {r2:>8.3f} {rho:>8.3f}")
        if ycol == "docv":
            best = max([k for k in preds if k != "기준선(중앙값)"],
                       key=lambda k: -np.mean((preds[k]-y)**2))
            best_pred, best_name = preds[best], best

    # ── 판정 시뮬레이션 ───────────────────────────────────────────────
    print("\n" + "-"*78); print(f" [3] 트레이 평균+3σ 판정  (모델: {best_name})"); print("-"*78)
    D["pred"] = best_pred
    def flag(col):
        g = D.groupby("tray")[col]
        return (D[col] > g.transform("mean") + 3*g.transform("std")).values
    ft, fp = flag("docv"), flag("pred")
    tp, fn_, fpo = (ft&fp).sum(), (ft&~fp).sum(), (~ft&fp).sum()
    print(f"  3일 실측 기준 불량 {ft.sum()}개 / 15분 예측 기준 불량 {fp.sum()}개")
    print(f"    검출 {tp} / 미검 {fn_} / 과검 {fpo}")
    if ft.sum(): print(f"    검출률 {tp/ft.sum()*100:.1f}%   과검률 {fpo/(~ft).sum()*100:.2f}%")
    for q in (0.005, 0.01, 0.05):
        k = max(int(round(n*q)), 1)
        a = set(np.argsort(-D.docv.values)[:k]); b = set(np.argsort(-D.pred.values)[:k])
        print(f"  상위 {q*100:>4.1f}% ({k}셀) 재현율 {len(a&b)}/{k} = {len(a&b)/k*100:.0f}%"
              f"   (무작위 기대 {k*k/n:.1f})")

    # ── 불량 라벨 ─────────────────────────────────────────────────────
    if D.label.notna().any():
        print("\n" + "-"*78); print(" [4] 불량 라벨 평가"); print("-"*78)
        lab = D.label.astype(str).str.upper()
        ng = lab.isin(["NG","1","TRUE","불량","Y","BAD"])
        print(f"  불량 {ng.sum()}개 / 양품 {(~ng).sum():,}개 (불량률 {ng.mean()*100:.3f}%)")
        if 0 < ng.sum():
            for q in (0.005, 0.01, 0.05):
                k = max(int(round(n*q)), 1)
                top = set(np.argsort(-D.pred.values)[:k])
                hit = sum(1 for i in np.where(ng)[0] if i in top)
                print(f"    예측 상위 {q*100:>4.1f}% ({k}셀) 안의 실제 불량"
                      f" {hit}/{ng.sum()}  (무작위 기대 {ng.sum()*k/n:.2f})")

    print("\n" + "-"*78); print(" [5] 피처 기여도"); print("-"*78)
    from sklearn.inspection import permutation_importance
    m = HistGradientBoostingRegressor(random_state=0).fit(D[fn], D.docv)
    pi = permutation_importance(m, D[fn], D.docv, n_repeats=5, random_state=0)
    for nm, v in sorted(zip(fn, pi.importances_mean), key=lambda x: -x[1]):
        s = max(pi.importances_mean.max(), 1e-12)
        print(f"    {nm:<10} {'█'*int(v/s*36):<36} {v/s*100:>5.1f}%")

    if do_plot:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5.4,5.2))
        plt.scatter(D.pred, D.docv, s=5, alpha=.25, edgecolor="none")
        lo, hi = D.docv.quantile(.001), D.docv.quantile(.999)
        plt.plot([lo,hi],[lo,hi],"k--",lw=.9); plt.xlim(lo,hi); plt.ylim(lo,hi)
        plt.xlabel("predicted from 15 min [mV]"); plt.ylabel("measured 3-day dOCV [mV]")
        plt.title(f"n={n:,}  rho={spearmanr(D.pred,D.docv).statistic:.3f}")
        plt.grid(alpha=.3); plt.tight_layout(); plt.savefig("predict_ocv.png", dpi=140)
        print("\n  그림 저장: predict_ocv.png")

    print("\n" + "="*78)
    print(" 공유용: [1]~[5] 의 숫자.  셀 단위 값은 없습니다.")
    print("="*78)


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    if len(a) < 2: print(__doc__)
    else: main(a[0], a[1], int(a[2]) if len(a) > 2 else 15,
               "--plot" in sys.argv, "--inspect" in sys.argv)
