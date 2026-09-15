# -*- coding: utf-8 -*-
"""
엑셀 한 파일로 끝내는 예측 — 이미 계산된 SDM 피처로 3일 ΔOCV 예측

  python predict_xlsx.py "데이터.xlsx"              # 구조 확인 + 예측
  python predict_xlsx.py "데이터.xlsx" --inspect     # 구조만 확인
  python predict_xlsx.py "폴더"                      # 폴더 안 엑셀 전부
  python predict_xlsx.py "데이터.xlsx" --plot

전제
  파일에 이미 i_5min, i_30min, slope_0_5, slope_0_30 같은 피처가 있으므로
  원시 곡선 없이 이 파일만으로 학습·검증이 가능하다.

자동 인식
  트레이   tray_id
  셀       cell_no (+ device_no, channel_no)
  타깃     Delta OCV ... 컬럼
  피처     i_○min / slope_○_○ / rwiring / delta_v / delta_t / v_init / t_init 등
  라벨     판정등급
  제외     OCV_OCV #○○, End Voltage 등 결과 계열은 누수 위험이 있어 기본 제외

원본/셀단위 값은 출력하지 않는다.
"""
import sys, os, re, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr

TARGET_PAT = re.compile(r"delta\s*ocv", re.I)
I_PAT      = re.compile(r"^i[_\s]*(\d+)\s*min", re.I)
SLOPE_PAT  = re.compile(r"^slope[_\s]*(\d+)[_\s]*(\d+)", re.I)
EXTRA_FEAT = re.compile(r"^(rwiring|delta_v|delta_t|v_init|v_final|t_init|t_final|layer)$", re.I)
LEAK_PAT   = re.compile(r"(ocv_ocv|end\s*voltage|ocv\d|판정|grade|delta\s*ocv)", re.I)
TRAY_PAT   = re.compile(r"(tray)", re.I)
CELLNO_PAT = re.compile(r"^(cell_no|cell\s*no|셀번호)$", re.I)
CELLID_PAT = re.compile(r"(cell\s*id)", re.I)
GRADE_PAT  = re.compile(r"(판정등급|grade)", re.I)


def read_table(path):
    if str(path).lower().endswith((".xlsx", ".xlsm", ".xls")):
        return pd.read_excel(path)
    for enc in ("utf-8-sig", "cp949", "euc-kr", "utf-8"):
        try: return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError: continue
    return pd.read_csv(path, encoding="latin-1")


def load(spec):
    if os.path.isdir(spec):
        files = sorted(sum([glob.glob(os.path.join(spec, e))
                            for e in ("*.xlsx","*.xlsm","*.xls","*.csv","*.CSV")], []))
    else:
        files = sorted(glob.glob(spec)) or [spec]
    print(f"  파일 {len(files)}개 읽는 중...")
    df = pd.concat([read_table(f) for f in files], ignore_index=True)
    df.columns = [str(c).strip() for c in df.columns]
    return df, len(files)


def classify(df):
    tray  = next((c for c in df.columns if TRAY_PAT.search(c)), None)
    cno   = next((c for c in df.columns if CELLNO_PAT.search(c)), None)
    cid   = next((c for c in df.columns if CELLID_PAT.search(c)), None)
    grade = next((c for c in df.columns if GRADE_PAT.search(c)), None)
    tgts  = [c for c in df.columns if TARGET_PAT.search(c)]
    feats, times = [], {}
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]): continue
        if LEAK_PAT.search(c): continue
        m = I_PAT.search(c)
        if m: feats.append(c); times.setdefault(int(m.group(1)), []).append(c); continue
        m = SLOPE_PAT.search(c)
        if m: feats.append(c); times.setdefault(int(m.group(2)), []).append(c); continue
        if EXTRA_FEAT.search(c): feats.append(c)
    return dict(tray=tray, cno=cno, cid=cid, grade=grade, targets=tgts,
                feats=feats, times=times)


def inspect(df, info, nfile):
    print("="*78); print(" 파일 구조"); print("="*78)
    print(f"  {df.shape[0]:,}행 x {df.shape[1]}열  (파일 {nfile}개)")
    print(f"\n  트레이   : {info['tray']}")
    print(f"  셀 번호  : {info['cno']}")
    print(f"  셀 ID    : {info['cid']}")
    print(f"  판정등급 : {info['grade']}")
    print(f"\n  타깃 후보 ({len(info['targets'])}개):")
    for c in info["targets"]:
        v = pd.to_numeric(df[c], errors="coerce")
        print(f"    {c:<34} 중앙값 {v.median():.3f}  범위 {v.min():.3f}~{v.max():.3f}")
    print(f"\n  피처 {len(info['feats'])}개")
    if info["times"]:
        print(f"  시간대별 피처:")
        for m in sorted(info["times"]):
            print(f"    {m:>4}분 → {', '.join(info['times'][m])}")
    other = [c for c in info["feats"] if not I_PAT.search(c) and not SLOPE_PAT.search(c)]
    if other: print(f"  기타 피처: {', '.join(other)}")
    excl = [c for c in df.columns if LEAK_PAT.search(c)]
    print(f"\n  누수 위험으로 제외한 컬럼 ({len(excl)}개): {', '.join(excl[:8])}"
          + (" ..." if len(excl) > 8 else ""))


def main(spec, do_plot=False, only_inspect=False, target=None):
    df, nfile = load(spec)
    info = classify(df)
    inspect(df, info, nfile)
    if only_inspect: return
    if not info["targets"]:
        print("\n  !! 타깃(Delta OCV) 컬럼을 못 찾았습니다."); return

    tcol = target or info["targets"][-1]
    print("\n" + "="*78); print(f" 예측 — 타깃: {tcol}"); print("="*78)
    D = df.copy()
    D["_y"] = pd.to_numeric(D[tcol], errors="coerce")
    D["_tray"] = D[info["tray"]].astype(str) if info["tray"] else "ALL"
    F = info["feats"]
    D = D.dropna(subset=["_y"] + F)
    n, ntray = len(D), D["_tray"].nunique()
    print(f"  유효 {n:,}셀 / 트레이 {ntray}개 / 피처 {len(F)}개")
    print(f"  타깃 분포: 중앙값 {D._y.median():.3f}, 5~95% {D._y.quantile(.05):.3f}"
          f"~{D._y.quantile(.95):.3f}, 최대 {D._y.max():.3f}")
    if ntray < 3:
        print("  !! 트레이가 3개 미만이라 트레이 단위 교차검증이 어렵습니다.")

    D["_yn"] = D._y - D.groupby("_tray")._y.transform("median")

    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.linear_model import RidgeCV
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    cv = GroupKFold(n_splits=min(5, max(ntray, 2)))
    g  = D["_tray"].values

    def run(cols, y, label):
        base = np.full(n, np.median(y))
        rows = [("기준선(중앙값)", base)]
        for nm, mdl in [("선형", make_pipeline(StandardScaler(), RidgeCV())),
                        ("부스팅", HistGradientBoostingRegressor(random_state=0))]:
            rows.append((nm, cross_val_predict(mdl, D[cols].values, y, cv=cv, groups=g)))
        out = []
        for nm, p in rows:
            rmse = np.sqrt(np.mean((p-y)**2)); mae = np.mean(np.abs(p-y))
            r2 = 1-np.sum((p-y)**2)/np.sum((y-y.mean())**2)
            rho = spearmanr(p, y).statistic if np.ptp(p) > 0 else float("nan")
            out.append((nm, p, rmse, mae, r2, rho))
            print(f"    {label+' · '+nm:<24} {rmse:>8.4f} {mae:>8.4f} {r2:>8.3f} {rho:>8.3f}")
        return out

    print("\n" + "-"*78); print(" [1] 타깃 종류별 성능"); print("-"*78)
    print(f"    {'구성':<24} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'ρ':>8}")
    print("    " + "-"*60)
    res_abs = run(F, D._y.values, "절대")
    print()
    res_nrm = run(F, D._yn.values, "트레이정규화")

    if info["times"]:
        print("\n" + "-"*78); print(" [2] ★ 몇 분이면 충분한가 — 시간대별 피처만 사용"); print("-"*78)
        print(f"    {'사용 구간':<16} {'피처수':>6} {'RMSE':>9} {'R²':>8} {'ρ':>8}")
        print("    " + "-"*52)
        y = D._y.values; cum = []
        for m in sorted(info["times"]):
            cum += info["times"][m]
            p = cross_val_predict(HistGradientBoostingRegressor(random_state=0),
                                  D[cum].values, y, cv=cv, groups=g)
            r2 = 1-np.sum((p-y)**2)/np.sum((y-y.mean())**2)
            print(f"    {str(m)+'분까지':<16} {len(cum):>6} "
                  f"{np.sqrt(np.mean((p-y)**2)):>9.4f} {r2:>8.3f}"
                  f" {spearmanr(p,y).statistic:>8.3f}")
        print("\n    → 어느 구간부터 성능이 포화되는지가 최소 측정시간이다")

    best = max(res_abs[1:], key=lambda r: -r[2])
    print("\n" + "-"*78); print(f" [3] 트레이 평균+3σ 판정  (모델: 절대 · {best[0]})"); print("-"*78)
    D["_p"] = best[1]
    def flag(c):
        gg = D.groupby("_tray")[c]
        return (D[c] > gg.transform("mean") + 3*gg.transform("std")).values
    ft, fp = flag("_y"), flag("_p")
    tp, miss, over = (ft&fp).sum(), (ft&~fp).sum(), (~ft&fp).sum()
    print(f"  실측 기준 불량 {ft.sum()}개 / 예측 기준 불량 {fp.sum()}개")
    print(f"    검출 {tp} / 미검 {miss} / 과검 {over}"
          + (f"   검출률 {tp/ft.sum()*100:.1f}%" if ft.sum() else ""))
    for q in (0.005, 0.01, 0.05):
        k = max(int(round(n*q)), 1)
        a = set(np.argsort(-D._y.values)[:k]); b = set(np.argsort(-D._p.values)[:k])
        print(f"  상위 {q*100:>4.1f}% ({k}셀) 재현율 {len(a&b)}/{k}"
              f" = {len(a&b)/k*100:.0f}%  (무작위 기대 {k*k/n:.1f})")

    if info["grade"]:
        print("\n" + "-"*78); print(" [4] 판정등급별 분포"); print("-"*78)
        gr = D.groupby(D[info["grade"]].astype(str)).agg(
            n=("_y","size"), 실측중앙값=("_y","median"), 예측중앙값=("_p","median"))
        print(gr.to_string())

    print("\n" + "-"*78); print(" [5] 피처 기여도 (상위 12개)"); print("-"*78)
    from sklearn.inspection import permutation_importance
    m = HistGradientBoostingRegressor(random_state=0).fit(D[F], D._y)
    pi = permutation_importance(m, D[F], D._y, n_repeats=5, random_state=0)
    s = max(pi.importances_mean.max(), 1e-12)
    for nm, v in sorted(zip(F, pi.importances_mean), key=lambda x: -x[1])[:12]:
        print(f"    {nm:<22} {'█'*int(max(v,0)/s*30):<30} {v/s*100:>5.1f}%")

    if do_plot:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5.4,5.2))
        plt.scatter(D._p, D._y, s=5, alpha=.25, edgecolor="none")
        lo, hi = D._y.quantile(.001), D._y.quantile(.999)
        plt.plot([lo,hi],[lo,hi],"k--",lw=.9); plt.xlim(lo,hi); plt.ylim(lo,hi)
        plt.xlabel("predicted [mV]"); plt.ylabel("measured delta OCV [mV]")
        plt.title(f"n={n:,}  rho={spearmanr(D._p,D._y).statistic:.3f}")
        plt.grid(alpha=.3); plt.tight_layout(); plt.savefig("predict_xlsx.png", dpi=140)
        print("\n  그림 저장: predict_xlsx.png")

    print("\n" + "="*78)
    print(" 공유용: [1]~[5] 의 숫자.  셀 단위 값은 없습니다.")
    print("="*78)


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    if not a: print(__doc__)
    else: main(a[0], "--plot" in sys.argv, "--inspect" in sys.argv,
               a[1] if len(a) > 1 else None)
