# -*- coding: utf-8 -*-
"""
트레이 내 순위 문제로 다시 세우기 — 회귀(R2)가 아니라 순위와 판정을 직접 푼다.

  python rank_tray.py "데이터.xlsx"
  python rank_tray.py "데이터.xlsx" --at=15
  python rank_tray.py "데이터.xlsx" --k=3            # 판정 임계 sigma 배수
  python rank_tray.py "데이터.xlsx" --no-cond        # 조건 피처(온도·전압) 빼고 순위학습

결과는 analysis/results/ 에 자동 저장된다 (--save=경로 / --no-save).
────────────────────────────────────────────────────────────────────────
 왜 다시 세우는가
────────────────────────────────────────────────────────────────────────
 앞선 분석에서 3일 전압강하량 회귀는 결정계수가 음수였다.
 그런데 현장 판정은 '값' 이 아니라 '그 트레이 안에서 튀는가' 다.
 그러면 맞혀야 할 것은 값이 아니라 트레이 내 순위다.

   회귀   : 모든 셀의 mV 를 맞힌다.  트레이 기준선(전압대)이 오차의 대부분을 먹는다.
   순위학습: 같은 트레이 셀끼리 누가 더 나쁜지만 맞힌다. 기준선은 차분으로 사라진다.

 이 스크립트는 세 가지를 한다.
   [1] mu+3sigma 판정의 자기가림(self-masking) 을 수치로 보인다
       — 불량 셀이 자기 트레이의 sigma 를 키워 자기를 임계 밖으로 밀어낸다.
         median/MAD 로 바꾸면 사라지는 문제다.
   [2] 트레이 내 쌍 비교로 선형 순위모델을 학습한다 (RankNet 의 선형판)
       — 가중치가 그대로 해석되므로 발표에서 설명할 수 있다.
   [3] 계층 판정 — 트레이 내 z 와 트레이 간 z 를 함께 본다
       — 트레이 전체가 나쁜 경우는 전자가 못 잡고 후자가 잡는다.

원본/셀단위 값은 출력하지 않는다.
"""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import runlog
from predict_xlsx import load, I_PAT, SLOPE_PAT, COND_PAT, TARGET_PAT, TRAY_PAT, measured_upto
from correct import within_tray_rho, topk_recall, safe_z, r2_linear

GRADE_KEY = "판정등급"


def z_mean(v, tray):
    s = pd.Series(v); g = s.groupby(pd.Series(tray).values)
    return ((s - g.transform("mean")) / g.transform("std").replace(0, np.nan)).values


def z_mad(v, tray):
    s = pd.Series(v); g = s.groupby(pd.Series(tray).values)
    med = g.transform("median")
    mad = g.transform(lambda x: np.median(np.abs(x - np.median(x)))) * 1.4826
    return ((s - med) / mad.replace(0, np.nan)).values


def z_mean_loo(v, tray):
    """자기 자신을 뺀 평균·표준편차로 z 를 계산 (leave-one-out).

    mu+3sigma 판정이 자기 자신을 통계량에 포함시키는 탓에 생기는
    자기가림을 제거한 버전. 두 값의 차이가 곧 자기가림의 크기다.
    """
    s = pd.Series(v).astype(float); g = pd.Series(tray).values
    n = pd.Series(s).groupby(g).transform("count")
    ssum = s.groupby(g).transform("sum"); ssq = (s ** 2).groupby(g).transform("sum")
    m = (ssum - s) / (n - 1)
    var = ((ssq - s ** 2) - (n - 1) * m ** 2) / (n - 2)
    return ((s - m) / np.sqrt(var.clip(lower=1e-300))).values


def main(spec, at=None, ksig=3.0, no_cond=False):
    df, _ = load(spec)
    icols = sorted([c for c in df.columns if I_PAT.match(c)], key=lambda c: int(I_PAT.match(c).group(1)))
    scols = sorted([c for c in df.columns if SLOPE_PAT.match(c)], key=lambda c: int(SLOPE_PAT.match(c).group(1)))
    mins  = [int(I_PAT.match(c).group(1)) for c in icols]
    conds = [] if no_cond else [c for c in df.columns
                                if COND_PAT.match(c) and pd.api.types.is_numeric_dtype(df[c])]
    tgts  = [c for c in df.columns if TARGET_PAT.search(c)]
    tray  = next((c for c in df.columns if TRAY_PAT.search(c)), None)
    gcol  = next((c for c in df.columns if GRADE_KEY in str(c)), None)
    if not icols or not tgts:
        print("  !! 전류 컬럼 또는 타깃(delta OCV)을 못 찾았습니다."); return

    D = df.copy()
    D["_tray"] = D[tray].astype(str) if tray else "ALL"
    for c in icols + scols + conds: D[c] = pd.to_numeric(D[c], errors="coerce")
    D["_y"] = pd.to_numeric(D[tgts[-1]], errors="coerce")
    D = D.dropna(subset=icols + ["_y"]).reset_index(drop=True)
    D["_upto"] = measured_upto(D[icols].values, mins)
    D = D[D["_upto"] >= mins[-1]].reset_index(drop=True)
    if len(D) < 30:
        print(f"\n  !! {mins[-1]}분까지 측정된 셀이 {len(D)}개뿐입니다. 분석을 건너뜁니다.")
        print("     모든 시점의 전류가 같으면 '측정 조기 종료' 로 판정되어 전부 걸러집니다.")
        print("     입력 엑셀의 i_XXmin 컬럼이 시점마다 다른 값인지 확인하십시오.")
        return
    n = len(D); g = D["_tray"].values
    at = at or (15 if 15 in mins else mins[-1])
    acol = f"i_{at}min" if f"i_{at}min" in D.columns else icols[-1]
    y = D["_y"].values
    ng = D[gcol].astype(str).str.strip().str.upper().isin(["E"]).values.astype(int) \
        if gcol else np.zeros(n, int)

    print("=" * 78); print(" 트레이 내 순위 문제로 다시 세우기"); print("=" * 78)
    sz = D.groupby("_tray").size()
    print(f"  {n:,}셀 / 트레이 {D['_tray'].nunique()}개"
          f"  (트레이당 {sz.min()}~{sz.max()}셀, 중앙 {int(sz.median())})")
    if no_cond: print("  [--no-cond] 조건 피처(온도·전압·배선저항)를 빼고 전류 형상만 씁니다.")
    print(f"  평가 시점 {at}분   타깃 {tgts[-1]}"
          + (f"   불량(E) {int(ng.sum())}개" if ng.sum() else ""))

    # ── [1] 자기가림 ──────────────────────────────────────────────
    print("\n" + "-" * 78)
    print(f" [1] mu+{ksig:.0f}sigma 판정의 자기가림 — 불량 셀이 자기 임계값을 키운다")
    print("-" * 78)
    zi, zl, zm = z_mean(y, g), z_mean_loo(y, g), z_mad(y, g)
    f_i, f_l, f_m = (zi > ksig), (zl > ksig), (zm > ksig)
    print(f"    {'판정 규칙':<34}{'적출 셀':>9}{'적출률':>9}"
          + (f"{'E 검출':>9}" if ng.sum() else ""))
    print("    " + "-" * (52 + (9 if ng.sum() else 0)))
    for nm, f in [(f"현행  mu + {ksig:.0f}sigma (자기 포함)", f_i),
                  (f"개선A mu + {ksig:.0f}sigma (자기 제외 LOO)", f_l),
                  (f"개선B median + {ksig:.0f}x1.4826MAD", f_m)]:
        line = f"    {nm:<34}{np.nansum(f):>9,.0f}{np.nanmean(f) * 100:>8.3f}%"
        if ng.sum(): line += f"{int(np.nansum(f & (ng == 1))):>6}/{int(ng.sum())}"
        print(line)
    add = int(np.nansum(f_m & ~f_i))
    print(f"""
    → 자기 자신을 통계량에 넣으면, 크게 튀는 셀일수록 자기 트레이의 sigma 를 키운다.
      셀 수가 적은 트레이일수록 심하다. median/MAD 는 원리적으로 이 영향을 받지 않는다.
      median/MAD 로 바꾸면 현행이 놓치던 셀 {add:,}개가 새로 적출된다.
      ★ 코드 한 줄 교체로 끝나는 개선이다. 측정도 설비도 바꾸지 않는다.""")

    # ── [2] 순위 학습 ─────────────────────────────────────────────
    print("\n" + "-" * 78)
    print(" [2] 트레이 내 쌍 비교 순위학습  (트레이 단위 교차검증)")
    print("-" * 78)
    ci = [c for c, m in zip(icols, mins) if m <= at]
    cs = [c for c in scols if int(SLOPE_PAT.match(c).group(1)) <= at]
    feats = ci + cs + conds
    X = D[feats].astype(float).copy()
    for c in feats:                                   # 트레이 내 중심화
        X[c] = X[c] - X[c].groupby(g).transform("median")
    X = X.fillna(0.0)
    Xv, _, _, dead_f = safe_z(X.values)
    if dead_f.any():
        print(f"    트레이 내 변화가 없어 제외한 피처 {int(dead_f.sum())}개"
              f" (예: {', '.join([f for f, d in zip(feats, dead_f) if d][:4])})")
    yc = y - pd.Series(y).groupby(g).transform("median").values

    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingRegressor

    rng = np.random.default_rng(0)
    idx_by_tray = {t: np.where(g == t)[0] for t in np.unique(g)}

    def make_pairs(rows, n_pair=40000):
        """같은 트레이 안에서만 쌍을 뽑는다. 트레이 기준선은 차분으로 사라진다."""
        rs = set(rows); A, B = [], []
        pools = [v[np.isin(v, list(rs))] for v in idx_by_tray.values()]
        pools = [p for p in pools if len(p) >= 2]
        if not pools: return np.array([], int), np.array([], int)
        w = np.array([len(p) for p in pools], float); w /= w.sum()
        pick = rng.choice(len(pools), n_pair, p=w)
        for pi in pick:
            a, b = rng.choice(pools[pi], 2, replace=False)
            A.append(a); B.append(b)
        return np.array(A), np.array(B)

    cv = GroupKFold(n_splits=min(5, max(D["_tray"].nunique(), 2)))
    p_rank = np.zeros(n); p_gbm = np.zeros(n)
    W = np.zeros(len(feats))
    for tr, te in cv.split(Xv, yc, groups=g):
        A, B = make_pairs(tr)
        if len(A) == 0: continue
        dX = Xv[A] - Xv[B]; dy = (y[A] > y[B]).astype(int)
        keep = np.abs(y[A] - y[B]) > 0
        lr = LogisticRegression(max_iter=2000, C=0.1).fit(dX[keep], dy[keep])
        p_rank[te] = Xv[te] @ lr.coef_[0]; W += lr.coef_[0]
        p_gbm[te] = HistGradientBoostingRegressor(random_state=0).fit(
            Xv[tr], yc[tr]).predict(Xv[te])

    base = [("원시 전류 (현행)", D[acol].values),
            ("트레이 정규화 전류", D[acol].values - pd.Series(D[acol].values).groupby(g).transform("median").values)]
    cand = base + [("쌍비교 선형 순위모델", p_rank), ("트레이중심 부스팅", p_gbm)]

    print(f"    {'점수':<24}{'트레이내 rho':>13}{'트레이내 R2':>12}"
          + (f"{'상위1% E':>10}{'상위5% E':>10}" if ng.sum() else ""))
    print("    " + "-" * (49 + (20 if ng.sum() else 0)))
    for nm, s in cand:
        rw, used = within_tray_rho(s, y, g)
        r2 = r2_linear(s, yc)
        line = f"    {nm:<24}{rw:>13.3f}{r2:>12.3f}"
        if ng.sum():
            t = topk_recall(s, ng)
            line += f"{t[0.01] * 100:>9.0f}%{t[0.05] * 100:>9.0f}%"
        print(line)
    print(f"    (트레이 {used}개가 평가에 쓰였습니다. 20셀 미만 트레이는 제외)")

    print(f"\n    쌍비교 모델 가중치 상위 8개  (부호가 곧 '크면 나쁘다/좋다')")
    for nm_, w_ in sorted(zip(feats, W / max(cv.get_n_splits(), 1)), key=lambda x: -abs(x[1]))[:8]:
        wmax = float(np.abs(W).max()) / max(cv.get_n_splits(), 1)
        bar = "█" * (int(min(abs(w_) / wmax * 22, 22)) if wmax > 0 else 0)
        print(f"      {nm_:<14}{w_:>+8.3f}  {bar}")
    print("""
    → '트레이내 R2' 는 트레이 기준선을 제거한 뒤의 결정계수다.
      원래 회귀의 R2 가 음수였던 것은 트레이 기준선을 맞히지 못해서였다.
      목적(트레이 상대평가)에 맞춰 문제를 다시 세우면 같은 데이터에서 양수가 나온다.""")

    # ── [3] 계층 판정 ─────────────────────────────────────────────
    print("\n" + "-" * 78)
    print(" [3] 계층 판정 — 트레이 내 z 와 트레이 간 z 를 함께 본다")
    print("-" * 78)
    s_best = p_rank if np.std(p_rank) > 0 else D[acol].values
    z_in = z_mad(s_best, g)
    tmed = pd.Series(s_best).groupby(g).median()
    mad_o = float(np.median(np.abs(tmed - tmed.median())) * 1.4826)
    z_of = (tmed - tmed.median()) / mad_o if mad_o > 0 else tmed * 0.0
    z_out = D["_tray"].map(z_of).values
    print(f"    트레이 내 z > {ksig:.0f}   : {int(np.nansum(z_in > ksig)):,}셀"
          f"  ({np.nanmean(z_in > ksig) * 100:.3f}%)")
    print(f"    트레이 간 z > {ksig:.0f}   : {int((z_of > ksig).sum())}트레이"
          f"  {list(z_of[z_of > ksig].index)[:5]}")
    comb = np.fmax(np.nan_to_num(z_in, nan=-9), np.nan_to_num(z_out, nan=-9))
    if ng.sum():
        for nm, s in [("트레이 내 z 만", np.nan_to_num(z_in, nan=-9)),
                      ("계층 (내/간 최대)", comb)]:
            t = topk_recall(s, ng)
            print(f"    {nm:<20} 상위1% {t[0.01] * 100:>5.0f}%   상위5% {t[0.05] * 100:>5.0f}%")
    print("""
    → 현행은 트레이 내 z 만 본다. 그래서 '그 트레이가 통째로 나쁜 경우' 를
      원리적으로 못 잡는다 (다 같이 나쁘면 아무도 튀지 않는다).
      트레이 간 z 를 함께 보면 그 사각지대가 닫힌다.
      단, 트레이 간 비교는 온도 보정이 먼저다. correct.py 의 P2 점수를 쓸 것.""")

    print("\n" + "=" * 78)
    print(""" 정리
   · 지표를 바꾸는 것만으로 결과가 달라진다. 값이 아니라 순위를 맞히면 된다.
   · mu+3sigma 를 median/MAD 로 바꾸는 것은 코드 한 줄이고 즉시 적용 가능하다.
   · 트레이 간 판정을 더하면 현행이 원리적으로 못 보던 사각지대가 닫힌다.""")
    print("=" * 78)


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    if not a: print(__doc__)
    else:
        at, k = None, 3.0
        for x in sys.argv:
            if x.startswith("--at="): at = int(x.split("=")[1])
            if x.startswith("--k="):  k = float(x.split("=")[1])
        sv, en = runlog.parse(sys.argv)
        with runlog.saving("rank_tray", a[0], sys.argv, sv, en):
            main(a[0], at, k, "--no-cond" in sys.argv)
