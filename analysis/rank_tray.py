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

  --target="컬럼명"   3일 ΔOCV 컬럼을 직접 지정 (DOCV 처럼 표기가 다를 때)
  --drop-tray=A,B     특정 트레이 제외
  --drop-empty-target ΔOCV 가 20개 미만인 트레이 자동 제외 (=N 으로 기준 변경)
"""
import sys, re, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import runlog
from predict_xlsx import load, I_PAT, SLOPE_PAT, COND_PAT, TARGET_PAT, TRAY_PAT, measured_upto, find_targets, parse_target, drop_trays
from correct import target_report, within_tray_rho, topk_recall, safe_z, r2_linear

GRADE_KEY = "판정등급"
LAYER_PAT = re.compile(r"^(layer|dummy[_\s]*l\d+)$", re.I)


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
    tgts  = find_targets(df)
    tray  = next((c for c in df.columns if TRAY_PAT.search(c)), None)
    gcol  = next((c for c in df.columns if GRADE_KEY in str(c)), None)
    if not icols or not tgts:
        print("  !! 전류 컬럼 또는 타깃(delta OCV)을 못 찾았습니다."); return

    D = df.copy()
    D["_tray"] = D[tray].astype(str) if tray else "ALL"
    D = drop_trays(D)
    for c in icols + scols + conds: D[c] = pd.to_numeric(D[c], errors="coerce")
    D["_y"] = pd.to_numeric(D[tgts[-1]], errors="coerce")
    target_report(D["_y"].values, D[tray].astype(str).values if tray else "ALL", str(tgts[-1]))
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
    n_i, n_l, n_m = int(np.nansum(f_i)), int(np.nansum(f_l)), int(np.nansum(f_m))
    e_i, e_l, e_m = [int(np.nansum(f & (ng == 1))) for f in (f_i, f_l, f_m)]
    print("""
    → 자기 자신을 통계량에 넣으면, 크게 튀는 셀일수록 자기 트레이의 sigma 를 키운다.
      개선A(LOO)는 그 영향만 제거한다. 임계값의 눈금은 그대로다.
      개선B(median/MAD)는 자기가림도 없애지만 산포 추정 자체가 달라져
      적출 기준이 함께 움직인다. 둘을 같은 개선으로 묶어 말하면 안 된다.""")
    if ng.sum():
        print(f"""
    이 데이터에서의 판정
      자기가림의 크기      : 현행 {n_i}셀 → LOO {n_l}셀 ({n_l - n_i:+d}셀)
                            E 검출 {e_i}/{int(ng.sum())} → {e_l}/{int(ng.sum())}""")
        if e_m > e_i:
            print(f"      median/MAD           : 적출 {n_i}→{n_m}셀, E 검출 {e_i}→{e_m}"
                  f"  ★ 검출이 늘었다. 채택 검토할 것.")
        else:
            print(f"      median/MAD           : 적출 {n_i}→{n_m}셀"
                  f" ({(n_m / max(n_i, 1) - 1) * 100:+.0f}%), E 검출 {e_i}→{e_m} 로 변화 없음")
            print(f"""      ★ 이 데이터에서는 개선이 아니다. 과검만 늘고 검출 이득이 없다.
        현행 규칙이 이미 E 를 {e_i}/{int(ng.sum())} 잡고 있어 올릴 여지가 없다.
        자기가림 보정이 필요하다면 눈금을 바꾸지 않는 LOO 만 채택하는 것이 맞다.
      ※ 등급 E 가 이 ΔOCV 로 매겨졌다면 이 표는 부분적으로 동어반복이다.
        E 판정에 다른 기준이 함께 쓰였는지 확인할 것.""")

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
        """같은 트레이 안에서만 쌍을 뽑는다. 트레이 기준선은 차분으로 사라진다.

        n_pair 번 반복하지 않고 트레이별로 묶어 한 번에 뽑는다.
        셀이 5천 개를 넘으면 반복문으로는 느려서 쓸 수 없다.
        """
        sel = np.zeros(n, bool); sel[rows] = True
        pools = [v[sel[v]] for v in idx_by_tray.values()]
        pools = [p for p in pools if len(p) >= 2]
        if not pools: return np.array([], int), np.array([], int)
        w = np.array([len(p) for p in pools], float); w /= w.sum()
        pick = rng.choice(len(pools), n_pair, p=w)
        A = np.empty(n_pair, int); B = np.empty(n_pair, int)
        for pi in np.unique(pick):
            m = pick == pi; k = int(m.sum()); p = pools[pi]
            a = rng.integers(0, len(p), k)
            b = (a + 1 + rng.integers(0, len(p) - 1, k)) % len(p)   # a != b 보장
            A[m], B[m] = p[a], p[b]
        return A, B

    cv = GroupKFold(n_splits=min(5, max(D["_tray"].nunique(), 2)))
    # 폴드와 쌍을 한 번만 만들어 둔다. 절제 변형들이 같은 쌍을 써야
    # 피처 구성의 차이만 비교된다.
    folds = []
    for tr, te in cv.split(Xv, yc, groups=g):
        A, B = make_pairs(tr)
        keep = np.abs(y[A] - y[B]) > 0 if len(A) else np.array([], bool)
        folds.append((tr, te, A[keep] if len(A) else A, B[keep] if len(A) else B))

    p_rank = np.zeros(n); p_gbm = np.zeros(n)
    W = np.zeros(len(feats))
    for tr, te, A, B in folds:
        if len(A) < 50: continue
        lr = LogisticRegression(max_iter=2000, C=0.1).fit(
            Xv[A] - Xv[B], (y[A] > y[B]).astype(int))
        p_rank[te] = Xv[te] @ lr.coef_[0]; W += lr.coef_[0]
        p_gbm[te] = HistGradientBoostingRegressor(random_state=0).fit(
            Xv[tr], yc[tr]).predict(Xv[te])

    # ── 절제 실험 — 순위상관이 어디에서 오는가 ──────────────────
    #   조건 피처(특히 층)가 점수를 만들고 있는지 자동으로 가른다.
    #   --no-cond 를 따로 돌리지 않아도 여기서 보인다.
    def fit_rank(cols):
        """같은 폴드·같은 쌍으로 피처 구성만 바꿔 다시 학습한다."""
        if not cols: return np.zeros(n)
        Z, _, _, _ = safe_z(X[cols].values)
        out = np.zeros(n)
        for tr, te, A, B in folds:
            if len(A) < 50: continue
            lr = LogisticRegression(max_iter=2000, C=0.1).fit(
                Z[A] - Z[B], (y[A] > y[B]).astype(int))
            out[te] = Z[te] @ lr.coef_[0]
        return out

    lay = [c for c in feats if LAYER_PAT.match(c)]
    abl = [("전체 피처", feats)]
    if lay: abl.append((f"층 계열 제외 ({len(lay)}개)", [c for c in feats if c not in lay]))
    if conds: abl.append(("조건 전체 제외 (전류만)", [c for c in feats if c not in conds]))

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

    if len(abl) > 1:
        print(f"\n    ★ 절제 실험 — 그 순위상관은 어디에서 오는가 (쌍비교 선형 모델)")
        print(f"    {'쓴 피처':<26}{'트레이내 rho':>13}"
              + (f"{'상위1% E':>10}{'상위5% E':>10}" if ng.sum() else ""))
        print("    " + "-" * (39 + (20 if ng.sum() else 0)))
        rho_full, rho_cur = None, None
        for nm, cols in abl:
            sc = fit_rank(cols)
            rw, _ = within_tray_rho(sc, y, g)
            if rho_full is None: rho_full = rw
            if nm.startswith("조건 전체 제외"): rho_cur = rw
            line = f"    {nm:<26}{rw:>13.3f}"
            if ng.sum():
                t = topk_recall(sc, ng); line += f"{t[0.01] * 100:>9.0f}%{t[0.05] * 100:>9.0f}%"
            print(line)
            if lay and nm.startswith("층 계열 제외") and np.isfinite(rho_full) and abs(rho_full) > 0.2:
                drop = 1 - abs(rw) / abs(rho_full) if abs(rho_full) > 0 else 0
                if drop > 0.5:
                    print(f"""
      ★ 층을 빼면 상관이 {abs(rho_full):.3f} → {abs(rw):.3f} 로 {drop * 100:.0f}% 무너진다.
        이 모델은 자가방전이 아니라 '트레이 안에서 몇 층인가' 를 학습한 것이다.
        상관 숫자 자체는 진짜지만, SDM 전류의 성과로 말하면 안 된다.""")
        if rho_cur is not None and rho_full is not None and abs(rho_full) > 1e-9:
            keep = abs(rho_cur) / abs(rho_full)
            print(f"""
      ★ SDM 전류만으로 남는 상관은 {abs(rho_cur):.3f} 이다.
        전체 {abs(rho_full):.3f} 의 {keep * 100:.0f}%.  나머지 {(1 - keep) * 100:.0f}% 는
        측정이 아니라 조건 변수(층·온도·전압)가 만든 것이다.
        발표에서 쓸 숫자는 {abs(rho_cur):.3f} 쪽이다.""")

    print(f"\n    쌍비교 모델 가중치 상위 8개  (부호가 곧 '크면 나쁘다/좋다')")
    for nm_, w_ in sorted(zip(feats, W / max(cv.get_n_splits(), 1)), key=lambda x: -abs(x[1]))[:8]:
        wmax = float(np.abs(W).max()) / max(cv.get_n_splits(), 1)
        bar = "█" * (int(min(abs(w_) / wmax * 22, 22)) if wmax > 0 else 0)
        print(f"      {nm_:<14}{w_:>+8.3f}  {bar}")
    print("""
    → '트레이내 R2' 는 트레이 기준선을 제거한 뒤의 결정계수다.
      원래 회귀의 R2 가 음수였던 것은 트레이 기준선을 맞히지 못해서였다.
      목적(트레이 상대평가)에 맞춰 문제를 다시 세우면 같은 데이터에서 양수가 나온다.""")

    # ── [3] 층 효과 ───────────────────────────────────────────────
    lcol = next((c for c in D.columns if str(c).strip().lower() == "layer"), None)
    tl_score = None
    if lcol is not None and pd.to_numeric(D[lcol], errors="coerce").nunique() > 1:
        print("\n" + "-" * 78)
        print(" [3] ★ 층 효과 — 같은 트레이 안에서도 층마다 기준선이 다른가")
        print("-" * 78)
        L = pd.to_numeric(D[lcol], errors="coerce")
        tot = float(np.nanvar(yc))
        bl = pd.Series(yc).groupby(L.values).transform("mean").values
        eta = float(np.nanvar(bl) / tot) if tot > 0 else float("nan")
        print(f"    트레이 기준선을 뺀 ΔOCV 의 분산 중 층이 설명하는 비중  eta^2 = {eta * 100:.1f}%")
        prof = pd.Series(yc).groupby(L.values).agg(["median", "count"])
        print(f"\n    층별 ΔOCV 중앙값 (트레이 중앙값 기준 편차)")
        print(f"    {'층':>5}{'셀 수':>9}{'중앙 편차':>12}")
        print("    " + "-" * 26)
        for lv, r in prof.iterrows():
            print(f"    {lv:>5}{int(r['count']):>9,}{r['median']:>12.4f}")

        # 트레이x층 정규화가 검출을 개선하는가 — 이것이 실무 판정
        tl = pd.Series(y).groupby([pd.Series(g), L.values]).transform("median").values
        tl_score = y - tl
        tn_score = D[acol].values - pd.Series(D[acol].values).groupby(g).transform("median").values
        i_tl = D[acol].values - pd.Series(D[acol].values).groupby(
            [pd.Series(g), L.values]).transform("median").values
        if ng.sum():
            print(f"\n    판정 기준을 바꾸면 불량 검출이 어떻게 되는가 (SDM 전류 기준)")
            print(f"    {'정규화 단위':<24}{'상위1% E':>10}{'상위5% E':>10}")
            print("    " + "-" * 44)
            for nm_, sc_ in [("트레이 (현행)", tn_score), ("트레이 x 층", i_tl)]:
                t = topk_recall(sc_, ng)
                print(f"    {nm_:<24}{t[0.01] * 100:>9.0f}%{t[0.05] * 100:>9.0f}%")
        if ng.sum():
            t0, t1 = topk_recall(tn_score, ng), topk_recall(i_tl, ng)
            # 불량이 몇 개 없으므로 한 개 차이로 결론을 뒤집으면 안 된다.
            # 두 개(또는 불량의 25%) 이상 늘고, 다른 쪽이 나빠지지 않을 때만 '낫다' 로 본다.
            unit = 1.0 / max(int(ng.sum()), 1)
            marg = max(2 * unit, 0.25)
            print(f"\n    ※ 불량이 {int(ng.sum())}개뿐이다. 한 개 차이가 {unit * 100:.0f}%p 이므로,"
                  f"\n      {marg * 100:.0f}%p 미만의 차이는 판단 근거로 쓰지 않는다.")
            better = ((t1[0.01] - t0[0.01] >= marg and t1[0.05] >= t0[0.05]) or
                      (t1[0.05] - t0[0.05] >= marg and t1[0.01] >= t0[0.01]))
            if better:
                print("""
    ★ '트레이 x 층' 이 낫다. 판정 단위를 트레이에서 트레이 x 층 으로 바꾸는 것이
      개선이다. 현행은 층을 무시하므로 기준선이 높은 층의 셀이 계통적으로
      과검되고 낮은 층의 셀이 미검된다.""")
            else:
                print(f"""
    ★ 층을 반영해도 검출이 뚜렷하게 나아지지는 않는다
      (상위1% {t0[0.01] * 100:.0f}% → {t1[0.01] * 100:.0f}%, 상위5% {t0[0.05] * 100:.0f}% → {t1[0.05] * 100:.0f}%).
      층은 ΔOCV 를 흔들지만 불량과는 무관하다는 뜻이다. eta^2 = {eta * 100:.1f}% 는
      실재하는 효과이되 선별에는 쓸모가 없다.
      → 판정 단위는 트레이 그대로 두고, 층은 모델 피처에서 빼야 한다.
        순위는 맞히고 불량은 못 잡는 피처이므로 넣으면 손해다.""")
        else:
            print("""
    → 등급 라벨이 없어 검출 비교를 못 했다. eta^2 만으로는 판정 단위를
      바꿀지 결정할 수 없다.""")

    # ── [4] 계층 판정 ─────────────────────────────────────────────
    print("\n" + "-" * 78)
    print(" [4] 트레이 간 판정 — 셀 순위와 섞지 말 것")
    print("-" * 78)
    # 기준 점수는 순위모델이 아니라 '검출이 가장 좋았던 점수' 를 쓴다.
    # 순위모델은 조건 피처(층)에 끌려갈 수 있어 적출 판정의 기준으로 부적절하다.
    s_best = D[acol].values - pd.Series(D[acol].values).groupby(g).transform("median").values
    print("    기준 점수: 트레이 정규화 전류"
          "  (correct.py 의 P2 를 쓰면 트레이 간 비교가 더 정확해진다)")
    z_in = z_mad(s_best, g)
    tmed = pd.Series(s_best).groupby(g).median()
    mad_o = float(np.median(np.abs(tmed - tmed.median())) * 1.4826)
    z_of = (tmed - tmed.median()) / mad_o if mad_o > 0 else tmed * 0.0
    z_out = D["_tray"].map(z_of).values
    print(f"    트레이 내 z > {ksig:.0f}   : {int(np.nansum(z_in > ksig)):,}셀"
          f"  ({np.nanmean(z_in > ksig) * 100:.3f}%)   → 셀 단위 적출 대상")
    print(f"    트레이 간 z > {ksig:.0f}   : {int((z_of > ksig).sum())}트레이"
          f"  {list(z_of[z_of > ksig].index)[:5]}   → 트레이 단위 재확인 대상")
    if ng.sum():
        comb = np.fmax(np.nan_to_num(z_in, nan=-9), np.nan_to_num(z_out, nan=-9))
        a = topk_recall(np.nan_to_num(z_in, nan=-9), ng); b = topk_recall(comb, ng)
        print(f"\n    {'점수':<24}{'상위1% E':>10}{'상위5% E':>10}")
        print("    " + "-" * 44)
        print(f"    {'트레이 내 z 만':<24}{a[0.01] * 100:>9.0f}%{a[0.05] * 100:>9.0f}%")
        print(f"    {'max(내, 간) 로 합침':<24}{b[0.01] * 100:>9.0f}%{b[0.05] * 100:>9.0f}%")
        if b[0.01] < a[0.01] or b[0.05] < a[0.05]:
            print(f"""
      ★ 합치면 나빠진다. 당연하다 — 트레이 간 z 는 그 트레이의 모든 셀에 같은 값이라
        셀을 구별하지 못한다. max 로 합치면 z 가 높은 트레이의 평범한 셀들이
        상위로 올라와 진짜 불량을 밀어낸다.
        두 z 는 쓰임이 다르므로 합치지 말고 따로 쓸 것.
          트레이 내 z  →  셀 적출 (기존 흐름)
          트레이 간 z  →  그 트레이 전체 재확인·재측정 (별도 조치)""")
    print("""
    → 현행은 트레이 내 z 만 본다. '그 트레이가 통째로 나쁜 경우' 는 원리적으로
      못 잡는다 (다 같이 나쁘면 아무도 튀지 않는다). 트레이 간 z 가 그 사각지대를
      닫지만, 셀 순위와 같은 축에 놓아서는 안 된다.
      단, 트레이 간 비교는 온도 보정이 먼저다. correct.py 의 P2 점수를 쓸 것.""")

    print("\n" + "=" * 78)
    print(""" 읽는 순서
   1. [2] 의 절제 실험을 먼저 볼 것. 순위상관이 전류에서 온 것인지,
      층 같은 조건 변수에서 온 것인지가 거기서 갈린다.
      조건 변수에서 온 것이면 '예측했다' 고 말하면 안 된다.
   2. [1] 과 [3] 은 판정 규칙을 바꿀지 말지의 근거다.
      검출이 늘지 않고 적출만 늘면 개선이 아니다.
   3. [4] 의 두 z 는 쓰임이 다르다. 합치지 말 것.""")
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
        parse_target(sys.argv)
        with runlog.saving("rank_tray", a[0], sys.argv, sv, en):
            main(a[0], at, k, "--no-cond" in sys.argv)
