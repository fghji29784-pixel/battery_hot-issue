# -*- coding: utf-8 -*-
"""판정 규칙을 바꾸면 과검이 줄어드는가 — 전역 컷 vs 트레이 상대평가

  python judge.py "데이터.xlsx"
  python judge.py "데이터.xlsx" --at=5          # 5분 시점으로
  python judge.py "데이터.xlsx" --score=slope   # 점수를 기울기로

결과는 analysis/results/ 에 자동 저장된다 (--save=경로 / --no-save).

  --target="컬럼명"   3일 ΔOCV 컬럼을 직접 지정 (DOCV 처럼 표기가 다를 때)
  --drop-tray=A,B     특정 트레이 제외
  --drop-empty-target ΔOCV 가 20개 미만인 트레이 자동 제외 (=N 으로 기준 변경)

────────────────────────────────────────────────────────────────────────
 무엇을 묻는가
────────────────────────────────────────────────────────────────────────
 지금까지의 '최악셀 검사율' 은 전 셀을 한 줄로 세우고 위에서 자르는 방식을
 전제했다. 그런데 현행 공정은 트레이별 상대평가(μ+3σ)를 쓴다.
 상대평가로 자르면 과검이 줄어드는가?

 상반된 두 효과가 있어서 계산해 봐야 한다.

   줄어드는 쪽
     트레이마다 산포가 다르면, 전역 컷은 산포가 큰 트레이에서 양품을
     무더기로 집어온다. 트레이별로 자르면 그 쏠림이 사라진다.

   늘어나는 쪽 — 자기 마스킹
     μ 와 σ 를 그 트레이 안에서 구하는데, 불량이 그 안에 섞여 있으면
     불량 자신이 σ 를 키운다. 문턱이 올라가고 불량이 빠져나간다.
     빠져나간 불량을 잡으려 k 를 낮추면 다른 트레이에서 양품이 쏟아진다.
     → 트레이당 셀이 적을수록, 불량이 클수록 심해진다.

   또 하나 — 트레이당 불량 개수가 고르지 않다
     상대평가는 '어느 트레이든 비슷한 비율로 나쁜 놈이 있다' 를 전제한다.
     실데이터에서는 불량 4개가 서로 다른 트레이에 하나씩 있었다.
     불량이 없는 트레이에서도 문턱을 넘는 셀이 나오면 그건 전부 과검이다.

────────────────────────────────────────────────────────────────────────
 비교 방법 — 검출을 맞춰 놓고 과검을 센다
────────────────────────────────────────────────────────────────────────
 규칙마다 파라미터(상위 N, k 등)를 느슨한 쪽에서 조여 가며
 '불량을 전부 잡는 가장 빡빡한 지점' 을 찾고, 그때 적출된 셀 수를 센다.
 검출률이 같으므로 과검만 비교하면 된다. 이것이 공정에서 의미 있는 비교다.

 ※ 점수 자체는 이미 트레이 상대값이다 (같은 트레이 같은 행의 중앙값을 뺀 것).
   여기서 비교하는 것은 '문턱을 어디서 정하느냐' 다. 둘을 혼동하지 말 것.
"""
import sys, re, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import runlog
from predict_xlsx import (load, I_PAT, TARGET_PAT, TRAY_PAT, CELL_PAT,
                          measured_upto, find_targets, parse_target, drop_trays)
from correct import derive_conds, target_report
from rescue import grid_pos

GRADE_KEY = "판정등급"


def group_z(s, grp, how):
    """그룹 안에서의 z 값. how='sigma' 는 평균·표준편차, 'mad' 는 중앙값·MAD.

    '문턱 k 로 잘랐을 때 이 셀이 걸리는가' 는 곧 'k <= 이 셀의 z 인가' 다.
    그러므로 불량을 전부 잡는 가장 빡빡한 k 는 불량들의 z 중 최소값이고,
    파라미터를 훑을 필요 없이 바로 구해진다.
    """
    s = np.asarray(s, float); z = np.full(len(s), -np.inf)
    for _, idx in pd.Series(np.arange(len(s))).groupby(pd.Series(grp).values):
        i = idx.values
        v = s[i]; fv = v[np.isfinite(v)]
        if len(fv) < 5: continue
        if how == "sigma":
            c, w = float(np.mean(fv)), float(np.std(fv, ddof=1))
        else:
            c = float(np.median(fv)); w = 1.4826 * float(np.median(np.abs(fv - c)))
        if not np.isfinite(w) or w <= 0: continue
        z[i] = (v - c) / w
    return z


def worst_rank(v, y):
    """전역 순위 컷. 불량을 전부 잡는 최소 적출 수 = 가장 뒤처진 불량의 순위."""
    v = np.asarray(v, float)
    return int(max(int(np.sum(v > v[i])) + 1 for i in np.where(y == 1)[0]))


def main(spec, at=None, which="tnrow", rows=None, cols=None, order="col"):
    df, _ = load(spec)
    icols = sorted([c for c in df.columns if I_PAT.match(c)], key=lambda c: int(I_PAT.match(c).group(1)))
    mins = [int(I_PAT.match(c).group(1)) for c in icols]
    tgts = find_targets(df)
    tray = next((c for c in df.columns if TRAY_PAT.search(c)), None)
    cell = next((c for c in df.columns if CELL_PAT.match(str(c).strip())), None)
    gcol = next((c for c in df.columns if GRADE_KEY in str(c)), None)
    if not icols or not gcol:
        print("  !! 전류 컬럼(i_XXmin)과 판정등급이 모두 있어야 합니다."); return

    D = df.copy()
    D["_tray"] = D[tray].astype(str) if tray else "ALL"
    D = drop_trays(D)
    D["_num"] = pd.to_numeric(D[cell], errors="coerce") if cell else np.nan
    for c in icols: D[c] = pd.to_numeric(D[c], errors="coerce")
    D = D.dropna(subset=icols + ["_num"]).reset_index(drop=True)
    D["_upto"] = measured_upto(D[icols].values, mins)
    D = D[D["_upto"] >= mins[-1]].reset_index(drop=True)
    derive_conds(D, "t"); derive_conds(D, "v")
    n = len(D)
    if n < 30:
        print(f"\n  !! 셀이 {n}개뿐입니다."); return
    y = D[gcol].astype(str).str.strip().str.upper().eq("E").values.astype(int)
    if y.sum() == 0:
        print("\n  !! 불량(E) 이 없어 비교할 수 없습니다."); return
    g = D["_tray"].values
    if tgts:
        target_report(pd.to_numeric(D[tgts[-1]], errors="coerce").values, g, str(tgts[-1]))

    # 격자 자리 복원 — rescue.py 와 같은 규칙(셀 수가 아니라 셀 번호 최대값)
    mx = int(np.nanmax(D.groupby("_tray")["_num"].max().values))
    rw, cl = rows, cols
    if rw is None or cl is None:
        cand = [(abs(r - mx / r), r, mx // r) for r in range(1, mx + 1) if mx % r == 0]
        _, rw, cl = min(cand) if cand else (0, 1, mx)
    D["_r"], D["_c"] = grid_pos(D["_num"].values, rw, cl, order)
    if rw < 3 or cl < 3:
        print(f"\n  !! 격자가 {rw} x {cl} 로 얇습니다. --grid=12x12 처럼 지정하십시오.")

    at = at or (5 if 5 in mins else mins[-1])
    ai = int(np.argmin(np.abs(np.array(mins) - at)))
    X = D[icols].values.astype(float)

    # ── 점수 ─────────────────────────────────────────────────────
    # 이미 '같은 트레이 같은 행의 중앙값을 뺀' 상대값이다.
    if which == "slope":
        tw = np.asarray(mins[:ai + 1], float); Lv = tw - tw.mean()
        raw = (X[:, :ai + 1] - X[:, :ai + 1].mean(1, keepdims=True)) @ Lv / np.sum(Lv ** 2)
        nm = f"같은 행 12셀 대비 0~{mins[ai]}분 기울기"
    else:
        raw = X[:, ai]
        nm = f"같은 행 12셀 대비 {mins[ai]}분 전류"
    key = [pd.Series(g), pd.Series(D["_r"].values)]
    prof = pd.Series(raw).groupby(key).transform("median").values
    cnt = pd.Series(raw).groupby(key).transform("count").values
    s = np.where(cnt >= 5, raw - np.nan_to_num(prof, nan=0.0), raw)
    s = np.nan_to_num(s, nan=-1e18, posinf=-1e18, neginf=-1e18)

    print("=" * 78)
    print(" 판정 규칙을 바꾸면 과검이 줄어드는가 — 전역 컷 vs 트레이 상대평가")
    print("=" * 78)
    print(f"  {n:,}셀 / 트레이 {D['_tray'].nunique()}개 / 격자 {rw} x {cl} / 불량(E) {int(y.sum())}개")
    print(f"  점수: {nm}")
    print(f"  ※ 점수는 이미 트레이 상대값이다. 여기서 비교하는 것은 '문턱을 어디서 정하느냐' 다.")

    # ── [1] 불량이 트레이에 어떻게 흩어져 있는가 ─────────────────
    print("\n" + "-" * 78)
    print(" [1] 상대평가의 전제 — 불량이 트레이마다 고르게 있는가")
    print("-" * 78)
    per_tray = pd.Series(y).groupby(pd.Series(g)).sum()
    nt = len(per_tray)
    withz = int((per_tray > 0).sum())
    print(f"    불량이 있는 트레이 {withz} / {nt}개   트레이당 불량 "
          f"{per_tray[per_tray > 0].min()}~{per_tray.max()}개")
    if withz < nt:
        print(f"    → 불량이 없는 트레이가 {nt - withz}개다. 그 트레이에서 문턱을 넘는 셀은")
        print(f"      전부 과검이다. 트레이별 상대평가는 '어느 트레이에나 나쁜 놈이 있다' 를")
        print(f"      전제하는데, 이 데이터는 그렇지 않다.")
    else:
        print(f"    → 모든 트레이에 불량이 있다. 상대평가의 전제가 맞는 쪽이다.")

    # ── [2] 자기 마스킹 ───────────────────────────────────────────
    print("\n" + "-" * 78)
    print(" [2] 자기 마스킹 — 불량이 자기 트레이의 문턱을 밀어올리는가")
    print("-" * 78)
    print(f"    {'불량 셀':>8}{'트레이':>16}{'제 자신 포함 z':>15}{'자신 제외 z':>14}{'차이':>10}")
    print("    " + "-" * 64)
    worst_in, worst_ex = [], []
    for i in np.where(y == 1)[0]:
        m = (g == g[i])
        v = s[m]
        z_in = (s[i] - v.mean()) / (v.std(ddof=1) or 1e-12)
        vo = s[m & (np.arange(n) != i)]
        z_ex = (s[i] - vo.mean()) / (vo.std(ddof=1) or 1e-12)
        worst_in.append(z_in); worst_ex.append(z_ex)
        print(f"    {int(D['_num'][i]):>8}{str(g[i]):>16}{z_in:>15.2f}{z_ex:>14.2f}{z_ex - z_in:>10.2f}")
    infl = float(np.mean(np.array(worst_ex) - np.array(worst_in)))
    print(f"\n    평균 z 상승폭 {infl:+.2f}  (자신을 빼고 재면 이만큼 더 튄다)")
    need_k = float(np.min(worst_in))
    print(f"    불량을 전부 잡으려면 k ≤ {need_k:.2f} 여야 한다 (가장 안 튀는 불량의 z).")
    if need_k < 3:
        print(f"    → 현행 μ+3σ 로는 이 중 일부가 빠져나간다. k 를 낮춰야 하고,")
        print(f"      낮추는 순간 불량 없는 트레이에서도 셀이 쏟아진다. 그게 과검이다.")

    # ── [3] 규칙별 비교 ───────────────────────────────────────────
    print("\n" + "-" * 78)
    print(" [3] 검출을 맞춰 놓고 과검을 센다 (모두 불량 전량 검출 기준)")
    print("-" * 78)
    zt_s = group_z(s, g, "sigma")
    zt_m = group_z(s, g, "mad")
    zr_m = group_z(s, D["_r"].values, "mad")
    rules = [
        ("전역 상위 N (지금 방식)", s, "rank"),
        ("트레이별 μ + kσ (현행 공정)", zt_s, "k"),
        ("트레이별 중앙값 + k·MAD", zt_m, "k"),
        ("행별 중앙값 + k·MAD", zr_m, "k"),
        ("트레이별 표준화 후 전역 컷", zt_m, "rank"),
    ]
    print(f"    {'판정 규칙':<26}{'파라미터':>11}{'적출':>8}{'과검':>8}{'수율손실':>10}")
    print("    " + "-" * 63)
    res = {}
    for label, v, kind in rules:
        if kind == "rank":
            cnt_f = worst_rank(v, y); ptxt = f"상위 {cnt_f}"
        else:
            kstar = float(np.min(np.asarray(v)[y == 1]))
            if not np.isfinite(kstar):
                print(f"    {label:<26}{'—':>11}   그룹이 너무 작아 판정 불가"); continue
            cnt_f = int(np.sum(np.asarray(v) >= kstar)); ptxt = f"k={kstar:.2f}"
        over = cnt_f - int(y.sum())
        res[label] = (ptxt, cnt_f, over)
        print(f"    {label:<26}{ptxt:>11}{cnt_f:>7}셀{over:>7}셀{over / n * 100:>9.2f}%")
    neg = [l for l, r in res.items() if r[0].startswith("k=-")]
    if neg:
        print(f"\n    ※ 문턱 k 가 음수인 규칙: {', '.join(neg)}")
        print(f"      평균 아래까지 내려야 불량이 걸린다는 뜻이다. 그러면 그 트레이의")
        print(f"      절반 이상이 같이 적출된다. 공정에 쓸 수 있는 설정이 아니다.")

    # 음수 k 는 위에서 '공정에 못 쓴다' 고 했다. 그걸 최선으로 추천하면 안 된다.
    usable = {l: r for l, r in res.items() if not r[0].startswith("k=-")}
    pool = usable or res
    if pool:
        best = min(pool, key=lambda k: pool[k][1])
        base = res.get("전역 상위 N (지금 방식)")
        tail = "" if usable else "   (쓸 수 있는 규칙이 없어 전체에서 고름)"
        print(f"\n    ★ 가장 적게 버리는 규칙: {best}  ({pool[best][1]}셀){tail}")
        if base and best != "전역 상위 N (지금 방식)":
            d = base[1] - pool[best][1]
            if d <= 0:
                print(f"      그래도 지금 방식({base[1]}셀)보다 적지 않다. 바꿀 이유가 없다.")
            else:
                print(f"      지금 방식보다 {d}셀 적다 (수율 {d / n * 100:.2f}%p 개선).")
                if d <= max(2, 0.1 * base[1]):
                    print(f"      ※ 차이가 작다. 불량이 {int(y.sum())}개뿐이라 셀 한 개 순위로")
                    print(f"        뒤집힌다. 규칙을 바꿀 근거로는 약하다.")
        elif base:
            print(f"      트레이 상대평가로 바꿔도 줄지 않는다. [1][2] 가 이유다.")

    # ── [4] 같은 예산에서 몇 개를 잡는가 ─────────────────────────
    print("\n" + "-" * 78)
    print(" [4] 반대로 — 적출 예산을 고정하면 몇 개를 잡는가")
    print("-" * 78)
    budget = res.get("전역 상위 N (지금 방식)", (None, max(10, int(y.sum()) * 4), None))[1]
    print(f"    적출 예산 {budget}셀 고정")
    print(f"    {'판정 규칙':<26}{'검출':>10}{'실제 적출':>12}")
    print("    " + "-" * 50)
    for label, v, kind in rules:
        v = np.asarray(v, float); fin = np.isfinite(v)
        if fin.sum() == 0: continue
        # 예산만큼 적출되도록 문턱을 맞춘다 — 전역이든 그룹 z 든 방법은 같다.
        thr = np.sort(v[fin])[::-1][min(budget, int(fin.sum())) - 1]
        f = fin & (v >= thr)
        print(f"    {label:<26}{int(f[y == 1].sum())}/{int(y.sum()):<9}{int(f.sum()):>11}셀")

    print("\n" + "=" * 78)
    print(" 읽는 법")
    print("   [1] 불량이 없는 트레이가 많으면 트레이 상대평가는 불리하다.")
    print("       그 트레이에서 나오는 적출은 전부 과검이기 때문이다.")
    print("   [2] 불량의 z 가 3보다 작으면 현행 μ+3σ 로는 못 잡는다.")
    print("       자기 자신이 σ 를 키워서 문턱을 밀어올리기 때문이다(자기 마스킹).")
    print("       중앙값+MAD 는 그 영향을 덜 받는다. [3] 에서 둘을 비교한다.")
    print("   [3] 검출을 맞춰 놓았으므로 적출 셀 수가 곧 비용이다. 적을수록 좋다.")
    print("   [4] 예산을 맞춰 놓고 보면 어느 규칙이 같은 비용으로 더 잡는지 나온다.")
    print("=" * 78)


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    if not a: print(__doc__)
    else:
        at, sc, rw, cl, od = None, "tnrow", None, None, "col"
        for x in sys.argv:
            if x.startswith("--at="): at = int(x.split("=")[1])
            if x.startswith("--score="): sc = x.split("=")[1].strip().lower()
            if x.startswith("--order="): od = x.split("=")[1].strip().lower()[:3]
            if x.startswith("--grid="):
                m = re.match(r"(\d+)\s*[xX*]\s*(\d+)", x.split("=")[1])
                if m: rw, cl = int(m.group(1)), int(m.group(2))
        sv, en = runlog.parse(sys.argv)
        parse_target(sys.argv)
        with runlog.saving("judge", a[0], sys.argv, sv, en):
            main(a[0], at, sc, rw, cl, od)
