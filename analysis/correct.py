# -*- coding: utf-8 -*-
"""
SDM 전류 물리 보정 — 음수 전류와 트레이 오프셋의 정체를 온도 드리프트로 설명하고,
트레이 중앙값 빼기를 '물리 보정'으로 대체한다.

  python correct.py "데이터.xlsx"
  python correct.py "데이터.xlsx" --at=15            # 평가 기준 시점 [분]
  python correct.py "데이터.xlsx" --cov=dTdt,t_init  # 보정에 쓸 조건 변수
  python correct.py "데이터.xlsx" --sweep            # 시점별 비교표까지

────────────────────────────────────────────────────────────────────────
 가설
────────────────────────────────────────────────────────────────────────
 전위고정 SDM에서 계측기가 읽는 전류는 자가방전만이 아니다.

     I_meas(t) = I_sd(t)  +  C_eff x (dU/dT) x (dT/dt)  +  잡음
                 ~~~~~~~     ~~~~~~~~~~~~~~~~~~~~~~~~~
                 우리가       셀 온도가 변하면 OCV가 따라 움직이고,
                 원하는 것    전압을 붙들기 위해 계측기가 전류를 흘린다.
                              부호는 dT/dt 부호를 따른다 → 음수가 나온다.

 C_eff 는 32Ah 셀에서 10^5 F 급이다. dU/dT 가 0.1 mV/K 수준이어도
 0.01 K/min 의 드리프트가 수 uA 의 가짜 전류를 만든다.
 자가방전 신호 자체가 수 uA 이므로, 이 항은 무시할 수 있는 크기가 아니다.

 이 가설이 맞으면 세 가지가 동시에 설명된다.
   ① 자가방전 전류가 음수로 나오는 셀이 있다
   ② 트레이마다 전류 분포가 통째로 다르다 (트레이마다 투입 온도·대기시간이 다르다)
   ③ 온도와 전류의 상관이 아레니우스와 반대 부호로 나온다
      (아레니우스는 정적 효과(+), 열드리프트는 동적 효과(부호 자유).
       지금 데이터에서는 후자가 지배적이라 부호가 뒤집힌다)

────────────────────────────────────────────────────────────────────────
 방법 — 계수는 '트레이 중앙값끼리' 회귀해서 얻는다
────────────────────────────────────────────────────────────────────────
 셀 단위로 회귀하면 셀 고유의 자가방전 신호까지 같이 지워질 위험이 있다.
 그래서 트레이 중앙값 대 트레이 중앙값으로 회귀해 계수 beta 를 얻는다.
 셀 고유 편차는 회귀에 참여하지 않으므로 지워지지 않는다.

   P1 (셀 단위 보정)   I - (x_cell  - x_bar) @ beta   트레이 내 순위도 바뀐다
   P2 (트레이 오프셋만) I - (x_tray  - x_bar) @ beta   트레이 내 순위는 그대로

 ★ P2 가 이 스크립트의 제안이다.
   트레이 중앙값 빼기(TN)와 달리, '온도로 설명되는 만큼만' 뺀다.
   그래서 트레이 전체가 불량인 경우에도 그 트레이가 높게 남는다.
   TN 은 그 경우를 원리적으로 못 잡는다 (다 같이 빼버리므로).

원본/셀단위 값은 출력하지 않는다.
"""
import sys, re, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from predict_xlsx import load, I_PAT, SLOPE_PAT, COND_PAT, TARGET_PAT, TRAY_PAT, measured_upto

TMIN_PAT = re.compile(r"^t[_\s]*(\d+)\s*min$", re.I)
GRADE_KEY = "판정등급"
KB, EA_DEFAULT = 8.617333262e-5, 0.94      # eV/K, eV  (Keysight 2023 실측)


# ──────────────────────────────────────────────────────────────────────
#  지표
# ──────────────────────────────────────────────────────────────────────
def between_tray_share(x, tray):
    """전체 분산 중 트레이 간 분산이 차지하는 비중."""
    s = pd.Series(x)
    bt = s.groupby(pd.Series(tray).values).transform("median")
    return float(np.var(bt) / max(np.var(s), 1e-300))


def within_tray_rho(score, y, tray, min_n=20):
    """트레이 안에서만 잰 순위상관. Fisher-z 로 트레이를 가로질러 합친다.

    현장 판정이 트레이 상대평가이므로, 이것이 목적에 맞는 상관이다.
    전체 풀링 상관은 트레이 간 오프셋에 오염된다.
    """
    zs, ws, used = [], [], 0
    for _, idx in pd.Series(np.arange(len(y))).groupby(pd.Series(tray).values):
        i = idx.values
        if len(i) < min_n: continue
        r = spearmanr(score[i], y[i]).statistic
        if not np.isfinite(r): continue
        r = np.clip(r, -0.999999, 0.999999)
        zs.append(np.arctanh(r)); ws.append(len(i) - 3); used += 1
    if not zs: return float("nan"), 0
    return float(np.tanh(np.average(zs, weights=ws))), used


def topk_recall(score, y, qs=(0.005, 0.01, 0.05)):
    n, ng = len(y), int(y.sum())
    if ng == 0: return {q: float("nan") for q in qs}
    hits = np.cumsum(y[np.argsort(-score)])
    return {q: hits[max(int(round(n * q)), 1) - 1] / ng for q in qs}


def tray_flag_rate(score, tray, k=3.0, robust=True):
    """트레이별 임계 초과 셀 비율. robust=True 면 median/MAD, False 면 mean/std."""
    s = pd.Series(score); g = s.groupby(pd.Series(tray).values)
    if robust:
        med = g.transform("median")
        mad = g.transform(lambda v: np.median(np.abs(v - np.median(v))) * 1.4826)
        z = (s - med) / mad.replace(0, np.nan)
    else:
        z = (s - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    return z.values


# ──────────────────────────────────────────────────────────────────────
#  본체
# ──────────────────────────────────────────────────────────────────────
def main(spec, at=None, cov_req=None, do_sweep=False, ea=EA_DEFAULT):
    df, nfile = load(spec)
    icols = sorted([c for c in df.columns if I_PAT.match(c)], key=lambda c: int(I_PAT.match(c).group(1)))
    mins  = [int(I_PAT.match(c).group(1)) for c in icols]
    tmins = sorted([c for c in df.columns if TMIN_PAT.match(c)], key=lambda c: int(TMIN_PAT.match(c).group(1)))
    conds = [c for c in df.columns if COND_PAT.match(c) and pd.api.types.is_numeric_dtype(df[c])]
    tgts  = [c for c in df.columns if TARGET_PAT.search(c)]
    tray  = next((c for c in df.columns if TRAY_PAT.search(c)), None)
    gcol  = next((c for c in df.columns if GRADE_KEY in str(c)), None)
    if not icols:
        print("  !! i_XXmin 형태의 전류 컬럼을 못 찾았습니다."); return

    D = df.copy()
    D["_tray"] = D[tray].astype(str) if tray else "ALL"
    for c in icols + conds + tmins: D[c] = pd.to_numeric(D[c], errors="coerce")
    D = D.dropna(subset=icols).reset_index(drop=True)
    D["_upto"] = measured_upto(D[icols].values, mins)
    D = D[D["_upto"] >= mins[-1]].reset_index(drop=True)
    n = len(D)

    at = at or (15 if 15 in mins else mins[-1])
    acol = f"i_{at}min" if f"i_{at}min" in D.columns else icols[-1]

    print("=" * 78)
    print(" SDM 전류 물리 보정 — 열드리프트 가설 검정")
    print("=" * 78)
    print(f"  {n:,}셀 / 트레이 {D['_tray'].nunique()}개 / 파일 {nfile}개")
    print(f"  전류 피처 {len(icols)}개 ({mins[0]}~{mins[-1]}분)   평가 시점 {at}분")

    # ── dT/dt 만들기 ──────────────────────────────────────────────
    if len(tmins) >= 3:
        tm = np.array([int(TMIN_PAT.match(c).group(1)) for c in tmins], float)
        TT = D[tmins].values
        tc = tm - tm.mean()
        D["_dTdt"] = (TT - TT.mean(1, keepdims=True)) @ tc / np.sum(tc ** 2)
        D["_Tmean"] = TT.mean(1)
        src = f"분단위 온도 {len(tmins)}점 회귀"
    elif "t_init" in D.columns and "t_final" in D.columns:
        span = float(mins[-1] - 0) if mins[-1] else 30.0
        D["_dTdt"] = (D["t_final"] - D["t_init"]) / span
        D["_Tmean"] = (D["t_final"] + D["t_init"]) / 2.0
        src = f"t_final - t_init 를 {span:.0f}분으로 나눔"
    else:
        print("\n  !! 온도 컬럼(t_init/t_final 또는 t_XXmin)이 없어 보정할 수 없습니다.")
        return
    print(f"  dT/dt 산출: {src}")

    # ── [1] 진단 ──────────────────────────────────────────────────
    print("\n" + "-" * 78)
    print(" [1] 진단 — 음수 전류와 트레이 오프셋")
    print("-" * 78)
    print(f"    {'시점':>6}{'음수 비율':>11}{'중앙값':>13}{'트레이간 분산':>14}")
    print("    " + "-" * 44)
    for c, m in zip(icols, mins):
        if m not in (mins[0], 10, at, mins[-1]): continue
        print(f"    {m:>4}분{np.mean(D[c] < 0) * 100:>10.1f}%{np.median(D[c]):>13.4g}"
              f"{between_tray_share(D[c].values, D['_tray'].values) * 100:>13.1f}%")

    print(f"\n    dT/dt   중앙값 {np.median(D['_dTdt']):+.4f} K/min"
          f"   5~95% {np.percentile(D['_dTdt'], 5):+.4f} ~ {np.percentile(D['_dTdt'], 95):+.4f}")
    print(f"    트레이 중앙 dT/dt 의 분산 비중 "
          f"{between_tray_share(D['_dTdt'].values, D['_tray'].values) * 100:.1f}%"
          "   ← 온도 이력은 거의 트레이 단위로 결정된다")
    r_dt = spearmanr(D["_dTdt"], D[acol], nan_policy="omit").statistic
    print(f"    rho(dT/dt , I_{at}min) = {r_dt:+.3f}"
          + ("   ← 열드리프트 가설과 부합" if abs(r_dt) > 0.2 else "   ← 상관이 약하다. 가설 기각 쪽"))

    if "delta_v" in D.columns:
        iv = D[acol].values; dv = D["delta_v"].values
        ok = np.abs(iv) > np.percentile(np.abs(iv), 50)
        rimp = np.median(dv[ok] / iv[ok])
        print(f"\n    delta_v 대 I : rho {spearmanr(dv, iv, nan_policy='omit').statistic:+.3f}"
              f"   함의 저항 median(delta_v/I) = {rimp:.4g} ohm")
        print("      → 이 값이 장비 출력저항과 같으면 delta_v 와 I 는 옴의 법칙으로 묶인 "
              "같은 측정이다.\n        독립 피처로 세면 안 된다.")

    # ── [2] 트레이 중앙값 회귀 ────────────────────────────────────
    print("\n" + "-" * 78)
    print(" [2] 트레이 오프셋을 온도로 얼마나 설명하는가  (트레이 중앙값끼리 회귀)")
    print("-" * 78)
    pool = ["_dTdt", "_Tmean"] + [c for c in ("v_init", "rwiring", "layer") if c in D.columns]
    cov = [c for c in (cov_req or pool) if c in D.columns]
    cov = [c for c in cov if D[c].notna().sum() > 0 and D[c].nunique() > 2]

    T = D.groupby("_tray")[cov + [acol]].median()
    ntray = len(T)
    if ntray < 6:
        print(f"    !! 트레이가 {ntray}개뿐이라 트레이 단위 회귀가 불안정합니다.")
    Xt = T[cov].values.astype(float); yt = T[acol].values.astype(float)
    mu, sd = np.nanmean(Xt, 0), np.nanstd(Xt, 0) + 1e-300
    Zt = np.nan_to_num((Xt - mu) / sd)

    # 물리적 기준점. dT/dt = 0 (열평형), T = 25도. 나머지는 물리적 영점이 없어
    # 전체 중앙값을 기준으로 둔다. 기준점 선택은 모든 셀을 같은 양만큼 평행이동
    # 시킬 뿐이므로 순위는 바뀌지 않는다. 음수 비율을 물리적으로 읽기 위한 것이다.
    ref = np.array([0.0 if c == "_dTdt" else 25.0 if c == "_Tmean"
                    else float(np.nanmedian(D[c])) for c in cov])

    from sklearn.linear_model import RidgeCV
    reg = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(Zt, yt)
    pred = reg.predict(Zt)
    r2 = 1 - np.sum((pred - yt) ** 2) / max(np.sum((yt - yt.mean()) ** 2), 1e-300)

    print(f"    대상: 트레이 {ntray}개의 중앙값,  설명변수 {len(cov)}개")
    print(f"    {'변수':<10}{'표준화 계수':>14}{'단독 R2':>10}")
    print("    " + "-" * 34)
    for j, c in enumerate(cov):
        s1 = 1 - np.sum((np.polyval(np.polyfit(Zt[:, j], yt, 1), Zt[:, j]) - yt) ** 2) \
                 / max(np.sum((yt - yt.mean()) ** 2), 1e-300)
        nm = {"_dTdt": "dT/dt", "_Tmean": "T_mean"}.get(c, c)
        print(f"    {nm:<10}{reg.coef_[j]:>+14.4g}{s1:>10.3f}")
    print(f"\n    ★ 트레이 오프셋의 {r2 * 100:.1f}% 를 온도·전압 조건으로 설명한다  (R2)")
    if r2 > 0.5:
        print("      → 트레이 정규화가 듣는 이유가 '트레이라는 라벨' 때문이 아니라")
        print("        측정 가능한 물리량 때문임을 뜻한다. 그러면 라벨 대신 물리량으로 뺄 수 있다.")

    # ── [3] 보정된 전류 만들기 ────────────────────────────────────
    def correct(col, level):
        """트레이 중앙값끼리 얻은 계수로 셀 전류에서 온도항을 뺀다.

        level='tray' 는 트레이 중앙 조건만 쓰므로 트레이 내 순위를 바꾸지 않는다.
        level='cell' 은 셀별 조건까지 쓰므로 트레이 내 순위도 바뀐다.
        기준점은 dT/dt=0, T=25도 이므로 결과는 '열평형 25도 조건의 전류' 로 읽는다.
        """
        b = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(
            Zt, D.groupby("_tray")[col].median().reindex(T.index).values)
        if level == "cell":
            Xc = D[cov].values.astype(float)
        else:
            Xc = D.groupby("_tray")[cov].transform("median").values.astype(float)
        Zc = np.nan_to_num((Xc - ref) / sd)
        return D[col].values - (Zc @ b.coef_)

    y_grade = D[gcol].astype(str).str.strip().str.upper().isin(["E"]).values.astype(int) \
        if gcol else np.zeros(n, int)
    ycol = tgts[-1] if tgts else None
    yv = pd.to_numeric(D[ycol], errors="coerce").values if ycol else None

    tn = lambda v: v - pd.Series(v).groupby(D["_tray"].values).transform("median").values
    ratio_col = f"i_{max(m for m in mins if m <= at // 2)}min" if any(m <= at // 2 for m in mins) else icols[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(np.abs(D[ratio_col]) > 1e-12, D[acol] / D[ratio_col], np.nan)
    ratio = np.nan_to_num(ratio, nan=np.nanmedian(ratio))

    tk = D["_Tmean"].values + 273.15
    p2 = correct(acol, "tray")
    # 아레니우스는 자가방전(양수)에만 뜻이 있다. 보정 후에도 음수인 셀은
    # 열드리프트가 다 안 빠진 셀이므로 배율을 적용하지 않고 그대로 둔다.
    arr = np.where(p2 > 0, p2 * np.exp(ea / KB * (1.0 / tk - 1.0 / 298.15)), p2)

    scores = [
        ("① 원시 전류",              D[acol].values),
        ("② 트레이 중앙값 빼기 TN",   tn(D[acol].values)),
        ("③ 물리보정 P1 (셀)",        correct(acol, "cell")),
        ("④ 물리보정 P2 (트레이)",    p2),
        ("⑤ P2 + 아레니우스 25도",    arr),
        ("⑥ P2 + TN",                tn(p2)),
        (f"⑦ 형상비 I({at})/I({ratio_col.split('_')[1]})", ratio),
        ("⑧ 형상비 + TN",            tn(ratio)),
    ]

    # ── [4] 비교표 ────────────────────────────────────────────────
    print("\n" + "-" * 78)
    print(f" [3] 점수별 비교  ({at}분 시점)")
    print("-" * 78)
    hdr = f"    {'점수':<26}{'음수%':>7}{'트레이간':>9}"
    if yv is not None: hdr += f"{'전체rho':>9}{'트레이내rho':>12}"
    if y_grade.sum(): hdr += f"{'상위1%':>8}{'상위5%':>8}"
    print(hdr); print("    " + "-" * (len(hdr) - 4))
    rows = []
    for nm, s in scores:
        line = f"    {nm:<26}{np.mean(s < 0) * 100:>6.1f}%" \
               f"{between_tray_share(s, D['_tray'].values) * 100:>8.1f}%"
        rec = {"name": nm}
        if yv is not None:
            rp = spearmanr(s, yv, nan_policy="omit").statistic
            rw, _ = within_tray_rho(s, yv, D["_tray"].values)
            line += f"{rp:>9.3f}{rw:>12.3f}"; rec.update(rho=rp, rho_w=rw)
        if y_grade.sum():
            tkr = topk_recall(s, y_grade)
            line += f"{tkr[0.01] * 100:>7.0f}%{tkr[0.05] * 100:>7.0f}%"
            rec.update(r1=tkr[0.01], r5=tkr[0.05])
        print(line); rows.append(rec)

    print("""
    읽는 법
      · '트레이간' 은 전체 분산 중 트레이 간 분산 비중. 낮을수록 셀 고유 신호가 남은 것.
      · '트레이내 rho' 가 목적 지표다. 현장 판정이 트레이 상대평가이기 때문.
      · ②와 ⑥/⑧ 은 트레이 내 순위를 바꾸지 않으므로 '트레이내 rho' 가 같은 계열끼리
        같게 나온다. 차이는 '전체 rho' 와 전역 상위 k% 검출에서 난다.
      · ④가 ②에 가까우면, 트레이 라벨 없이도 같은 일을 할 수 있다는 뜻이다.""")

    # ── [5] 전트레이 불량 사각지대 ────────────────────────────────
    print("\n" + "-" * 78)
    print(" [4] ★ TN 이 원리적으로 못 잡는 것 — 트레이 전체가 나쁜 경우")
    print("-" * 78)
    tm_raw = D.groupby("_tray")[acol].median()
    tm_p2  = pd.Series(p2).groupby(D["_tray"].values).median()
    print(f"    트레이 중앙값의 산포 (표준편차)")
    print(f"      원시      {tm_raw.std():.4g}")
    print(f"      물리보정  {tm_p2.std():.4g}"
          f"   ({(1 - tm_p2.std() / max(tm_raw.std(), 1e-300)) * 100:.0f}% 축소)")
    print(f"      TN        0        ← 정의상 전부 0. 트레이 간 비교가 불가능해진다.")
    z_tray = (tm_p2 - tm_p2.median()) / (np.median(np.abs(tm_p2 - tm_p2.median())) * 1.4826 + 1e-300)
    hi = z_tray.sort_values(ascending=False).head(3)
    print(f"\n    물리보정 후 트레이 단위 robust z 상위 3개")
    for t_, z_ in hi.items():
        print(f"      {str(t_):<16} z = {z_:+.2f}")
    print("""
    → 이 z 가 큰 트레이는 '온도로 설명되지 않는 만큼 전류가 높은 트레이' 다.
      TN 만 쓰면 이 신호가 통째로 사라진다. 계층 판정(트레이 내 z + 트레이 간 z)이
      필요한 이유다.""")

    # ── [6] 시점 스윕 ─────────────────────────────────────────────
    if do_sweep and yv is not None:
        print("\n" + "-" * 78)
        print(" [5] 시점별 — 몇 분에서 포화되는가 (보정 전후)")
        print("-" * 78)
        print(f"    {'구간':>6}{'원시 내부rho':>13}{'P2 내부rho':>12}{'원시 전체rho':>13}{'P2 전체rho':>12}")
        print("    " + "-" * 52)
        for c, m in zip(icols, mins):
            if m not in (5, 8, 10, 12, 15, 20, 25, 30) or m > mins[-1]: continue
            s0 = D[c].values; s2 = correct(c, "tray")
            w0, _ = within_tray_rho(s0, yv, D["_tray"].values)
            w2, _ = within_tray_rho(s2, yv, D["_tray"].values)
            print(f"    {m:>4}분{w0:>13.3f}{w2:>12.3f}"
                  f"{spearmanr(s0, yv, nan_policy='omit').statistic:>13.3f}"
                  f"{spearmanr(s2, yv, nan_policy='omit').statistic:>12.3f}")

    print("\n" + "=" * 78)
    print(""" 이 스크립트가 주장하는 것
   1. 음수 전류는 불량이 아니라 온도 드리프트의 부호다.
   2. 트레이 오프셋의 상당 부분은 '트레이'가 아니라 '온도 이력'이다.
      위 [2] 의 R2 가 그 근거다.
   3. 그러므로 트레이 중앙값 빼기는 물리 보정으로 대체할 수 있고,
      대체하면 트레이 전체가 불량인 경우도 볼 수 있게 된다 ([4]).
 반증 조건 — 아래면 이 가설은 틀린 것이다
   · [1] 의 rho(dT/dt, I) 가 0 근처
   · [2] 의 R2 가 0.3 미만
   · [3] 에서 ④가 ①보다 나아지지 않음""")
    print("=" * 78)


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    if not a: print(__doc__)
    else:
        at, cv, ea = None, None, EA_DEFAULT
        for x in sys.argv:
            if x.startswith("--at="):  at = int(x.split("=")[1])
            if x.startswith("--cov="): cv = [t.strip() for t in x.split("=", 1)[1].split(",")]
            if x.startswith("--ea="):  ea = float(x.split("=")[1])
        main(a[0], at, cv, "--sweep" in sys.argv, ea)
