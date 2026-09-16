# -*- coding: utf-8 -*-
"""곡선만으로 교란을 분리한다 — 보정 변수를 쓰지 않는 접근

  python decompose.py "데이터.xlsx"
  python decompose.py "데이터.xlsx" --k=2 --at=15

결과는 analysis/results/ 에 자동 저장된다 (--save=경로 / --no-save).

────────────────────────────────────────────────────────────────────────
 왜 다른 길로 가는가
────────────────────────────────────────────────────────────────────────
 전압·온도·자리로 보정하는 길은 이미 다른 사람이 갔다.
 그 길은 공통점이 있다 — '교란을 설명하는 변수를 사람이 지정하고, 그 변수를
 재서 뺀다'. 그래서 온도계가 틀리거나, 자리를 모르거나, 교란 원인이 목록에
 없으면 작동하지 않는다.

 여기서는 변수를 쓰지 않는다. 대신 '모양' 으로 가른다.

   자가방전과 열드리프트는 시간에 따라 서로 다른 모양으로 나타난다.
   15분 창 안에서 둘은 구별된다.

   자가방전   I_sd x (1 - exp(-t/tau_cell)),  tau_cell = 7.45 시간
              15분은 tau 의 3% 구간이라 사실상 '직선 상승' 이다.
   열드리프트  K x dT/dt(t).  온도는 지수적으로 안정되므로 이 항은 '감쇠' 한다.
              올라갔다가 스스로 사라진다.

 → 직선 성분은 자가방전, 곡률 성분은 열드리프트.
   값을 읽지 말고 기울기를 읽으면 열 항이 스스로 빠진다.
   온도계도, 자리도, 전압도 필요 없다. 곡선 하나면 된다.

 ★ 주의 — 순진한 '공통모드 제거' 는 실패한다
   같은 트레이 곡선들을 주성분분석해 첫 성분을 빼면 될 것 같지만 안 된다.
   모든 곡선의 모양이 비슷하면 첫 성분은 교란이 아니라 그냥 '진폭' 이고,
   진폭이 곧 자가방전이라 빼는 순간 신호가 지워진다.
   합성 데이터에서 PC1 의 시간 모양이 평균 곡선과 같았고(끝값과 상관 -0.999),
   잔차 점수가 아무 보정도 안 한 것보다 나빴다.
   [2] 가 이 함정을 자동으로 검사한다.

────────────────────────────────────────────────────────────────────────
 두 번째 축 — 전류가 아니라 전하량을 본다
────────────────────────────────────────────────────────────────────────
   I(t) = I_sd + K x dT/dt(t)
   Q(t) = I_sd x t + K x [T(t) - T(0)]          (양변 적분)

 적분하면 항의 성질이 갈린다.
   자가방전 항  : t 에 비례해 계속 자란다
   열드리프트 항 : 온도 '변화량' 에만 의존하므로 유계다 (온도는 결국 안정된다)

 그러므로 Q(t) 에 직선을 맞추면 기울기가 I_sd 추정값이고, 열 항은 절편으로
 흡수된다. 온도를 재지 않아도 된다. 적분이 고주파 잡음도 눌러준다.

 ★ 모든 셀이 같은 측정을 받는다는 제약과 맞는다.
   셀마다 측정을 다르게 하지 않는다. 같은 곡선을 다르게 읽을 뿐이다.

────────────────────────────────────────────────────────────────────────
 세 번째 축 — 불량을 찾지 말고 양품에서 벗어난 정도를 재기
────────────────────────────────────────────────────────────────────────
 불량이 8개뿐이라 지도학습은 한계가 분명하다.
 위 분해에서 '공통 모양' 을 뺀 잔차의 크기는 라벨 없이 계산된다.
 양품 5,807개가 만드는 정상 범위에서 얼마나 벗어났는지가 곧 이상 점수다.
 불량 개수에 의존하지 않고, 처음 보는 불량 유형에도 반응한다.

  --target="컬럼명"   3일 ΔOCV 컬럼을 직접 지정 (DOCV 처럼 표기가 다를 때)
"""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import runlog
from predict_xlsx import load, I_PAT, COND_PAT, TARGET_PAT, TRAY_PAT, measured_upto, find_targets, parse_target
from correct import (target_report, between_tray_share, within_tray_rho, topk_recall,
                     derive_conds)

GRADE_KEY = "판정등급"


def tray_svd(X, g, k):
    """트레이마다 곡선 행렬을 분해해 상위 k개 공통 모양을 뺀다.

    반환: (잔차 행렬, 첫 성분 점수, 트레이별 [PC1 설명비율, PC1~k 설명비율])
    """
    Rm = np.zeros_like(X); pc1 = np.zeros(len(X)); ev = {}
    for t in pd.unique(g):
        i = np.where(g == t)[0]
        if len(i) < max(k + 2, 8):
            Rm[i] = X[i] - X[i].mean(0); continue
        Xc = X[i] - X[i].mean(0)                      # 시점별 트레이 평균 제거
        Xc = np.nan_to_num(Xc, nan=0.0, posinf=0.0, neginf=0.0)
        try:
            U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        except np.linalg.LinAlgError:
            Rm[i] = Xc; continue
        tot = float(np.sum(S ** 2)) or 1.0
        ev[t] = (float(S[0] ** 2 / tot), float(np.sum(S[:k] ** 2) / tot))
        pc1[i] = U[:, 0] * S[0]
        Rm[i] = Xc - (U[:, :k] * S[:k]) @ Vt[:k]
    return Rm, pc1, ev


def charge_slope(X, mins, lo, hi):
    """전하량 Q(t) 에 직선을 맞춘 기울기. 열 항은 절편으로 흡수된다."""
    t = np.asarray(mins, float) * 60.0
    Q = np.concatenate([np.zeros((len(X), 1)),
                        np.cumsum((X[:, 1:] + X[:, :-1]) / 2 * np.diff(t), axis=1)], 1)
    m = (np.asarray(mins) >= lo) & (np.asarray(mins) <= hi)
    if m.sum() < 3: m = np.ones(len(mins), bool)
    tt = t[m] - t[m].mean()
    return (Q[:, m] - Q[:, m].mean(1, keepdims=True)) @ tt / np.sum(tt ** 2)


def main(spec, at=None, k=2):
    df, _ = load(spec)
    icols = sorted([c for c in df.columns if I_PAT.match(c)], key=lambda c: int(I_PAT.match(c).group(1)))
    mins  = [int(I_PAT.match(c).group(1)) for c in icols]
    conds = [c for c in df.columns if COND_PAT.match(c) and pd.api.types.is_numeric_dtype(df[c])]
    tgts  = find_targets(df)
    tray  = next((c for c in df.columns if TRAY_PAT.search(c)), None)
    gcol  = next((c for c in df.columns if GRADE_KEY in str(c)), None)
    if len(icols) < 6:
        print("  !! 시점별 전류 컬럼이 6개 미만이라 분해할 수 없습니다."); return

    D = df.copy()
    D["_tray"] = D[tray].astype(str) if tray else "ALL"
    for c in icols + conds: D[c] = pd.to_numeric(D[c], errors="coerce")
    D = D.dropna(subset=icols).reset_index(drop=True)
    D["_upto"] = measured_upto(D[icols].values, mins)
    D = D[D["_upto"] >= mins[-1]].reset_index(drop=True)
    made = derive_conds(D, "t") + derive_conds(D, "v")
    if made:
        print(f"  시계열에서 만든 조건 변수 {len(made)}개: {', '.join(made)}")
    # 분해는 행렬 연산이라 inf 하나로 SVD 가 발산한다. 유한값이 아닌 셀은 뺀다.
    fin = np.isfinite(D[icols].values).all(1)
    if not fin.all():
        print(f"  전류에 유한값이 아닌 값이 있는 셀 {int((~fin).sum())}개를 제외했습니다.")
        D = D[fin].reset_index(drop=True)
    n = len(D); g = D["_tray"].values
    if n < 30:
        print(f"\n  !! {mins[-1]}분까지 측정된 셀이 {n}개뿐입니다."); return
    at = at or (15 if 15 in mins else mins[-1])
    ai = int(np.argmin(np.abs(np.array(mins) - at)))
    X = D[icols].values.astype(float)
    y = D[gcol].astype(str).str.strip().str.upper().eq("E").values.astype(int) if gcol else np.zeros(n, int)
    yv = pd.to_numeric(D[tgts[-1]], errors="coerce").values if tgts else None
    if tgts: target_report(yv, D["_tray"].values, str(tgts[-1]))

    print("=" * 78); print(" 곡선만으로 교란을 분리한다 — 보정 변수를 쓰지 않는 접근"); print("=" * 78)
    print(f"  {n:,}셀 / 트레이 {D['_tray'].nunique()}개 / 시점 {len(mins)}개 / 평가 {at}분"
          + (f" / 불량(E) {int(y.sum())}개" if y.sum() else ""))
    print(f"  쓰는 것: 전류 곡선만.  온도·전압·자리·층 전부 쓰지 않음.")

    # ── [1] 모양 분해 ────────────────────────────────────────────
    Xw = X[:, :ai + 1]
    print("\n" + "-" * 78)
    print(" [1] 모양으로 가른다 — 직선 성분과 곡률 성분")
    print("-" * 78)

    def shape_split(jend):
        """창 [0, jend] 에서 직선 계수와 곡률 계수를 뽑는다."""
        tw = np.asarray(mins[:jend + 1], float)
        L = tw - tw.mean()
        tau = max(tw[-1] / 3.0, 1.0)
        Cb = 1.0 - np.exp(-tw / tau); Cb = Cb - Cb.mean()
        Cb = Cb - (Cb @ L) / (L @ L) * L                  # 직선과 직교
        Xc = X[:, :jend + 1] - pd.DataFrame(X[:, :jend + 1]).groupby(g).transform("mean").values
        return (Xc @ L) / (L @ L), (Xc @ Cb) / (Cb @ Cb)

    from scipy.stats import spearmanr
    tcols = [c for c in ("delta_t", "t_init") if c in D.columns and D[c].nunique() > 2]

    def verdict(jend, label):
        aL_, aC_ = shape_split(jend)
        print(f"\n    ── {label} ({mins[jend]}분 창) ──")
        print(f"    {'':<22}{'직선 계수':>12}{'곡률 계수':>12}")
        print("    " + "-" * 46)
        rY = (np.nan, np.nan)
        if yv is not None:
            rY = (within_tray_rho(aL_, yv, g)[0], within_tray_rho(aC_, yv, g)[0])
            print(f"    {'3일 ΔOCV (트레이내)':<22}{rY[0]:>+12.3f}{rY[1]:>+12.3f}")
        rT = []
        for c in tcols + [c for c in ("delta_v", "v_init") if c in D.columns]:
            r1 = spearmanr(aL_, D[c], nan_policy="omit").statistic
            r2 = spearmanr(aC_, D[c], nan_policy="omit").statistic
            print(f"    {c:<22}{r1:>+12.3f}{r2:>+12.3f}")
            if c in tcols: rT.append((abs(r1), abs(r2)))
        # 대조할 변수가 없으면 '분리 안 됨' 이 아니라 '판정 불가' 다. 구분해야 한다.
        if not np.isfinite(rY[0]) or not rT:
            miss = []
            if not np.isfinite(rY[0]):
                if yv is None:
                    miss.append("3일 ΔOCV (컬럼 자체가 없다)")
                elif not np.isfinite(yv).any():
                    miss.append("3일 ΔOCV (컬럼은 있는데 값이 전부 비어 있다)")
                else:
                    miss.append(f"3일 ΔOCV (값이 {int(np.isfinite(yv).sum()):,}개뿐이라"
                                f" 트레이내 상관을 못 낸다)")
            if not rT: miss.append("온도(delta_t / t_init)")
            print(f"      → ※ 판정 불가 — 대조할 것이 없다: {', '.join(miss)}")
            print(f"        이것은 '분리가 안 된다' 와 다르다. 채우고 다시 볼 것.")
            return aL_, aC_, None
        ok = (abs(rY[0]) > abs(rY[1]) and max(r[1] for r in rT) > max(r[0] for r in rT))
        print(f"      → {'★ 두 성분이 서로 다른 것과 묶인다. 분리 성립.' if ok else '☠ 두 성분이 같은 것과 묶인다. 이 창에서는 분리가 안 된다.'}")
        return aL_, aC_, ok

    aL, aC, ok_at = verdict(ai, "평가 창")
    if ai < len(mins) - 1:
        aLf, aCf, ok_full = verdict(len(mins) - 1, "전체 창")
        if ok_full and not ok_at:
            print(f"""
    ★★ 평가 창({at}분)에서는 안 갈리는데 전체 창({mins[-1]}분)에서는 갈린다.
      열 항의 변곡점이 {at}분 부근이라 짧은 창에서는 곡률이 잘려 나간 것이다.
      → 이 접근을 쓰려면 창을 {mins[-1]}분까지 열어야 한다.
        측정시간 단축과 상충하므로, 둘 중 무엇을 택할지는 [2][3] 으로 판단할 것.""")
        elif ok_full is None or ok_at is None:
            print("""
    ※ 판정을 못 했다. 모양 분해가 기각된 게 아니라 채점을 못 한 것이다.
      · ΔOCV 컬럼이 없으면 : ingest.py --join="본체.xlsx" 로 붙인다.
      · 값이 비어 있으면   : 그 셀들은 3일 보관 ΔOCV 가 안 찍힌 것이다.
        채워진 셀만으로도 트레이당 20개 이상이면 상관이 나온다.
        그보다 적으면 로트를 더 모으는 수밖에 없다.
      · 온도는 시계열에서 자동으로 만들어진다.
      ※ 불량 검출력(최악셀 검사율)은 판정등급으로 재므로 ΔOCV 가 비어도
        그대로 나온다. [2][3] 표는 유효하다.""")
        elif not ok_full and not ok_at:
            print(f"""
    ☠ 두 창 모두에서 안 갈린다. 15~30분 구간에서 자가방전과 열드리프트의
      시간 모양이 충분히 다르지 않다는 뜻이다.
      → 더 촘촘한 원시 샘플(1 Hz)이 있어야 모양 차이를 볼 수 있다.
        지금은 1분 평균 {len(mins)}점뿐이다.""")
        if ok_full and not ok_at:
            aL, aC = aLf, aCf

    # ── [1-b] 공통모드 제거의 함정 검사 ──────────────────────────
    Rm, pc1, ev = tray_svd(Xw, g, k)
    print("\n" + "-" * 78)
    print(" [1-b] 순진한 공통모드 제거가 여기서 통하는가  (함정 검사)")
    print("-" * 78)
    if ev:
        e1 = np.array([v[0] for v in ev.values()])
        # 평가 창에서 트레이 공통 궤적을 뺀 전류 진폭
        amp_ref = (Xw - pd.DataFrame(Xw).groupby(g).transform("mean").values)[:, -1]
        amp = abs(spearmanr(pc1, amp_ref, nan_policy="omit").statistic)
        print(f"    첫 성분이 설명하는 비율   중앙 {np.median(e1)*100:.1f}%")
        print(f"    첫 성분 점수와 전류 진폭의 상관  |rho| = {amp:.3f}")
        if amp > 0.8:
            print(f"""
    ☠ 첫 성분은 교란이 아니라 '진폭' 이다. 모든 곡선의 모양이 비슷해서
      생기는 일이며, 진폭이 곧 자가방전이므로 빼면 신호가 지워진다.
      → 공통모드 제거는 쓰지 말 것. [1] 의 모양 분해를 쓸 것.
        (참고로 이 함정은 곡선 다발에 주성분분석을 쓰는 어떤 접근에도 해당한다.)""")
        else:
            print(f"""
    → 첫 성분이 진폭과 다르다. 모양이 다른 교란이 실제로 존재한다는 뜻이므로
      공통모드 제거가 통할 수 있다. [2] 의 ③④ 를 볼 것.""")

    # ── [2] 점수 비교 ────────────────────────────────────────────
    print("\n" + "-" * 78); print(" [2] 점수 비교"); print("-" * 78)
    tn = lambda v: v - pd.Series(v).groupby(g).transform("median").values
    resid_at = Rm[:, -1]
    recon = np.sqrt((Rm ** 2).mean(1))                      # 잔차 크기 = 비지도 이상 점수
    qs = charge_slope(Xw, mins[:ai + 1], max(mins[0], at // 3), at)
    cand = [("① 전류 끝값 (현행)", X[:, ai]),
            ("② 트레이 정규화 전류", tn(X[:, ai])),
            ("★ 직선 계수 (모양 분해)", aL),
            ("★ 직선 - 곡률 (열 항 뺀 것)", aL - (aL @ aC) / (aC @ aC + 1e-300) * aC),
            (f"③ 공통성분 {k}개 제거 잔차", resid_at),
            ("④ 잔차 크기 (비지도 이상점수)", recon),
            ("⑤ 전하량 기울기", qs),
            ("⑥ 전하량 기울기 + 트레이정규화", tn(qs)),
            ("⑦ 잔차 + 전하량 기울기", tn(qs) / (np.std(tn(qs)) + 1e-30)
                                     + resid_at / (np.std(resid_at) + 1e-30))]
    hdr = f"    {'점수':<30}{'트레이간':>9}"
    if yv is not None: hdr += f"{'트레이내rho':>12}"
    if y.sum(): hdr += f"{'상위1%':>8}{'상위5%':>8}{'최악셀 검사율':>14}"
    print(hdr); print("    " + "-" * (len(hdr) - 4))
    for nm, s in cand:
        line = f"    {nm:<30}{between_tray_share(s, g)*100:>8.1f}%"
        if yv is not None: line += f"{within_tray_rho(s, yv, g)[0]:>12.3f}"
        if y.sum():
            t_ = topk_recall(s, y)
            need = max((np.sum(s > s[i]) + 1) / n for i in np.where(y == 1)[0])
            line += f"{t_[0.01]*100:>7.0f}%{t_[0.05]*100:>7.0f}%{need*100:>13.2f}%"
        print(line)
    print("""
    ④ 는 불량 라벨을 전혀 쓰지 않고 만든 점수다. 그런데도 검출이 되면,
      불량 8개에 의존하지 않는 선별이 가능하다는 뜻이다.
      처음 보는 불량 유형에도 반응한다는 것이 지도학습과 다른 점이다.""")

    # ── [3] 시점별 ───────────────────────────────────────────────
    if y.sum():
        print("\n" + "-" * 78)
        print(" [3] 몇 분이면 되는가 — 같은 틀에서 다시 (모든 셀 동일 측정 전제)")
        print("-" * 78)
        print(f"    {'구간':>6}{'전류 끝값':>11}{'트레이정규화':>13}{'직선 계수':>12}"
              f"{'전하량기울기':>13}{'최선 최악셀':>13}")
        print("    " + "-" * 64)
        for m in mins:
            if m not in (5, 8, 10, 12, 15, 20, 25, 30) or m > mins[-1]: continue
            j = mins.index(m)
            if j < 2: continue            # 시점이 3개 미만이면 기울기·직선계수가 정의되지 않는다
            Xm = X[:, :j + 1]
            q2 = charge_slope(Xm, mins[:j + 1], max(mins[0], m // 3), m)
            t2 = np.asarray(mins[:j + 1], float); L2 = t2 - t2.mean()
            Xc2 = Xm - pd.DataFrame(Xm).groupby(g).transform("mean").values
            aL2 = (Xc2 @ L2) / (L2 @ L2)
            ss = [X[:, j], tn(X[:, j]), aL2, tn(q2)]
            # 상수가 된 점수는 '전부 1등' 이라 거짓 최고값을 만든다. 빼고 계산한다.
            live = [s for s in ss if np.std(s) > 0]
            r5 = [topk_recall(s, y)[0.05] if np.std(s) > 0 else float("nan") for s in ss]
            nd = min((max((np.sum(s > s[i]) + 1) / n for i in np.where(y == 1)[0])
                      for s in live), default=float("nan"))
            f = lambda v, w: (f"{v*100:>{w-1}.0f}%" if np.isfinite(v) else f"{'-':>{w}}")
            print(f"    {m:>4}분{f(r5[0],10)}{f(r5[1],12)}{f(r5[2],11)}{f(r5[3],12)}"
                  + (f"{nd*100:>12.1f}%" if np.isfinite(nd) else f"{'-':>13}"))
        print("""
    → 각 칸은 상위 5% 검사 시 불량 검출률. 맨 오른쪽은 네 점수 중 최선의
      '최악셀 검사율' 이다. 모든 셀이 같은 시간을 측정하므로, 이 표에서
      고르는 것은 '전 셀 공통 측정시간' 하나다.""")

    print("\n" + "=" * 78)
    print(""" 이 접근의 차별점
   · 보정 변수를 쓰지 않는다. 온도계도, 자리도, 전압도 안 쓴다.
     쓰는 것은 전류 곡선 하나뿐이다.
   · 값이 아니라 모양을 읽는다. 자가방전은 15분 창에서 직선으로 자라고,
     열드리프트는 올라갔다 감쇠한다. 직선 성분만 뽑으면 열 항이 스스로 빠진다.
   · 전하량으로 보면 열 항이 유계가 되어 절편으로 흡수된다.
   · 불량 라벨 없이도 이상 점수를 만든다. 불량 8개에 의존하지 않는다.
   · 모든 셀이 같은 측정을 받는다. 측정을 바꾸지 않고 읽는 법만 바꾼다.

 읽는 순서
   1. [1] 의 판정을 먼저 볼 것. 직선 계수가 ΔOCV 쪽에, 곡률 계수가 온도 쪽에
      묶여야 이 접근이 성립한다. 아니면 나머지 표는 볼 필요가 없다.
   2. [1-b] 는 함정 검사다. 곡선 다발에 주성분분석을 쓰려는 어떤 시도에도
      해당하므로, 그 길을 가기 전에 반드시 확인할 것.
   3. [2] 에서 '직선 계수' 가 ② 보다 나으면 이 길이 값어치가 있다.
      특히 '최악셀 검사율' 을 볼 것. 모든 셀이 같은 시간을 측정하는 제약에서는
      이 값이 곧 3일 보관을 없앨 수 있느냐를 결정한다.""")
    print("=" * 78)


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    if not a: print(__doc__)
    else:
        at, k = None, 2
        for x in sys.argv:
            if x.startswith("--at="): at = int(x.split("=")[1])
            if x.startswith("--k="):  k = int(x.split("=")[1])
        sv, en = runlog.parse(sys.argv)
        parse_target(sys.argv)
        with runlog.saving("decompose", a[0], sys.argv, sv, en):
            main(a[0], at, k)
