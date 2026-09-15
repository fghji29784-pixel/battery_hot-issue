"""
SDM 데이터 진단 — 사내에서 실행하고, 출력된 '숫자 표'만 보면 됩니다.

이 스크립트는 네트워크를 쓰지 않고, 원본 데이터를 저장하거나 출력하지 않습니다.
셀 단위 값은 일절 찍지 않고 집계 통계만 출력합니다.

입력 형식 (둘 중 아무거나)
  (A) wide  : 첫 컬럼이 time_s, 나머지 각 컬럼이 셀 하나의 전류
  (B) long  : cell_id, time_s, current_uA  (+ 선택: tray_id)
  별도 파일 : cell_id, docv_3d_mV  (+ 선택: tray_id, label)

실행
  python3 diagnose.py curves.csv                 # 곡선만
  python3 diagnose.py curves.csv targets.csv     # 곡선 + 3일 ΔOCV
"""
import sys, numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr
from scipy.optimize import curve_fit

CHECK_TIMES = [60, 180, 300, 600, 900, 1200, 1800, 3600, 7200, 10800]  # 초

def load(path):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    if "cell_id" in cols and "time_s" in cols:                       # long
        cur = next(c for c in df.columns if "cur" in c.lower())
        W = df.pivot_table(index=cols["time_s"], columns=cols["cell_id"], values=cur)
        return W.index.values.astype(float), W.values.T, list(W.columns)
    t = df[df.columns[0]].values.astype(float)                        # wide
    return t, df[df.columns[1:]].values.T.astype(float), list(df.columns[1:])

def _exp2(t, Iinf, A1, t1, A2, t2):
    return Iinf - A1*np.exp(-t/t1) - A2*np.exp(-t/t2)

def main(curve_path, target_path=None):
    t, I, ids = load(curve_path)
    t = t - t[0]; n, T = I.shape
    print("="*74); print(" SDM 데이터 진단"); print("="*74)
    print(f"  셀 {n}개 / 시점 {T:,}개 / 측정 길이 {t[-1]/3600:.2f} 시간"
          f" / 샘플 간격 {np.median(np.diff(t)):.1f} 초")

    I_fin = I[:, -max(T//50, 3):].mean(axis=1)
    print(f"  최종 전류: 중앙값 {np.median(I_fin):.1f} µA, "
          f"5~95% {np.percentile(I_fin,5):.0f}~{np.percentile(I_fin,95):.0f} µA, "
          f"최대 {I_fin.max():.0f} µA")

    # ── [A] 정착 시정수 ───────────────────────────────────────────────
    print("\n" + "-"*74); print(" [A] 정착 시정수 (중앙값 곡선 기준)"); print("-"*74)
    med = np.median(I, axis=0)
    try:
        p, _ = curve_fit(_exp2, t, med,
                         p0=[med[-1]*1.1, med[-1]*.5, t[-1]/20, med[-1]*.5, t[-1]/3],
                         bounds=([0,0,1,0,1], [np.inf]*5), maxfev=80000)
        tf, ts = sorted([p[2], p[4]])
        print(f"  I∞ 추정 {p[0]:.1f} µA   τ_fast {tf/60:.1f}분   τ_slow {ts/3600:.2f}시간")
        print(f"  측정 종료 시점의 정착률 = {(1-np.exp(-t[-1]/ts))*100:.1f}%")
        print(f"  15분 시점의 정착률      = {(1-np.exp(-900/ts))*100:.1f}%")
    except Exception as e:
        print("  피팅 실패:", e)

    # ── [B] ★ 순위 보존 — 이 표가 핵심 ────────────────────────────────
    print("\n" + "-"*74)
    print(" [B] ★ 순위 보존:  t 시점의 값이 최종 순위를 얼마나 재현하는가")
    print("-"*74)
    print(f"  {'t':>8} {'Spearman ρ':>12} {'Pearson r':>11} {'상위10% 재현율':>15}")
    print("  " + "-"*50)
    k = max(n//10, 1); top_fin = set(np.argsort(-I_fin)[:k])
    for ct in CHECK_TIMES:
        if ct > t[-1]: break
        j = int(np.argmin(np.abs(t-ct))); v = I[:, j]
        rho = spearmanr(v, I_fin).statistic; r = pearsonr(v, I_fin)[0]
        rec = len(set(np.argsort(-v)[:k]) & top_fin)/k
        lab = f"{ct//60}분" if ct < 3600 else f"{ct/3600:.0f}시간"
        star = "  ← 15분" if ct == 900 else ""
        print(f"  {lab:>8} {rho:>12.3f} {r:>11.3f} {rec*100:>13.0f}%{star}")
    print("""
  해석:  ρ > 0.9  → 그 시점이면 최종 순위를 거의 그대로 재현. 충분
         ρ 0.7~0.9 → 쓸 만함. 다변량 피처로 보완하면 실용적
         ρ < 0.7  → 그 시점은 부족. 더 긴 측정이 필요""")

    # ── [C] 기울기 피처도 함께 보기 ───────────────────────────────────
    print("\n" + "-"*74); print(" [C] 15분 구간에서 뽑은 피처별 최종값 상관"); print("-"*74)
    j9 = int(np.argmin(np.abs(t-900)))
    if j9 > 5:
        seg_t, seg = t[:j9+1], I[:, :j9+1]
        feats = {
            "값 I(900s)":        seg[:, -1],
            "기울기 (후반 1/3)": np.array([np.polyfit(seg_t[-j9//3:], s[-j9//3:],1)[0]*3600 for s in seg]),
            "적분 ∫I dt":        np.trapezoid(seg, seg_t, axis=1)/seg_t[-1],
            "곡률 (기울기 변화)": np.array([np.polyfit(seg_t[:j9//2], s[:j9//2],1)[0]
                                        - np.polyfit(seg_t[-j9//2:], s[-j9//2:],1)[0] for s in seg]),
        }
        for nm, v in feats.items():
            print(f"    {nm:<20} ρ = {spearmanr(v, I_fin).statistic:+.3f}")
        print("\n  → 값 하나보다 기울기·곡률이 더 높으면, 다변량 피처가 유효하다는 뜻")

    # ── [D] 목표(3일 ΔOCV)와의 상관 ───────────────────────────────────
    if target_path:
        tg = pd.read_csv(target_path); tc = {c.lower(): c for c in tg.columns}
        idc = tc.get("cell_id"); dc = next(c for c in tg.columns if "ocv" in c.lower())
        m = pd.DataFrame({"cell_id": ids, "I900": I[:, j9], "Ifin": I_fin}).merge(
            tg[[idc, dc]].rename(columns={idc:"cell_id", dc:"docv"}), on="cell_id")
        print("\n" + "-"*74); print(" [D] ★★ 3일 ΔOCV 와의 상관 — 최종 판정"); print("-"*74)
        print(f"  매칭된 셀 {len(m)}개,  ΔOCV 중앙값 {m.docv.median():.3f} mV,"
              f" 5~95% {m.docv.quantile(.05):.3f}~{m.docv.quantile(.95):.3f} mV")
        for nm, v in [("15분 전류 → ΔOCV", m.I900), ("최종 전류 → ΔOCV", m.Ifin)]:
            print(f"    {nm:<18} ρ = {spearmanr(v, m.docv).statistic:+.3f}"
                  f"   r = {pearsonr(v, m.docv)[0]:+.3f}")
        print("""
  ★ 이 두 줄이 프로젝트의 가능/불가능을 가릅니다.
     '최종 전류 → ΔOCV' 가 낮으면  → 두 측정이 애초에 다른 걸 보고 있음.
                                      (SOC·이완 조건부터 점검해야 함)
     '최종'은 높은데 '15분'이 낮으면 → 측정 시간이 부족. [B] 표에서 필요 시간 확인
     둘 다 높으면                    → 바로 진행 가능""")

    print("\n" + "="*74)
    print(" 공유해 주실 것: 위 [A] 요약 3줄, [B] 표, [C] 4줄, [D] 2줄")
    print(" (셀 단위 값이나 원본은 전혀 포함되지 않습니다)")
    print("="*74)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(0)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
