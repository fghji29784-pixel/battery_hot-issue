# -*- coding: utf-8 -*-
"""
합성 데이터 생성기 — 본체 엑셀과 같은 스키마로 가짜 데이터를 만든다.

  python synth.py out.xlsx            # 41트레이 x 142셀
  python synth.py out.xlsx --trays=10 --per=100

왜 필요한가
  새 스크립트(correct.py / rank_tray.py / adaptive.py)가 실제 엑셀에서
  돌기 전에, 스키마와 계산 경로가 맞는지 확인하기 위한 것이다.
  ★ 여기서 나오는 수치는 전부 가짜다. 발표에 쓰지 말 것.

심어놓은 구조 (실데이터에서 의심하는 것과 같은 구조)
  · 트레이마다 온도 이력이 다르다 → 열드리프트가 트레이별 전류 오프셋을 만든다
  · 전류 = 자가방전(양수) + 열드리프트(부호 자유) + 잡음  → 음수가 나올 수 있다
  · 3일 ΔOCV 는 자가방전에만 연결되고 열드리프트와는 무관하다
    → 보정 전 전류는 ΔOCV 와 상관이 낮고, 보정 후에는 올라야 정상이다
"""
import sys
import numpy as np, pandas as pd

MINS = list(range(5, 31))
KB, EA = 8.617333262e-5, 0.94          # eV/K, eV


def make(n_tray=41, per=142, n_bad=8, seed=0, layer_effect=0.0, grid_effect=0.0,
         grid_flip=0.0, grid_edge=0.0):
    """layer_effect: 트레이 안에서 층이 3일 ΔOCV 를 흔드는 크기 [mV/층].
    grid_effect: 트레이 안 '자리'(행)에 따른 냉각 구배 [K/min per 행].
      실데이터 히트맵처럼 한쪽 행이 더 많이 식게 만든다.
      0 이면 공간 구배 없음. rescue.py 검정용.
    grid_flip: 구배의 부호가 트레이마다 뒤집힐 확률 (0~1).
      0 이면 모든 트레이가 같은 방향. 0.5 면 절반이 반대 방향.
      '자리 보정' 의 전제가 깨진 경우를 만들기 위한 것.
    grid_edge: 가장자리 효과 (0~1). 양 끝 행이 더 많이 식게 만든다.
      실측 41트레이 평균 프로파일이 U자였다. 단조 구배가 아니라
      가장자리가 낮은 모양이며, 선형 평면 모형으로는 못 따라간다."""
    rng = np.random.default_rng(seed)
    n = n_tray * per
    n_bad = min(n_bad, n)
    tray = np.repeat([f"TRAY{i:03d}" for i in range(n_tray)], per)

    # ── 트레이 단위 조건 ──────────────────────────────────────────
    t0_tray   = rng.normal(28.0, 1.6, n_tray)              # 투입 온도 [degC]
    drift_tray= (25.0 - t0_tray) / 260.0                   # 챔버 25도로 수렴 [K/min]

    # 트레이 안 자리 — 셀 번호 1..per 를 열-우선 격자로 (실데이터와 같은 규칙)
    ncol = 12 if per % 12 == 0 else max(1, int(round(np.sqrt(per))))
    pos = np.tile(np.arange(per), n_tray)
    grow = pos // ncol                                      # 행 인덱스

    # 트레이마다 구배 방향이 뒤집힐 수 있다
    sgn = np.repeat(np.where(rng.random(n_tray) < grid_flip, -1.0, 1.0), per)
    gnorm = sgn * grow / max(grow.max(), 1)
    if grid_edge:                                   # 가장자리일수록 더 식는다 (U자)
        mid = grow.max() / 2.0
        gnorm = gnorm + grid_edge * (np.abs(grow - mid) / max(mid, 1))
    ti = np.repeat(t0_tray, per) + rng.normal(0, 0.12, n) \
         + grid_effect * 30.0 * gnorm                       # 한쪽 행이 뜨겁게 들어옴
    dTdt = np.repeat(drift_tray, per) + rng.normal(0, 0.0012, n) \
           - grid_effect * gnorm                            # 그만큼 더 식는다
    tf = ti + dTdt * 30.0
    # 층은 트레이 '안에서' 갈린다. 한 트레이에 여러 층이 섞여 있다.
    layer = np.tile(np.arange(1, 9), n // 8 + 1)[:n]

    v_init = rng.normal(3.860, 0.004, n)
    rwiring = rng.normal(0.031, 0.002, n)

    # ── 셀 고유 자가방전 ─────────────────────────────────────────
    i_sd = np.exp(rng.normal(np.log(5.0e-6), 0.32, n))      # [A] 중앙값 5 uA
    bad = rng.choice(n, n_bad, replace=False)
    i_sd[bad] *= rng.uniform(6.0, 20.0, n_bad)
    tau = 7.45 * 60.0 * np.exp(rng.normal(0, 0.30, n)) * (i_sd / np.median(i_sd)) ** 0.25

    # 아레니우스: 측정 온도가 높으면 자가방전도 크다
    tk = (ti + tf) / 2.0 + 273.15
    i_sd_at_t = i_sd * np.exp(-EA / KB * (1.0 / tk - 1.0 / 298.15))

    # ── 열드리프트 인공전류 ──────────────────────────────────────
    #   I_th = Ceff * dU/dT * dT/dt.  여기서는 합쳐서 계수 K 하나로 둔다.
    #   ★ dT/dt 는 온도가 안정되면서 사라지므로 이 항은 '올라갔다 감쇠' 한다.
    #     실측 셀 140 의 U자 곡선이 그 모양이다. 자가방전(거의 직선)과
    #     모양이 다르다는 것이 모양 분해의 전제이므로 그대로 재현한다.
    K_TH = 9.0e-4                                            # [A / (K/min)]
    TAU_TH = 10.0                                            # [분] 열 항의 시정수
    i_th = K_TH * dTdt

    I = np.empty((n, len(MINS)))
    for j, m in enumerate(MINS):
        settle = 1.0 - np.exp(-m / tau)
        noise = rng.normal(0, 2.5e-7, n)
        shape_th = (m / TAU_TH) * np.exp(1.0 - m / TAU_TH)   # t=TAU_TH 에서 최대, 이후 감쇠
        I[:, j] = i_sd_at_t * settle + i_th * shape_th + noise

    S = np.empty_like(I)
    for j, m in enumerate(MINS):
        S[:, j] = I[:, j] / m

    # ── 타깃: 3일 ΔOCV ───────────────────────────────────────────
    #   자가방전에만 연결. 트레이마다 기준선(전압대)이 다르다.
    base = np.repeat(rng.normal(1.60, 0.09, n_tray), per)
    docv = (base + 2.6e4 * i_sd + rng.normal(0, 0.028, n)
            + layer_effect * (layer - layer.mean()))

    grade = np.array(["A"] * n, dtype=object)
    rest = np.setdiff1d(np.arange(n), bad)
    grade[rng.choice(rest, min(72, len(rest)), replace=False)] = "Q"
    grade[bad] = "E"

    d = {"tray_id": tray, "cell_no": np.tile(np.arange(1, per + 1), n_tray)}
    for j, m in enumerate(MINS):
        d[f"i_{m}min"] = I[:, j]
    for j, m in enumerate(MINS):
        d[f"slope_0_{m}"] = S[:, j]
    d.update({"rwiring": rwiring, "v_init": v_init,
              "v_final": v_init - 0.5 * I[:, -1],
              "t_init": ti, "t_final": tf, "delta_t": tf - ti, "layer": layer,
              **{f"dummy_L{L}": (layer == L).astype(int) for L in range(1, 9)},
              "delta OCV(3day)": docv, "판정등급": grade})
    d["delta_v"] = d["v_final"] - d["v_init"]
    return pd.DataFrame(d)


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    kw = {}
    for x in sys.argv:
        if x.startswith("--trays="): kw["n_tray"] = int(x.split("=")[1])
        if x.startswith("--per="):   kw["per"] = int(x.split("=")[1])
        if x.startswith("--seed="):  kw["seed"] = int(x.split("=")[1])
        if x.startswith("--layer="):  kw["layer_effect"] = float(x.split("=")[1])
        if x.startswith("--grid="):   kw["grid_effect"] = float(x.split("=")[1])
        if x.startswith("--flip="):   kw["grid_flip"] = float(x.split("=")[1])
        if x.startswith("--edge="):   kw["grid_edge"] = float(x.split("=")[1])
    out = a[0] if a else "synth.xlsx"
    df = make(**kw)
    df.to_excel(out, index=False) if out.lower().endswith((".xlsx", ".xlsm")) \
        else df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"  생성: {out}   {df.shape[0]:,}행 x {df.shape[1]}열")
    print("  ※ 전부 가짜 수치입니다. 스크립트 동작 확인용입니다.")
