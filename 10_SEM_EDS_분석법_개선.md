# 10. SEM/EDS 분석 자동화 및 현행 분해 검사 개선 방안 (2023–2026)

---

## 1. 현행 저전압 불량 분석 프로세스의 문제

### 현행 방법
```
저전압 불량 선별 → 셀 분해 (글로브박스) → 분리막 육안 확인 (검은 스팟) 
→ SEM/EDS 이물 성분 분석 → 이물 발생 개소 역추적
```

### 문제점
| 문제 | 영향 |
|------|------|
| 육안 검사 의존 | 미세 스팟(수십 μm 이하) 누락 가능 |
| 수동 SEM/EDS 분석 | 분석 1건당 수 시간 ~ 수 일 소요 |
| 파괴 검사 | 분해 과정에서 이물 위치 변형, 공기 노출 |
| 정량화 어려움 | 이물 크기·분포 통계 데이터 부족 |
| 재현성 부족 | 분해자에 따라 결과 편차 |

---

## 2. SEM-EDS 자동화 기술

### 2.1 종합 정량 SEM-EDS 분석 프로세스 (2025)

- 리튬이온전지 전극에 적용 가능한 종합적·정량적 SEM-EDS 분석 방법론
- 자동화된 입자 탐지, 성분 매핑, 크기 분포 통계 생성
- **출처**: PMC, "A comprehensive and quantitative SEM–EDS analytical process applied to lithium-ion battery electrodes", 2025  
  URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC11825835/

### 2.2 HORIBA XGT-9000 — 분리막 이물 자동 분석 ★

**기능**:
- 분리막 위 이물 성분 자동 검출 및 동정
- 이물 크기 계측 및 분포 지도 자동 생성
- 좌표 기록 → 재분석 재현성 확보
- EDS로 Fe, Cu, Cr, Zn, Al 등 성분 자동 분류

**현행 수동 SEM/EDS 대비 장점**:
- 분석 시간 대폭 단축
- 자동 좌표 기록으로 재현성 확보
- 통계 데이터 자동 생성

**출처**: HORIBA Application Note  
URL: https://www.horiba.com/usa/scientific/applications/energy/foreign-particle-analysis-on-a-separator-film-of-lithium-ion-battery/

### 2.3 3D 구조·성분·원자가 동시 매핑 (2025)

- ADF-EDS-EELS 토모그래피 동시 활용으로 3D 구조+성분+원자가 상태 동시 분석
- 양극 활물질 劣化 메커니즘 규명에 활용
- **출처**: arXiv, "Correlative 3D Mapping of Structure, Composition, and Valence State in Battery Cathodes via Simultaneous ADF-EDS-EELS Tomography", 2025  
  URL: https://arxiv.org/pdf/2509.23034

### 2.4 AFM-in-SEM 상관 분석 (2025)

- SEM 내에서 AFM(원자간력현미경) 동시 수행
- 차세대 배터리 전극 표면 나노 스케일 특성 분석
- **출처**: Microscopy and Microanalysis, 2025  
  URL: https://academic.oup.com/mam/article/31/Supplement_1/ozaf048.646/8213366

---

## 3. 이물 오염 실패 분석 최신 연구

### 3.1 배터리 파티클 오염 실패 분석 (2025, IRPS)

- 실패한 코인셀에서 2D X-ray 이미징 → 글로브박스 내 분해 → 에어리스 SEM 전송 → EDS 분석
- 음극 내 복수의 금속 입자 확인
- **출처**: EAG Laboratories, IRPS 2025 발표  
  URL: https://www.eag.com/wp-content/uploads/2025/04/IRPS-2025-Poster_DanielLiu_w.pdf

### 3.2 NCM 배터리 양극 분석 (ScienceDirect, 2024)

- 전이금속(Co, Ni, Mn) 분포 + 오염물(Fe, Cu, Al) EDS 원소 매핑
- **출처**: ScienceDirect, NREL, 2024  
  URL: https://docs.nrel.gov/docs/fy24osti/87220.pdf

---

## 4. 비파괴 대안 기술과의 비교

### 4.1 현행 파괴 분석 vs 비파괴 대안

| 분석 항목 | 현행 방법 | 비파괴 대안 | 실용성 |
|---------|---------|----------|------|
| 이물 위치 확인 | 분해+육안 | X-ray CT | ✅ 상용화 중 |
| 이물 성분 분석 | SEM/EDS (분해 후) | HORIBA XGT (분해 후 자동화) | ✅ 도입 가능 |
| 이물 크기·분포 | 수동 측정 | 자동화 SEM-EDS | ✅ 도입 가능 |
| 내부 단락 위치 | 분해+열화상 | 레이저 초음파 C-스캔 | ✅ 연구 단계 |
| 분리막 손상 확인 | 육안 | X-ray CT (고해상도) | ⚠️ 해상도 한계 |
| 덴드라이트 확인 | SEM (분해 후) | In-situ X-ray | ⚠️ 연구실 수준 |

### 4.2 단기 개선 로드맵

```
[Phase 1: 기존 파괴 분석 품질 향상]
- HORIBA XGT-9000 도입: 분리막 이물 자동 좌표 기록 + 성분 자동 분류
- 자동화 SEM-EDS 소프트웨어 활용으로 분석 시간 단축
- 글로브박스 에어리스 전송 프로토콜 표준화

[Phase 2: 비파괴 전처리 스크리닝 도입]
- X-ray CT로 분해 전 이물 위치 사전 파악
- 레이저 초음파로 기포·공동 사전 스크리닝
- 분해 대상 셀을 미리 선별하여 파괴 분석 건수 감소

[Phase 3: 완전 비파괴 진단 시스템]
- 인라인 X-ray CT + AI 이물 자동 분류
- EIS+DNN으로 Formation 완료 직후 불량 셀 선별
- 파괴 분석 제로화 목표
```

---

## 5. 배터리 소재 SEM 분석 일반 지침

### 5.1 SEM으로 분석 가능한 배터리 불량 유형
- 입자 형태·크기 분포 불균일
- 층간 박리(Delamination)
- 입자 내 균열(Intragranular Cracking)
- 코팅 불균일
- 오염 입자 (Fe, Cu, Cr, Zn 등)

**출처**: "SEM for Lithium-Ion Battery Materials", *Scimed*  
URL: https://www.scimed.co.uk/education/sem-for-lithium-ion-battery-materials/

### 5.2 EDS 원소 분석 대상

- **양극**: Co, Ni, Mn, Fe, Al, O 분포
- **음극**: C, Si, 금속 오염물 (Fe, Cu)
- **분리막 이물**: Fe, Cu, Cr, Zn, Ni, Al 등 금속 원소

**출처**: Thermo Fisher Scientific, "Elemental Imaging of Battery Materials Using EDS Analysis"  
URL: https://www.thermofisher.com/us/en/home/materials-science/battery-research/technologies/eds-analysis.html

---

## 출처 목록

1. PMC, "Comprehensive and quantitative SEM-EDS analytical process for LIB electrodes", 2025  
   https://pmc.ncbi.nlm.nih.gov/articles/PMC11825835/
2. HORIBA, "Foreign Particle Analysis on Separator Film"  
   https://www.horiba.com/usa/scientific/applications/energy/foreign-particle-analysis-on-a-separator-film-of-lithium-ion-battery/
3. EAG/IRPS 2025, "Failure Analysis of Particle Contamination in Battery"  
   https://www.eag.com/wp-content/uploads/2025/04/IRPS-2025-Poster_DanielLiu_w.pdf
4. arXiv, "3D Mapping via Simultaneous ADF-EDS-EELS Tomography", 2025  
   https://arxiv.org/pdf/2509.23034
5. Oxford Academic, "Correlative AFM-in-SEM for Next-Gen Batteries", 2025  
   https://academic.oup.com/mam/article/31/Supplement_1/ozaf048.646/8213366
6. Thermo Fisher, "Elemental Imaging Using EDS Analysis"  
   https://www.thermofisher.com/us/en/home/materials-science/battery-research/technologies/eds-analysis.html
7. Scimed, "SEM for Lithium-Ion Battery Materials"  
   https://www.scimed.co.uk/education/sem-for-lithium-ion-battery-materials/
8. NREL/ScienceDirect, "NCM Battery Analysis", 2024  
   https://docs.nrel.gov/docs/fy24osti/87220.pdf
