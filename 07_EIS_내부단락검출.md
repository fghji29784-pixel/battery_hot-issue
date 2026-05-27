# 07. EIS 기반 내부단락(ISC) 조기 검출 기법 (2023–2026)

---

## 1. 개요

전기화학 임피던스 분광법(EIS, Electrochemical Impedance Spectroscopy)은 배터리 내부 저항, SEI 상태, 확산 특성을 비파괴적으로 측정하는 기법이다. 2023–2025년 딥러닝과 결합하여 내부단락(ISC) 조기 검출에 탁월한 성능을 보이는 연구가 집중 발표되었다.

---

## 2. EIS + 딥러닝 기반 내부단락 조기 검출 ★ 핵심 기술

### 2.1 DNN + EIS 조기 검출 (2023, ScienceDirect)

**핵심 성과**:
- 내부단락 검출 평균 정확도: **97.5%**
- 정상 배터리에 대한 위양성(False Positive): **0건**
- ISC 저항 200Ω ~ 10Ω 범위에서 전체 배터리 수명 주기에 걸쳐 유효

**방법론**:
- EIS 데이터 + 심층 신경망(DNN) 조합
- 단락된 셀: 임피던스가 정상 셀과 다름 → 확산계수, 분극 저항 차이로 ISC 심각도 구분

**출처**: "Internal short circuit early detection of lithium-ion batteries from impedance spectroscopy using deep learning", *Journal of Power Sources*, 2023  
URL: https://www.sciencedirect.com/science/article/abs/pii/S0378775323001994  
ADS: https://ui.adsabs.harvard.edu/abs/2023JPS...56332824C/abstract

### 2.2 ML 기반 ISC 조기 검출 (2025, ScienceDirect)

- 머신러닝 기반 조기 단계 내부단락 검출
- 다양한 심각도의 ISC 식별 가능
- **출처**: "Machine learning-based early-stage internal short circuit detection using EIS", *ScienceDirect*, 2025  
  URL: https://www.sciencedirect.com/science/article/abs/pii/S001346862501998X

---

## 3. 고속 EIS 측정 기술 (2025)

### 3.1 멀티사인 가진 고속 EIS

- **15초** EIS 데이터 취득 (다중 정현파 동시 가진)
- 다양한 심각도의 ISC 식별 가능
- 배터리 수명 전반에 걸쳐 ISC 검출 가능

**출처**: 2025년 연구 (ScienceDirect, 위 동일)

### 3.2 완화 시간 분포(DRT) 기반 최적화

- Distribution of Relaxation Times(DRT) 분석으로 ISC에 민감한 EIS 주파수 선별
- 측정 시간 단축 + 계산 효율 향상

---

## 4. 외부단락 고장 평가 (EIS 기반, 2025)

- EIS + 완화 시간 분포의 차분 특성 추출로 외부단락 고장 평가
- **출처**: "Research on External Short Circuit Fault Evaluation Method Based on Impedance Spectrum Feature Extraction", *Batteries*, MDPI, 2025  
  URL: https://www.mdpi.com/2313-0105/11/12/437

---

## 5. EIS 측정 원리 및 배터리 진단 적용

### 5.1 EIS 측정 파라미터

| 파라미터 | 의미 | 저전압 불량 연관성 |
|---------|------|----------------|
| Rct (Charge Transfer Resistance) | 계면 반응 저항 | 내부단락 시 감소 |
| Warburg Impedance | 확산 저항 | 함침 불량 시 변화 |
| Rs (Bulk Resistance) | 전해액 저항 | 전해액 열화 지표 |
| CPE (Constant Phase Element) | SEI 특성 | SEI 품질 평가 |
| 분극 저항 | 전극 반응 | ISC 심각도 구분 |

### 5.2 Formation 공정 EIS 적용 포인트

```
Formation 공정 흐름:
전해액 주입 → 함침 → Pre-charge → Formation 충방전 → 디개싱 → Aging

EIS 적용 포인트:
- Formation 전: 함침 상태 평가
- Formation 중: SEI 형성 모니터링, 리튬 플레이팅 검출
- Formation 후: 내부단락 검출, SOH 분류
- Aging 후: 최종 불량 선별
```

---

## 6. EIS 장비 및 실용화 고려사항

### 6.1 인라인 EIS 측정 가능성
- 기존 EIS는 측정 시간 수십 분 ~ 수 시간 소요
- **멀티사인 EIS**: 15초로 단축 → 인라인 적용 실용적

### 6.2 Formation 충방전기 EIS 모듈 통합
- 일부 제조사(Biologic, Gamry 등) 고속 EIS 모듈 제공
- Formation 완료 직후 EIS 자동 측정 → 불량 셀 즉시 선별

### 6.3 EIS 데이터 해석 자동화
- DNN 모델을 Formation 충방전기 소프트웨어에 통합
- 실시간 EIS 스펙트럼 → 자동 ISC 위험도 점수 산출

---

## 7. ICA + EIS 복합 진단 전략

```
[제안 복합 진단 시스템]

Formation 1주기 완료
↓
IC 곡선 자동 생성 → 이상 피크 → 1차 필터링
↓
의심 셀: 15초 EIS 측정
↓
DNN으로 ISC 위험도 점수 계산
↓
고위험 셀: 추가 자가방전 측정 (6분 OCV 예측)
↓
최종 판정: 정상 / 재검사 / 불량

기대 효과:
- 7일 자가방전 대기 → 수 시간 이내 완료
- 인력 개입 최소화
- 정량적 위험도 점수 체계
```

---

## 출처 목록

1. ScienceDirect, "ISC early detection from EIS using deep learning", *J. Power Sources*, 2023  
   https://www.sciencedirect.com/science/article/abs/pii/S0378775323001994
2. ScienceDirect, "ML-based early-stage ISC detection using EIS", 2025  
   https://www.sciencedirect.com/science/article/abs/pii/S001346862501998X
3. MDPI Batteries, "External Short Circuit Fault Evaluation via EIS", 2025  
   https://www.mdpi.com/2313-0105/11/12/437
4. SSRN Preprint, "ISC Early Detection from EIS"  
   https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4310664
