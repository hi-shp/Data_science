# 자율운항보트 경로 계획 강화학습 환경 구현
**Autonomous Surface Vehicle Navigation: From Reactive Avoidance to Map-Based Path Planning**

본 프로젝트는 단순한 제어 파라미터 최적화의 한계를 데이터 과학적 관점에서 분석하고, 이를 극복하기 위해 **인지(Perception) - 판단(Planning) - 제어(Control)** 프로세스를 완전히 재설계한 자율운항보트 소프트웨어 스택 개발 과정을 담고 있습니다.

---

## 1. 프로젝트 배경 및 문제 정의

### 1.1 초기 접근 방식의 한계 (시행착오)
처음에는 `avoidance_strength`, `gps_gain`, `gap_gain`이라는 3가지 핵심 파라미터를 조정하며 완주율을 예측하는 기계학습 모델을 구축하려 했습니다. 그러나 실제 데이터 생성 과정에서 다음과 같은 결정적 결함을 발견했습니다.

* **연산 효율 저하**: 시뮬레이션 속도가 너무 느려 10,000회의 시뮬레이션을 수행하는 것이 현실적으로 불가능했습니다.
* **구조적 비일관성**: 장애물 배치의 랜덤성으로 인해 동일 파라미터 내에서도 완주율 편차가 극심하게 나타났습니다.
* **제어 로직의 원시성**: 단순히 벡터를 가감하는 1차적 회피 로직은 특정 상황에서 무한 진동 패턴(Oscillation)을 유발했습니다.

![image01](https://raw.githubusercontent.com/hi-shp/Data_science/main/images/image01.png)
*시행착오가 담긴 초기 장면: 연산 시간이 과도하게 길고 구조적으로 불안정했던 구간*

---

## 2. 시스템 아키텍처 및 구현 (System Architecture)

### 2.1 실시간 환경 인지 및 격자 지도 (Perception)
기존의 즉흥적인 명령 생성 방식에서 탈피하여, 라이다 데이터를 기반으로 장애물 지도를 만들고 웨이포인트를 추출하는 구조로 전환했습니다.

![image02](https://raw.githubusercontent.com/hi-shp/Data_science/main/images/image02.png)  
*기존 알고리즘 구조: 매 프레임 즉흥적 명령 산출로 인해 신뢰성이 낮았던 방식*

* **Coarse Grid Map**: $600 \times 2000$ px 경기장을 $8 \times 8$ px 단위 격자로 압축하여 연산량을 대폭 감소시켰습니다.
* **DBSCAN Clustering**: `sklearn`의 DBSCAN을 도입하여 불연속적인 라이다 점들을 군집화하고 장애물의 중심(Centroid)을 추정했습니다.

![image03](https://raw.githubusercontent.com/hi-shp/Data_science/main/images/image03.png)  
*DBSCAN 도입 장면: 불연속적 점들을 군집화해 장애물 형태를 매우 정확하게 추정*

---

## 3. 경로 계획 및 제어 (Planning & Control)

보트의 동역학적 특성을 고려한 부드러운 경로 생성을 위해 **2차 베지어 곡선(Quadratic Bézier Curve)**과 **Pure Pursuit** 알고리즘을 결합했습니다.

### 3.1 베지어 곡선 및 Pure Pursuit
* **Bezier Path**: 현재 위치($p_0$), 예측 위치($p_1$), 웨이포인트($p_2$)를 이용해 부드러운 궤적을 생성합니다.
* **Pure Pursuit**: 생성된 곡선 위에서 일정 거리 앞의 점을 실시간 추종하여 조향의 불연속성을 제거했습니다.

![image04](https://raw.githubusercontent.com/hi-shp/Data_science/main/images/image04.png)  
*Pure Pursuit와 베지어 곡선을 결합하여 경로의 lookahead 지점을 추종하도록 구성* 

![image05](https://raw.githubusercontent.com/hi-shp/Data_science/main/images/image05.png)  
*베지어 곡선 원리: $(1-t)^2p_0 + 2(1-t)tp_1 + t^2p_2$ 공식을 사용한 궤적 생성*

---

## 4. 최종 결과 및 성능 분석

알고리즘 구조 전환 후, 시스템의 신뢰성과 효율성이 획기적으로 개선되었습니다.

| 분석 항목 | 기존 알고리즘 (Reactive) | 개선된 알고리즘 (Proposed) |
| :--- | :--- | :--- |
| **평균 완주 시간** | 약 50초 내외 (실패 빈번) | **약 10초 내외 (최대 5배 개선)** |
| **완주 성공률** | 매우 낮음 | **약 96.6% (30회 중 29회 성공)** |
| **주행 안정성** | 좌우 미세 진동 및 급선회 | **매끄러운 베지어 경로 추종** |

![image06](https://raw.githubusercontent.com/hi-shp/Data_science/main/images/image06.png)  
![image07](https://raw.githubusercontent.com/hi-shp/Data_science/main/images/image07.png)  
*최종 시뮬레이션 장면: 그리드 지도 위에서 베지어 곡선 경로를 따라 안정적으로 주행하는 모습*

---

## 5. 결론 및 성취

본 프로젝트는 단순한 파라미터 조정을 넘어, 실제 업계 알고리즘 구조와 유사한 **지도 기반 자율운항 시스템**을 직접 설계하고 검증했습니다.
* **기술적 성취**: `sklearn`의 기계학습 모델을 제어 중핵에 적용하고, 벡터화 연산을 통해 실시간성을 확보했습니다.
* **실전성**: 라이다와 GPS라는 최소한의 센서 구성으로도 높은 완성도를 달성했으며, 본 알고리즘은 실제 전국 자율운항보트 경진대회(KABOAT)의 핵심 소프트웨어 스택으로 채택되었습니다.

---

## 6. 설치 및 실행 (Quick Start)

### Prerequisites
* Python 3.8+
* Pygame, NumPy, SciPy, Scikit-learn

### Execution
```bash
python simulation.py
