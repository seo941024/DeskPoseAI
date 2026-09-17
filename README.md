# DeskPoseAI — 책상 앞 집중 자세 + 표정 인식

웹캠 하나로 **"지금 집중하고 있는가"** 를 두 방향에서 읽어낸다 — 자세(포즈)로 한 번, 표정(감정)으로 한 번.
둘 다 사전 학습된 모델(YOLO-Pose, DeepFace)을 그대로 쓰되, 나온 결과를 표시만 하지 않고 분류·집계까지 이어지게 만든 실습 프로젝트.

| 폴더 | 실습 | 핵심 |
|---|---|---|
| [`실습3_Pose/`](./실습3_Pose) | 2D Pose Estimation 기반 행동 인식 | YOLO-Pose로 키포인트 추출 → XGBoost로 자세 3종 분류 |
| [`실습4_Emotion/`](./실습4_Emotion) | DeepFace 나이·성별·감정 인식 | 실시간 나이/성별/감정 추출 + 감정 집계 + 검출 백엔드 비교 |
| [`실습3+4_Combined/`](./실습3+4_Combined) | 통합 데모 (보너스) | 자세 라벨 + 나이/성별/감정을 한 화면에 동시 표시 |

---

## 실습3_Pose — 자세 인식

책상 앞에 앉은 상태(전신 불필요)에서 3가지 집중 자세를 실시간으로 분류한다.

- `facing_forward`(정면 보기) · `head_down`(고개 숙이기) · `chin_on_hand`(턱 괴기)
- YOLO-Pose(`yolo26n-pose.pt`)로 17개 키포인트를 직접 추출, **사람 박스 기준으로 정규화**한 좌표를 XGBoost로 분류
- 키포인트 좌표를 반전·지터로 증강해 학습 샘플 확보
- **박스 기준 정규화 적용 전후 정확도 58.8% → 97.8%** — 이 비교가 "파생 특징 추가 후 성능 변화" 근거

자세한 실행 방법은 [실습3_Pose/README.md](./실습3_Pose/README.md) 참고.

## 실습4_Emotion — 나이·성별·감정 인식

DeepFace로 얼굴에서 나이·성별·감정을 동시에 뽑아 실시간으로 표시한다.

- 감정 결과를 표시만 하지 않고 `Counter`로 누적 집계 → 화면에 "지금까지 우세 감정" 표시, 종료 시 콘솔에 비율 요약
- 검출 백엔드(`opencv` / `mtcnn` / `retinaface` 등) 2개 이상을 같은 조건에서 비교
- 분석(무거운 DeepFace 호출)을 별도 스레드로 돌려 웹캠 화면은 끊기지 않게 함
- 보너스: `01_AgeGender/Age_Gender_Recognition.py` — DeepFace가 아닌 별도 WideResNet 모델로 나이·성별만 따로 확인

자세한 실행 방법은 [실습4_Emotion/README.md](./실습4_Emotion/README.md) 참고.

## 실습3+4_Combined — 통합 데모 (보너스)

두 실습을 한 화면에 합쳐서 자세 라벨 + 나이/성별/감정을 동시에 보여준다.

| head_down | chin_on_hand | facing_forward |
|---|---|---|
| ![head_down](./실습3+4_Combined/docs/demo_head_down.png) | ![chin_on_hand](./실습3+4_Combined/docs/demo_chin_on_hand.png) | ![facing_forward](./실습3+4_Combined/docs/demo_facing_forward.png) |

자세한 실행 방법은 [실습3+4_Combined/README.md](./실습3+4_Combined/README.md) 참고.

---

## 공통 설치

```bash
conda activate vision_new
pip install ultralytics opencv-contrib-python pandas xgboost scikit-learn numpy deepface
```

## 원본 대비 바뀐 점

두 실습 모두 원본 슬라이드 기본 소스코드(웹캠에 결과만 표시)로 시작했고,
아래를 채워서 슬라이드가 요구하는 필수 요건(직접 수집, 커스텀 클래스 3개 이상, 판단/집계 활용, 조건 비교, 실시간 데모)을 충족시켰다.

- 포즈 모델을 `yolov8n-pose.pt` → `yolo26n-pose.pt` 로 교체
- 포즈: 키포인트 정규화 방식을 화면 전체 기준 → 사람 박스 기준으로 수정 (정확도 대폭 개선)
- 포즈: 좌표 증강 스크립트 추가
- 감정: 감정 결과를 집계·판단에 활용하도록 추가, 백엔드 비교 스크립트 추가, 분석을 스레드로 분리해 프레임 드랍 해결
