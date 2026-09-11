# 실습 3 — 2D Pose Estimation 기반 행동 인식

원본 소스: [Tory-Hwang/06_PoseEstimation](https://github.com/Tory-Hwang/06_PoseEstimation)
**모델은 강사 안내대로 `yolov8n-pose.pt` 대신 `yolo26n-pose.pt` 로 변경함.**

책상 앞 웹캠 기준으로 **집중 자세 3종을 실시간으로 분류**한다 (슬라이드 추천 "강의실 집중 자세 모니터" 컨셉).
전신이 필요 없고 상반신/얼굴만 보이면 동작함.

| 자세 | 뜻 |
|---|---|
| `facing_forward` | 정면 보기 (집중) |
| `head_down` | 고개 숙이기 (졸음/딴짓) |
| `chin_on_hand` | 턱 괴기 |

## 폴더 구조

```
실습3_Pose/
├── 00_Models/                          # yolo26n-pose.pt (최초 실행 시 자동 다운로드)
└── 01_PoseAction/
    ├── 12.2D_Pose_Estimation_test.py   # 1부: 웹캠에 포즈 추정 결과만 표시 (스모크 테스트)
    ├── 13.2D_Pose_Save.py              # 2부: 자세 이름을 인자로 받아 웹캠에서 해당 폴더에 누적 수집
    ├── 13b_quick_collect.py            #   ㄴ 숫자키로 자세 전환하며 한 세션에 빠르게 모으는 버전
    ├── 13c_collect_from_images.py      #   ㄴ 이미지 폴더(raw_images/<자세>/)에서 일괄 추출하는 버전
    ├── 14.2D_Pose_Dataset_Save.py      # 3부: 폴더별로 라벨을 붙여 dataset.csv 생성
    ├── 14b_augment_keypoints.py        #   ㄴ 좌표 증강 (반전·지터) → dataset_aug.csv
    ├── 15.2D_Pose_Training.py          # 4부: XGBoost 분류기 학습 (dataset_aug.csv 우선 사용)
    ├── 16.2D_Pose_ActionRecognition.py # 5부: 실시간 행동 인식 데모
    └── pose_img/person/classes.json    # 자세 목록 — 직접 수정 가능
```

## 설치

```bash
conda activate vision_new
pip install ultralytics opencv-contrib-python pandas xgboost scikit-learn numpy
```

## 실행 순서

```bash
cd 01_PoseAction

# 1) 스모크 테스트 - 모델/카메라 확인
python 12.2D_Pose_Estimation_test.py

# 2) 자세별 웹캠 수집 (자세마다 실행, 몇 초씩 그 자세 유지하다가 q)
python 13.2D_Pose_Save.py facing_forward
python 13.2D_Pose_Save.py head_down
python 13.2D_Pose_Save.py chin_on_hand

# 3) 라벨 붙이기 -> 증강 -> 학습
python 14.2D_Pose_Dataset_Save.py
python 14b_augment_keypoints.py
python 15.2D_Pose_Training.py

# 4) 실시간 데모
python 16.2D_Pose_ActionRecognition.py
```

## 핵심 파생 특징 — 박스 기준 정규화

키포인트를 **화면 전체 기준이 아니라 "이 사람의 검출 박스" 기준으로 정규화**해서 저장·예측한다:

```
x' = (px - box_x1) / box_width
y' = (py - box_y1) / box_height
```

카메라와의 거리·프레임 안에서의 위치가 사진마다 달라도 값이 일정해져서, 모델이 "사진 속 위치"가 아니라
순수하게 "자세 모양"만 학습하게 된다. `13.2D_Pose_Save.py`(수집)와 `16.2D_Pose_ActionRecognition.py`(예측)이
반드시 같은 정규화 방식을 써야 함 — 방식이 어긋나면 예측이 랜덤에 가까워진다.

**효과**: 화면 전체 기준 정규화였을 때 val accuracy **58.8%** (train loss는 0에 가까운데 val loss 1.08 — 과적합) →
박스 기준 정규화 적용 후 **97.8%** 로 개선.

## 데이터 증강

수집한 실사진 수는 적당한 수준(자세당 100장 내외)이라, `14b_augment_keypoints.py`가
**이미지가 아니라 키포인트 좌표만** 증강해서 학습 샘플을 늘린다 (XGBoost 입력이 좌표라서 이걸로 충분함):

- **좌우 반전** — 좌/우 관절 쌍을 서로 바꾸고 x좌표를 뒤집음 (거울 자세도 같은 라벨)
- **위치·크기 지터** — 중심 기준 ±12% 확대/축소, ±5% 이동 (카메라 거리·위치 차이를 흉내)
- **좌표 노이즈** — 아주 작은 가우시안 노이즈 (검출 오차를 흉내)

기본 배수는 `--multiplier 6`. 더 늘리려면 `python 14b_augment_keypoints.py --multiplier 12` 처럼 값만 올리면 됨.

## 필수 요건 체크

| 요건 | 대응 |
|---|---|
| YOLO-Pose 17개+ 키포인트 직접 수집 | `13.2D_Pose_Save.py` → YOLO-Pose로 직접 추출 → `keypoints.csv` |
| 앉음·서있음 외 3개 이상 동작 정의 | `facing_forward`, `head_down`, `chin_on_hand` 3종 |
| 파생 특징 1개 이상 추가해 성능 비교 | 박스 기준 정규화 적용 전후 비교 (58.8% → 97.8%) |
| 웹캠·영상에서 동작 라벨 표시 데모 | `16.2D_Pose_ActionRecognition.py` |
