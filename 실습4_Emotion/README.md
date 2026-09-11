# 실습 4 — DeepFace 나이·성별·감정 인식

원본: `Emotion_Detection.zip` (`Emotion_Detection.py`). `Age_Gender_Recognition.py`는 DeepFace가 아닌
별도의 WideResNet 방식이라 슬라이드 필수 요건(DeepFace 사용)과 안 맞음 — **사용 안 함**.

## 원본 대비 바뀐 점 (필수 요건 채우기)

| 필수 요건 | 원본 상태 | 수정 |
|---|---|---|
| 나이·성별·감정 3가지 추출 | ✅ 이미 있었음 | 그대로 |
| 감정 결과를 판단·집계에 활용 | ❌ 화면 표시만 함 | `Counter`로 감정 누적 집계 → 화면에 "지금까지 우세 감정" 표시, 종료 시 콘솔에 백분율 요약, `results/emotion_log_<backend>.csv`로 저장 |
| 검출 백엔드 2개 이상 비교 | ❌ opencv 고정 | `--backend opencv/mtcnn/retinaface/...` 인자 추가. `compare_backends.py`로 결과 비교 |
| 웹캠·영상 실시간 데모 | ✅ | 그대로 |

## 설치

```bash
conda activate vision_new
pip install deepface opencv-contrib-python pandas
# retinaface 백엔드를 쓰려면 추가로:
pip install retina-face
# mtcnn 백엔드를 쓰려면 추가로:
pip install mtcnn
```

## 실행

```bash
# 1) 기본 백엔드(opencv, 제일 빠름)로 실행 — q 로 종료하면 콘솔에 감정 집계 요약이 뜸
python Emotion_Detection.py --backend opencv

# 2) 같은 조건으로 다른 백엔드 실행 (비교용)
python Emotion_Detection.py --backend mtcnn
python Emotion_Detection.py --backend retinaface

# 3) 두 번 이상 실행한 로그를 모아서 백엔드별 비교표 생성
python compare_backends.py
```

각 실행이 끝나면:
- 화면에 실시간 FPS + 사용 중인 backend 표시 (속도 비교 근거)
- 콘솔에 감정별 횟수·비율 요약 (예: `happy 12회 (40.0%)`)
- `results/emotion_log_<backend>.csv` 저장 (시각, 감정, 나이, 성별)

`compare_backends.py`는 이 로그들을 모아 **백엔드별 검출 횟수 / 감정 분포 / 평균 추정 나이**를 비교해 출력하고 `results/backend_compare.csv`로 저장함 — "같은 조건에서 정확도와 속도를 비교"한 근거 자료로 그대로 쓸 수 있다.

## 참고 — 백엔드 특성 (슬라이드 8)

| 백엔드 | 속도 | 정확도 | 비고 |
|---|---|---|---|
| opencv | 가장 빠름 | 정면은 무난, 측면은 약함 | 설치 불필요, 기본값 |
| mtcnn | 느린 편 | 정면·측면 모두 높음 | 랜드마크 포함 |
| retinaface | 보통~느림 | 작은 얼굴까지 가장 정확 | GPU 권장 |

## 다음에 할 수 있는 것 (확장, 슬라이드 10)

- 3~5프레임마다 한 번만 분석해 FPS 더 확보 (`--interval` 값 조정)
- 얼굴 임베딩으로 같은 사람 중복 집계 방지
- `results/emotion_log_*.csv`를 시간축 그래프로 시각화 (강의 몰입도 리포트 등)
- 개인정보: 얼굴 원본 이미지는 저장하지 않고 집계값만 남기는 지금 방식 유지
