# DeepFace 나이·성별·감정 인식

원본: `Emotion_Detection.zip` (`Emotion_Detection.py`). `Age_Gender_Recognition.py`는 DeepFace가 아닌
별도의 WideResNet 방식이라 DeepFace 기반 구성과 안 맞음 — **메인 기능에 사용 안 함**.

## 원본 대비 바뀐 점

| 기능 | 원본 상태 | 수정 |
|---|---|---|
| 나이·성별·감정 3가지 추출 | ✅ 이미 있었음 | 그대로 |
| 감정 결과를 판단·집계에 활용 | ❌ 화면 표시만 함 | `Counter`로 감정 누적 집계 → 화면에 "지금까지 우세 감정" 표시, 종료 시 콘솔에 백분율 요약, `results/emotion_log_<backend>.csv`로 저장 |
| 검출 백엔드 2개 이상 비교 | ❌ opencv 고정 | `--backend opencv/mtcnn/retinaface/...` 인자 추가. `compare_backends.py`로 결과 비교 |
| 웹캠·영상 실시간 데모 | ✅ | 그대로 |

## 설치

```bash
conda activate vision_new
pip install deepface tf-keras opencv-contrib-python pandas
# retinaface 백엔드를 쓰려면 추가로:
pip install retina-face
# mtcnn 백엔드를 쓰려면 추가로:
pip install mtcnn
```

- `tf-keras`: TensorFlow 2.16 이상에서 DeepFace가 요구하는 패키지. 설치된 TensorFlow와 같은 버전대로 설치, 예를 들어 TensorFlow 2.21이면 `tf-keras==2.21.*`
- deepface가 함께 설치하는 `opencv-contrib-python-headless`가 `cv2`를 덮어쓰면 `cv2.imshow`에서 `The function is not implemented` 오류 발생. 이때 `pip uninstall -y opencv-contrib-python-headless` 후 `pip install --force-reinstall --no-deps opencv-contrib-python`으로 화면 기능이 있는 판 재설치

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
- 감정 집계는 분석 1회당 1건, 한 화면에 얼굴이 여럿이면 가장 큰 얼굴 1개 기준. 요약 첫 줄의 `총 N회 분석`은 실제 DeepFace 분석 횟수, `얼굴 검출 M회`는 그중 얼굴을 찾은 횟수
- `results/emotion_log_<backend>.csv` 저장 (시각, 감정, 나이, 성별)

`compare_backends.py`는 이 로그들을 모아 **백엔드별 검출 횟수 / 감정 분포 / 평균 추정 나이**를 비교해 출력하고 `results/backend_compare.csv`로 저장함 — "같은 조건에서 정확도와 속도를 비교"한 근거 자료로 그대로 쓸 수 있다.

## 참고 — 백엔드 특성

| 백엔드 | 속도 | 정확도 | 비고 |
|---|---|---|---|
| opencv | 가장 빠름 | 정면은 무난, 측면은 약함 | 설치 불필요, 기본값 |
| mtcnn | 느린 편 | 정면·측면 모두 높음 | 랜드마크 포함 |
| retinaface | 보통~느림 | 작은 얼굴까지 가장 정확 | GPU 권장 |

## 다음에 할 수 있는 것

- 3~5프레임마다 한 번만 분석해 FPS 더 확보 (`--interval` 값 조정)
- 얼굴 임베딩으로 같은 사람 중복 집계 방지
- `results/emotion_log_*.csv`를 시간축 그래프로 시각화 (학습 몰입도 리포트 등)
- 개인정보: 얼굴 원본 이미지는 저장하지 않고 집계값만 남기는 지금 방식 유지
