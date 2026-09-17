# 실습3(자세 인식) + 실습4(나이/성별/감정 인식)을 한 화면에 합친 통합 데모.
# - 매 프레임: YOLO-Pose로 박스+키포인트 추출 -> XGBoost로 자세(집중도) 라벨 예측 (빠름, 매 프레임)
# - 별도 스레드: DeepFace로 나이/성별/감정 분석 (느려서 스레드로 돌려야 화면이 안 끊김)
# - 박스 위에는 자세 라벨, 박스 아래에는 나이/성별/감정을 같이 표시함
#
# 사전 조건: 실습3_Pose/01_PoseAction 에서 13~15번을 먼저 실행해서
#            pose_img/person/model_weights.xgb 와 classes.json 이 만들어져 있어야 함.
#
# 사용법:
#   python combined_demo.py --backend opencv

import argparse
import json
import os
import threading
import time
from collections import Counter, deque

import cv2
import pandas as pd
import xgboost as xgb
from ultralytics import YOLO
from deepface import DeepFace

HERE = os.path.dirname(os.path.abspath(__file__))
POSE_DIR = os.path.abspath(os.path.join(HERE, "../실습3_Pose"))
POSE_MODEL_FILE = os.path.join(POSE_DIR, "00_Models", "yolo26n-pose.pt")
POSE_SAVE_DIR = os.path.join(POSE_DIR, "01_PoseAction", "pose_img", "person")
POSE_WEIGHTS = os.path.join(POSE_SAVE_DIR, "model_weights.xgb")
POSE_CLASSES_FILE = os.path.join(POSE_SAVE_DIR, "classes.json")

POSE_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255)]


def get_args():
    ap = argparse.ArgumentParser(description="자세 인식 + 나이/성별/감정 인식 통합 데모")
    ap.add_argument("--backend", default="opencv",
                     choices=["opencv", "mtcnn", "retinaface", "ssd", "dlib", "mediapipe"],
                     help="DeepFace 얼굴 검출 백엔드")
    # DeepFace의 성별 판정은 프레임마다 Man/Woman이 흔들릴 수 있고(특히 헤어스타일 영향을 많이 받음),
    # 모델 판정과 무관하게 화면에 표시할 성별을 고정하고 싶을 때 씀. 나이·감정은 그대로 모델 결과를 보여줌.
    ap.add_argument("--gender-display", default=None, choices=["Man", "Woman"],
                     help="화면에 표시할 성별을 고정 (모델의 실제 판정과 무관하게 표시만 고정)")
    return ap.parse_args()


class EmotionAnalyzer(threading.Thread):
    """DeepFace 분석을 별도 스레드에서 돌려서 메인 루프(화면 표시)가 안 막히게 함."""

    def __init__(self, backend):
        super().__init__(daemon=True)
        self.backend = backend
        self.actions = ['age', 'gender', 'emotion']
        self._lock = threading.Lock()
        self._latest_frame = None
        self.results = []
        self.running = True
        self.busy = False

    def submit(self, frame):
        with self._lock:
            self._latest_frame = frame

    def run(self):
        while self.running:
            with self._lock:
                frame, self._latest_frame = self._latest_frame, None
            if frame is None:
                time.sleep(0.01)
                continue
            self.busy = True
            try:
                results = DeepFace.analyze(frame, actions=self.actions,
                                            detector_backend=self.backend,
                                            enforce_detection=False, silent=True)
                good = [r for r in results if r.get('face_confidence', 1) > 0]
                # 이번엔 얼굴을 못 찾았어도 직전 성공 결과를 화면에서 지우지 않음
                # (매 프레임 검출이 안정적으로 성공하지 않아서, 안 그러면 텍스트가 계속 깜빡이거나 안 보임)
                if good:
                    self.results = good
            except Exception:
                pass
            self.busy = False

    def stop(self):
        self.running = False


def main():
    args = get_args()

    if not os.path.exists(POSE_WEIGHTS):
        print(f"Error: {POSE_WEIGHTS} 가 없습니다.")
        print("먼저 실습3_Pose/01_PoseAction 에서 13 -> 14 -> 14b -> 15 번을 실행해 모델을 학습하세요.")
        return

    with open(POSE_CLASSES_FILE, encoding='utf-8') as fp:
        pose_classes = json.load(fp)

    pose_model = YOLO(POSE_MODEL_FILE)
    action_model = xgb.XGBClassifier()
    action_model.load_model(POSE_WEIGHTS)

    emotion_analyzer = EmotionAnalyzer(args.backend)
    emotion_analyzer.start()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: 카메라를 열 수 없습니다.")
        return

    emotion_counter = Counter()
    last_emotion_seen = None
    frame_times = deque(maxlen=30)
    prev_t = time.time()

    print(f"자세 {len(pose_classes)}종: {pose_classes} / 감정 백엔드: {args.backend}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # ---- 1) 자세 인식 (매 프레임, 빠름) ----
        results = pose_model(frame, verbose=False)
        r = results[0]
        if r.boxes is not None and len(r.boxes) > 0:
            bound_box = r.boxes.xyxy
            conf = r.boxes.conf.tolist()
            keypoints_px = r.keypoints.xy.tolist()

            for i, box in enumerate(bound_box):
                if conf[i] <= 0.75:
                    continue
                x1, y1, x2, y2 = box.tolist()
                box_w, box_h = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)

                data = {}
                for j, (px, py) in enumerate(keypoints_px[i]):
                    if px == 0 and py == 0:
                        data[f'x{j}'] = 0.0
                        data[f'y{j}'] = 0.0
                    else:
                        data[f'x{j}'] = min(max((px - x1) / box_w, 0.0), 1.0)
                        data[f'y{j}'] = min(max((py - y1) / box_h, 0.0), 1.0)

                pred = int(action_model.predict(pd.DataFrame([data]))[0])
                pose_name = pose_classes[pred]
                color = POSE_COLORS[pred % len(POSE_COLORS)]

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label_y = max(int(y1) - 10, 25)
                cv2.putText(frame, pose_name, (int(x1), label_y),
                            cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2)

        # ---- 2) 감정/나이/성별 (뒤 스레드, 느려도 화면은 안 막힘) ----
        if not emotion_analyzer.busy:
            emotion_analyzer.submit(frame.copy())

        for res in emotion_analyzer.results:
            region = res['region']
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            emo = res['dominant_emotion']
            info = f"{res['dominant_gender']}, {res['age']}  {emo}"
            cv2.putText(frame, info, (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

            if emo != last_emotion_seen:
                emotion_counter[emo] += 1
                last_emotion_seen = emo

        if emotion_counter:
            top_emotion, top_count = emotion_counter.most_common(1)[0]
            total = sum(emotion_counter.values())
            cv2.putText(frame, f"dominant emotion so far: {top_emotion} ({top_count}/{total})",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        now = time.time()
        if now > prev_t:
            frame_times.append(now - prev_t)
        prev_t = now
        fps = len(frame_times) / sum(frame_times) if frame_times else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow("Pose + Emotion Combined Demo", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    emotion_analyzer.stop()
    cap.release()
    cv2.destroyAllWindows()

    if emotion_counter:
        total = sum(emotion_counter.values())
        print("\n[감정 집계 요약]")
        for emo, cnt in emotion_counter.most_common():
            print(f"    {emo:<10} {cnt:>4}회  ({cnt/total*100:.1f}%)")


if __name__ == "__main__":
    main()
