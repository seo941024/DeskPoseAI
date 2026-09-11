# cv2: 이미지를 다루기 위한 OpenCV 라이브러리
# DeepFace: 나이, 성별, 감정을 한 번에 분석하는 라이브러리
# time: FPS 계산을 위한 시간 라이브러리
import argparse
import csv
import os
import threading
import time
from collections import deque, Counter  # FPS 이동평균 / 감정 집계용

import cv2
from deepface import DeepFace

# ----------------------------------------------------------------
# 커맨드라인 인자
#   --backend opencv   (기본, 설치 불필요, 제일 빠름)
#   --backend mtcnn     (랜드마크 포함, 측면 얼굴에 강함)
#   --backend retinaface (가장 정확, GPU 권장, 느림)
# 같은 웹캠/조건에서 --backend 만 바꿔 두 번 이상 실행해 속도·정확도를 비교한다.
# ----------------------------------------------------------------
def get_args():
    ap = argparse.ArgumentParser(description="DeepFace 실시간 나이/성별/감정 인식 + 감정 집계")
    ap.add_argument("--backend", default="opencv",
                     choices=["opencv", "mtcnn", "retinaface", "ssd", "dlib", "mediapipe"],
                     help="얼굴 검출 백엔드")
    return ap.parse_args()


# ------------------------------------------------------------
# 분석 워커: 별도 스레드에서 계속 DeepFace.analyze 를 돌림.
# 메인 루프(화면 표시)는 이 스레드를 기다리지 않고 항상 최신 프레임을 바로 보여주기 때문에
# 카메라 화면이 끊기지 않음. 분석 결과만 조금 늦게(스레드가 끝나는 대로) 갱신됨.
# ------------------------------------------------------------
class AnalyzerThread(threading.Thread):
    def __init__(self, backend):
        super().__init__(daemon=True)
        self.backend = backend
        self.actions = ['age', 'gender', 'emotion']  # race는 필수 요건에 없어서 빼고 그만큼 가볍게
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
                frame = self._latest_frame
                self._latest_frame = None
            if frame is None:
                time.sleep(0.01)
                continue
            self.busy = True
            try:
                results = DeepFace.analyze(frame, actions=self.actions,
                                            detector_backend=self.backend,
                                            enforce_detection=False, silent=True)
                self.results = [r for r in results if r.get('face_confidence', 1) > 0]
            except Exception:
                self.results = []
            self.busy = False

    def stop(self):
        self.running = False


def main():
    args = get_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    os.makedirs("results", exist_ok=True)
    log_path = os.path.join("results", f"emotion_log_{args.backend}.csv")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("오류: 웹캠을 열 수 없음.")
        exit()

    analyzer = AnalyzerThread(args.backend)
    analyzer.start()

    prev_frame_time = time.time()
    frame_times = deque(maxlen=30)

    # ------------------------------------------------------------
    # 감정 집계 (필수 요건: 감정 결과를 표시만 말고 판단·집계에 활용)
    #   - 분석될 때마다 dominant_emotion 카운트를 누적
    #   - 화면 상단에 "지금까지 우세 감정" 표시, 종료 시 콘솔에 비율 요약
    #   - 전체 로그는 results/emotion_log_<backend>.csv 로 저장 (시각, 감정, 나이, 성별)
    # ------------------------------------------------------------
    emotion_counter = Counter()
    session_start = time.time()
    log_rows = []
    last_emotion_seen = None  # 같은 감정이 연속으로 뜨는 동안 중복 카운트하지 않기 위함

    while True:
        ret, frame = cap.read()
        if not ret:
            print("오류: 프레임을 읽을 수 없음.")
            break

        # 분석 스레드가 지금 노는 중이면 이번 프레임을 넘겨줌 (바쁘면 건너뜀 -> 화면은 안 막힘)
        if not analyzer.busy:
            analyzer.submit(frame.copy())

        last_results = analyzer.results
        for result in last_results:
            region = result['region']
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            labels = [
                f"{result['dominant_gender']}, {result['age']}",
                result['dominant_emotion'],
            ]
            font = cv2.FONT_HERSHEY_SIMPLEX
            for i, text in enumerate(labels):
                text_y = max(20, y - 10 - i * 25)
                cv2.putText(frame, text, (x, text_y), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            emo = result['dominant_emotion']
            if emo != last_emotion_seen:
                emotion_counter[emo] += 1
                log_rows.append({
                    "t_sec": round(time.time() - session_start, 1),
                    "backend": args.backend,
                    "emotion": emo,
                    "age": result['age'],
                    "gender": result['dominant_gender'],
                })
                last_emotion_seen = emo

        if emotion_counter:
            top_emotion, top_count = emotion_counter.most_common(1)[0]
            total = sum(emotion_counter.values())
            summary = f"[{args.backend}] dominant so far: {top_emotion} ({top_count}/{total})"
            cv2.putText(frame, summary, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)

        # FPS 계산 (최근 30프레임 평균으로 안정화) - 이제 분석과 무관하게 항상 카메라 프레임레이트에 가까움
        new_frame_time = time.time()
        elapsed = new_frame_time - prev_frame_time
        prev_frame_time = new_frame_time
        if elapsed > 0:
            frame_times.append(elapsed)
        fps = len(frame_times) / sum(frame_times) if frame_times else 0
        cv2.putText(frame, f"FPS: {fps:.1f}  backend={args.backend}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("Real-time Face Analysis", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    analyzer.stop()
    cap.release()
    cv2.destroyAllWindows()

    if log_rows:
        with open(log_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=["t_sec", "backend", "emotion", "age", "gender"])
            w.writeheader()
            w.writerows(log_rows)
        print(f"\n[저장] {log_path} ({len(log_rows)}건)")

    if emotion_counter:
        total = sum(emotion_counter.values())
        print(f"\n[감정 집계 요약] backend={args.backend}, 총 {total}회 분석")
        for emo, cnt in emotion_counter.most_common():
            print(f"    {emo:<10} {cnt:>4}회  ({cnt/total*100:.1f}%)")


if __name__ == "__main__":
    main()
