# 13.2D_Pose_Save.py 를 자세마다 따로 실행하기 귀찮을 때 쓰는 초간단 버전.
# 창 하나만 띄워두고 숫자키(1~5)로 지금 자세가 뭔지만 알려주면, 그 자세로 계속 자동 저장됨.
# 각 자세를 5~10초씩만 잡아줘도 충분함 (0.15초 간격이라 초당 ~6~7장 모임).
#
# 사용법:
#   python 13b_quick_collect.py
#   1) classes.json 순서대로 화면에 번호가 뜸 -> 숫자키를 눌러 지금 자세 선택
#   2) 그 자세를 취한 채로 몇 초 유지 (자동 저장됨, 화면에 저장 개수 표시)
#   3) 다음 숫자키 눌러서 자세 전환, 반복
#   4) 5개 자세 다 모았으면 q 로 종료

import pandas as pd
import cv2
from ultralytics import YOLO
import os
import time
import json
import glob

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
Models_dir = os.path.abspath(os.path.join(script_dir, "../00_Models"))
model_file = os.path.join(Models_dir, "yolo26n-pose.pt")

save_dir = './pose_img/person/'
keypoint_path = f'{save_dir}keypoints.csv'
classes_file = f'{save_dir}classes.json'

if not os.path.exists(classes_file):
    print(f"Error: {classes_file} 가 없습니다.")
    exit()

with open(classes_file, encoding='utf-8') as fp:
    pose_classes = json.load(fp)

for pose_name in pose_classes:
    os.makedirs(os.path.join(save_dir, pose_name), exist_ok=True)

if len(pose_classes) > 9:
    print("자세가 9개를 넘으면 숫자키 하나로 매핑이 안 됨. classes.json을 줄이세요.")
    exit()

key_to_pose = {ord(str(i + 1)): name for i, name in enumerate(pose_classes)}
print("자세 배정:")
for i, name in enumerate(pose_classes):
    print(f"  [{i + 1}] {name}")
print("숫자키로 자세를 선택하면 그 순간부터 자동 저장 시작. q 로 종료.")

existing = glob.glob(os.path.join(save_dir, '*', '*.jpg'))
a = len(existing)

model = YOLO(model_file)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: 카메라를 열 수 없습니다.")
    exit()

interval_sec = 0.15   # 자세마다 5초만 유지해도 30장 넘게 모임
last_t = 0
current_pose = None
counts = {name: 0 for name in pose_classes}
all_data = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    view = results[0].plot()

    now = time.time()
    if current_pose is not None and (now - last_t) >= interval_sec:
        last_t = now
        for r in results:
            bound_box = r.boxes.xyxy
            conf = r.boxes.conf.tolist()
            keypoints_px = r.keypoints.xy.tolist()
            for index, box in enumerate(bound_box):
                if conf[index] > 0.75:
                    x1, y1, x2, y2 = box.tolist()
                    box_w, box_h = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)
                    pict = frame[int(y1):int(y2), int(x1):int(x2)]
                    image_name = f'person_{a}.jpg'
                    output_path = os.path.join(save_dir, current_pose, image_name)

                    # [파생 특징] 박스 기준 정규화 (화면 전체 기준이 아니라 사람 박스 기준)
                    data = {'image_name': image_name, 'label': current_pose}
                    for j, (px, py) in enumerate(keypoints_px[index]):
                        if px == 0 and py == 0:
                            data[f'x{j}'] = 0.0
                            data[f'y{j}'] = 0.0
                        else:
                            data[f'x{j}'] = min(max((px - x1) / box_w, 0.0), 1.0)
                            data[f'y{j}'] = min(max((py - y1) / box_h, 0.0), 1.0)

                    all_data.append(data)
                    cv2.imwrite(output_path, pict)
                    a += 1
                    counts[current_pose] += 1

    # 상태 표시
    status = f"pose: {current_pose or '(숫자키로 선택)'}"
    cv2.putText(view, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    summary = "  ".join(f"{k}:{v}" for k, v in counts.items())
    cv2.putText(view, summary, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    key_hint = "  ".join(f"[{i+1}]{n}" for i, n in enumerate(pose_classes))
    cv2.putText(view, key_hint, (10, view.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Quick Pose Collect", view)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key in key_to_pose:
        current_pose = key_to_pose[key]
        print(f"-> 자세 전환: {current_pose}")

cap.release()
cv2.destroyAllWindows()

df = pd.DataFrame(all_data)
if len(df) > 0:
    if os.path.exists(keypoint_path):
        df.to_csv(keypoint_path, mode='a', header=False, index=False)
    else:
        df.to_csv(keypoint_path, index=False)

print("\n[수집 완료]")
for k, v in counts.items():
    print(f"  {k:<15} {v}장")
print(f"Saved: {keypoint_path}")
