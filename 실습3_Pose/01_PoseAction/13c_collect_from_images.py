# 웹캠 대신, 미리 받아둔 자세별 이미지 폴더(../raw_images/<자세>/*.jpg)에
# YOLO-Pose를 돌려서 키포인트를 뽑는 버전. 13.2D_Pose_Save.py와 결과 포맷(csv, 크롭 이미지)이 동일해서
# 그대로 14/15/16번 스크립트로 이어진다.
#
# 이미지 출처: Pexels (무료 스톡사진, 저작권 표시 불필요) - 자세별 검색 결과 상위 20장
#
# [파생 특징] 키포인트를 원본 사진 전체 기준(xyn)이 아니라 "사람 박스 기준"으로 정규화해서 저장함.
#   같은 자세라도 사진마다 사람이 화면 어디에 얼마나 크게 찍혔는지 다 다른데, xyn 그대로 쓰면
#   모델이 "자세"가 아니라 "사진 속 위치·크기"를 학습해버림 (실제로 이렇게 했다가 val accuracy 0.59에서 정체됨).
#   박스 기준 정규화: x' = (px - box_x1) / box_width, y' = (py - box_y1) / box_height
#   -> 카메라 거리/위치와 무관하게 값이 일정해져서 순수하게 "자세"만 학습하게 됨 (슬라이드 4의 추천 파생 특징).
#
# 사용법:
#   python 13c_collect_from_images.py
#   (실행할 때마다 keypoints.csv 를 새로 씀 - 여러 번 실행해도 중복 누적되지 않음)

import glob
import json
import os

import cv2
import pandas as pd
from ultralytics import YOLO

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
Models_dir = os.path.abspath(os.path.join(script_dir, "../00_Models"))
model_file = os.path.join(Models_dir, "yolo26n-pose.pt")

raw_dir = os.path.abspath(os.path.join(script_dir, "../raw_images"))
save_dir = "./pose_img/person/"
keypoint_path = f"{save_dir}keypoints.csv"
classes_file = f"{save_dir}classes.json"

if not os.path.exists(classes_file):
    print(f"Error: {classes_file} 가 없습니다.")
    exit()

with open(classes_file, encoding="utf-8") as fp:
    pose_classes = json.load(fp)

for pose_name in pose_classes:
    os.makedirs(os.path.join(save_dir, pose_name), exist_ok=True)

model = YOLO(model_file)

all_data = []
a = 0
skipped_no_person = 0

for pose_name in pose_classes:
    src_dir = os.path.join(raw_dir, pose_name)
    if not os.path.isdir(src_dir):
        print(f"[건너뜀] {src_dir} 없음 (raw_images 에 '{pose_name}' 폴더를 만들고 이미지를 넣으세요)")
        continue

    files = sorted(glob.glob(os.path.join(src_dir, "*.jpg")) +
                    glob.glob(os.path.join(src_dir, "*.jpeg")) +
                    glob.glob(os.path.join(src_dir, "*.png")))
    saved_for_pose = 0
    skipped_for_pose = 0

    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue

        results = model(img, verbose=False)
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            skipped_for_pose += 1
            continue

        bound_box = r.boxes.xyxy
        conf = r.boxes.conf.tolist()
        keypoints_px = r.keypoints.xy.tolist()  # 픽셀 좌표 (박스 기준 정규화를 직접 계산하기 위해)

        # 이미지 1장에 사람이 여럿이면, 박스가 가장 큰(=주된 인물일 가능성이 큰) 사람 하나만 사용
        # (스톡사진은 여러 명이 섞여 나오는 경우가 많아 라벨과 무관한 사람이 섞이는 걸 줄이기 위함)
        areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in bound_box.tolist()]
        best = max(range(len(areas)), key=lambda i: areas[i]) if areas else None
        if best is None or conf[best] < 0.5:
            skipped_for_pose += 1
            continue

        x1, y1, x2, y2 = bound_box[best].tolist()
        box_w, box_h = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)
        pict = img[int(y1):int(y2), int(x1):int(x2)]
        image_name = f"person_{a}.jpg"
        output_path = os.path.join(save_dir, pose_name, image_name)

        data = {"image_name": image_name, "label": pose_name}
        for j, (px, py) in enumerate(keypoints_px[best]):
            # (0,0)은 미검출 관절이라 그대로 0으로 둠. 검출된 점만 박스 기준으로 정규화.
            if px == 0 and py == 0:
                data[f"x{j}"] = 0.0
                data[f"y{j}"] = 0.0
            else:
                data[f"x{j}"] = min(max((px - x1) / box_w, 0.0), 1.0)
                data[f"y{j}"] = min(max((py - y1) / box_h, 0.0), 1.0)

        all_data.append(data)
        cv2.imwrite(output_path, pict)
        a += 1
        saved_for_pose += 1

    skipped_no_person += skipped_for_pose
    print(f"{pose_name:<15} {saved_for_pose}/{len(files)}장 저장 (미검출/저신뢰 {skipped_for_pose}장)")

df = pd.DataFrame(all_data)
if len(df) > 0:
    # 매번 전체 클래스를 새로 훑으므로 append 하지 않고 덮어씀 (안 그러면 재실행할 때마다 중복 누적됨)
    df.to_csv(keypoint_path, index=False)

print(f"\n[완료] 총 {a}장 저장, 사람 미검출/저신뢰로 건너뜀 {skipped_no_person}장")
print(f"Saved: {keypoint_path}")
