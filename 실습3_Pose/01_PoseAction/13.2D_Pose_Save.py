# pandas는 ultralytics보다 먼저 임포트해야 함 (ultralytics가 먼저 올라오면 pyarrow 임포트가 WinError 6714로 실패함)
import pandas as pd
import cv2
from ultralytics import YOLO
import os
import sys
import time
import json
import glob

#=================================================================
#초기값 설정
#=================================================================
#실행 경로 설정
# 경로 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
Models_dir = os.path.abspath(os.path.join(script_dir, "../00_Models"))
model_file = os.path.join(Models_dir, "yolo26n-pose.pt" )

save_dir = './pose_img/person/'
keypoint_path = f'{save_dir}keypoints.csv'
classes_file = f'{save_dir}classes.json'

# classes.json에 적힌 자세 목록대로 분류용 폴더를 미리 만들어 둠
# (classes.json은 자동 생성하지 않고 직접 관리하는 파일임)
if not os.path.exists(classes_file):
    print(f"Error: {classes_file} 가 없습니다. 자세 목록을 먼저 작성하세요.")
    exit()

with open(classes_file, encoding='utf-8') as fp:
    pose_classes = json.load(fp)

for pose_name in pose_classes:
    os.makedirs(os.path.join(save_dir, pose_name), exist_ok=True)

# -----------------------------------------------------------------
# 이번 실행에서 수집할 자세를 커맨드라인 인자로 받음
#   python 13.2D_Pose_Save.py standing
# 자세별로 따로따로 실행해서 데이터를 모은다 (한 번에 한 자세만 촬영).
# -----------------------------------------------------------------
if len(sys.argv) < 2 or sys.argv[1] not in pose_classes:
    print(f"사용법: python 13.2D_Pose_Save.py <자세이름>")
    print(f"classes.json에 등록된 자세: {pose_classes}")
    exit()

target_pose = sys.argv[1]
pose_dir = os.path.join(save_dir, target_pose)
print(f"자세 {len(pose_classes)}종 폴더 준비 완료: {pose_classes}")
print(f"[수집 대상] '{target_pose}' -> {pose_dir}")

# 이미지 파일명이 자세/실행 회차와 상관없이 전체적으로 겹치지 않도록,
# 지금까지 모든 자세 폴더에 저장된 이미지 수를 세어 다음 번호부터 이어서 저장함
existing = glob.glob(os.path.join(save_dir, '*', '*.jpg'))
start_idx = len(existing)

# YOLOv8 포즈 추정 모델 불러오기
model = YOLO(model_file)

# 카메라 또는 비디오 캡처 객체 생성 (0은 기본 웹캠)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: 카메라를 열 수 없습니다.")
    exit()

# 수집 간격(초)과 수집 목표 장수 설정 (필요에 따라 변경 가능)
interval_sec = 0.5
frame_total = 600
last_t = 0
i = 0
a = start_idx

# 이번 실행에서 모은 데이터만 저장할 리스트 (기존 csv에는 append)
all_data = []

# 비디오가 열려 있는 동안 프레임 처리
while (cap.isOpened()):

    # 현재 프레임 읽기
    flag, frame = cap.read()

    # 더 이상 프레임이 없으면 종료
    if flag == False:
        break

    # 수집 목표 장수를 채우면 종료
    if i >= frame_total:
        break

    # 이번 프레임을 수집할 차례인지 판단 (미리보기는 수집 여부와 무관하게 매 프레임 표시)
    now = time.time()
    collect = (now - last_t) >= interval_sec

    # 미리보기가 끊겨 보이지 않도록 추론은 매 프레임 수행함 (저장만 간격에 맞춰 함)
    results = model(frame, verbose=False)

    # bbox와 포즈 키포인트가 그려진 프레임을 미리보기용으로 사용 (저장되는 이미지는 원본 frame에서 잘라냄)
    view = results[0].plot()

    if collect:
        last_t = now

        # 감지된 결과에 대해 반복 처리
        for r in results:
            bound_box = r.boxes.xyxy  # 프레임에서 경계 상자 좌표 가져오기
            conf = r.boxes.conf.tolist()  # 해당 객체가 사람일 확률 가져오기
            keypoints_px = r.keypoints.xy.tolist()  # 픽셀 좌표 (박스 기준 정규화를 직접 계산)

            # 1개의 이미지에서 감지된 모든 사람 이미지를 저장하는 코드
            # 만약 1개의 이미지에 10명의 사람이 있다면, 10개의 사람 이미지를 저장

            for index, box in enumerate(bound_box):
                # 사람일 확률(conf)이 0.75 이상인 경우만 처리 (흐릿한 이미지 제외)
                if conf[index] > 0.75:
                    x1, y1, x2, y2 = box.tolist()
                    box_w, box_h = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)
                    pict = frame[int(y1):int(y2), int(x1):int(x2)]
                    image_name = f'person_{a}.jpg'
                    output_path = os.path.join(pose_dir, image_name)

                    # CSV 파일에 저장할 이미지 파일 이름 + 라벨(자세) 을 바로 기록
                    data = {'image_name': image_name, 'label': target_pose}

                    # [파생 특징] 화면 전체 기준이 아니라 사람 박스 기준으로 정규화.
                    # 카메라 거리·프레이밍이 달라도 값이 일정해서 순수하게 자세만 학습하게 됨.
                    for j, (px, py) in enumerate(keypoints_px[index]):
                        if px == 0 and py == 0:
                            data[f'x{j}'] = 0.0
                            data[f'y{j}'] = 0.0
                        else:
                            data[f'x{j}'] = min(max((px - x1) / box_w, 0.0), 1.0)
                            data[f'y{j}'] = min(max((py - y1) / box_h, 0.0), 1.0)

                    # YOLO 모델이 감지한 사람의 키포인트를 나중에 기계 학습 모델에 학습시키기 위해 CSV 파일에 저장
                    all_data.append(data)
                    cv2.imwrite(output_path, pict)
                    a += 1

        # 프레임 카운터 증가
        i += 1

    # 진행 상황을 화면 좌측 상단에 표시
    cv2.putText(view, f"[{target_pose}] collect {i}/{frame_total}  saved {a - start_idx}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # 미리보기 창 출력
    cv2.imshow("2D Pose Save", view)

    # 'q' 키를 누르면 목표 장수를 다 못 채워도 중간에 종료
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 처리한 총 프레임 수와 저장한 총 이미지 수 출력
print(f"'{target_pose}' 자세 수집 완료: 프레임 {i}개 처리, 이미지 {a - start_idx}장 저장")
cap.release()
cv2.destroyAllWindows()

# 이번에 모은 데이터를 DataFrame으로 변환
df = pd.DataFrame(all_data)

# 기존 csv가 있으면 이어붙이고(append), 없으면 새로 만듦
# (자세별로 스크립트를 여러 번 실행해도 이전 자세 데이터가 지워지지 않음)
if os.path.exists(keypoint_path):
    df.to_csv(keypoint_path, mode='a', header=False, index=False)
else:
    df.to_csv(keypoint_path, index=False)

print(f"Saved: {keypoint_path} (총 누적 {a}장)")
