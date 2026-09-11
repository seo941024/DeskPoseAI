# pandas는 ultralytics보다 먼저 임포트해야 함 (ultralytics가 먼저 올라오면 pyarrow 임포트가 WinError 6714로 실패함)
import pandas as pd
from ultralytics import YOLO
import cv2
import xgboost as xgb
import os
import json

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
weight_file = f'{save_dir}model_weights.xgb'
classes_file = f'{save_dir}classes.json'

# YOLOv8 모델 로드
model_yolo = YOLO(model_file)

# XGBoost 모델 로드 (Booster 대신 XGBClassifier 사용)
action_model = xgb.XGBClassifier()
action_model.load_model(weight_file)

# 학습에 사용한 자세 목록 로드 (클래스 번호 -> 자세 이름)
if not os.path.exists(classes_file):
    print(f"Error: {classes_file} 가 없습니다. 15번 학습 스크립트를 먼저 실행하세요.")
    exit()

with open(classes_file, encoding='utf-8') as fp:
    pose_classes = json.load(fp)
print(f"자세 {len(pose_classes)}종: {pose_classes}")

# 자세별 표시 색상(BGR). 자세가 색상 수보다 많아지면 앞에서부터 순환해서 사용함
POSE_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
               (0, 255, 255), (255, 0, 255), (255, 255, 0)]

# 카메라에서 비디오 캡처 시작
cap = cv2.VideoCapture(0)

# 총 프레임 수 출력 (일부 카메라에서는 0일 수 있음)
print('Total Frame', cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 비디오 속성 가져오기
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

frame_tot = 0  # 처리된 프레임 수

# 비디오 프레임 반복 처리
while cap.isOpened():
    # 비디오 프레임 읽기
    success, frame = cap.read()

    if success:
        # YOLOv8로 프레임에서 객체 감지 수행
        results = model_yolo(frame, verbose=False)

        # 결과를 시각화하여 프레임에 그리기
        annotated_frame = results[0].plot(boxes=False)

        for r in results:
            bound_box = r.boxes.xyxy  # 경계 상자 정보
            conf = r.boxes.conf.tolist()  # 신뢰도 값
            keypoints_px = r.keypoints.xy.tolist()  # 픽셀 좌표 (학습 때와 동일하게 박스 기준으로 정규화하기 위함)

            for index, box in enumerate(bound_box):
                if conf[index] > 0.75:  # 신뢰도 0.75 이상일 때만 처리
                    x1, y1, x2, y2 = box.tolist()
                    box_w, box_h = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)
                    data = {}

                    # [중요] 13.2D_Pose_Save.py 에서 학습 데이터를 만들 때와 똑같이
                    # 화면 전체 기준이 아니라 "이 사람 박스" 기준으로 정규화해야 함.
                    # 여기가 학습 때 방식과 다르면 예측이 안 맞거나 항상 같은 라벨만 나옴.
                    for j, (px, py) in enumerate(keypoints_px[index]):
                        if px == 0 and py == 0:
                            data[f'x{j}'] = 0.0
                            data[f'y{j}'] = 0.0
                        else:
                            data[f'x{j}'] = min(max((px - x1) / box_w, 0.0), 1.0)
                            data[f'y{j}'] = min(max((py - y1) / box_h, 0.0), 1.0)

                    # 데이터프레임 생성 (DMatrix 대신 사용)
                    df = pd.DataFrame([data])

                    # XGBoost 모델을 이용해 예측 수행 (DMatrix 없이)
                    # 결과는 클래스 번호이므로 자세 이름과 색상으로 바꿔서 표시함
                    pred = int(action_model.predict(df)[0])
                    pose_name = pose_classes[pred]
                    color = POSE_COLORS[pred % len(POSE_COLORS)]

                    cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    # 라벨이 박스 위쪽 끝에 딱 붙어 그려지면 얼굴이 화면 상단에 가까울 때 잘려서 안 보임.
                    # 화면 안쪽(최소 y=25)으로 위치를 보정해서 항상 보이게 함.
                    label_y = max(int(y1) - 10, 25)
                    cv2.putText(annotated_frame, pose_name.capitalize(), (int(x1), label_y),
                                cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2)

        # 주석이 달린 프레임을 화면에 표시 (창 이름 필수)
        cv2.imshow("Pose Action Recognition", annotated_frame)

        frame_tot += 1  # 처리된 프레임 수 증가
        # print('Processed Frame : ', frame_tot)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # 비디오 끝에 도달하면 종료
        break

# 비디오 캡처 객체 및 창 닫기
cap.release()
cv2.destroyAllWindows()
