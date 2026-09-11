import pandas as pd
import os
import json

# 실행 경로 설정 (상대 경로가 소스 폴더 기준으로 풀리도록 함)
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# 저장 경로 설정
save_dir = './pose_img/person/'
keypoint_path = f'{save_dir}keypoints.csv'  # 확장자 수정
dataset_dir = f'{save_dir}/'

# 자세 목록 로드
# classes.json이 기준이며 직접 관리하는 파일임. 하위 폴더 이름은 목록의 이름과 같아야 함.
classes_file = f'{save_dir}classes.json'

if not os.path.exists(classes_file):
    print(f"Error: {classes_file} 가 없습니다. 자세 목록을 먼저 작성하세요.")
    exit()

with open(classes_file, encoding='utf-8') as fp:
    pose_classes = json.load(fp)

# 이미지 이름 -> 자세 이름 조회표 생성 (목록에 없는 폴더는 무시됨)
image_to_label = {}
empty_poses = []
for pose_name in pose_classes:
    pose_path = os.path.join(dataset_dir, pose_name)
    if not os.path.isdir(pose_path):
        empty_poses.append(pose_name)
        print(f"  {pose_name}: 폴더 없음")
        continue

    image_names = os.listdir(pose_path)
    if not image_names:
        empty_poses.append(pose_name)
    for image_name in image_names:
        image_to_label[image_name] = pose_name
    print(f"  {pose_name}: {len(image_names)}장")

if empty_poses:
    print(f"Warning: 이미지가 없는 자세가 있습니다 - {empty_poses}")

# CSV 파일 로드
df = pd.read_csv(keypoint_path)

# CSV 파일에 'label' 열 추가 (조회표에 없는 이미지는 NaN이 됨)
df['label'] = df['image_name'].map(image_to_label)

# 레이블이 없는 데이터 필터링 (어느 자세 폴더에도 없는 이미지 제거)
df = df.dropna(subset=['label'])

# 새로운 CSV 파일로 저장
df.to_csv(f'{dataset_dir}dataset.csv', index=False)
print(f"Saved: {dataset_dir}dataset.csv ({len(df)}행, 자세 {df['label'].nunique()}종)")
