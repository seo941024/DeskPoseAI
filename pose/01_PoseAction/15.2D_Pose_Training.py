import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time  # 시간 측정을 위한 모듈
import os
import json

# 실행 경로 설정 (상대 경로가 소스 폴더 기준으로 풀리도록 함)
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

save_dir = './pose_img/person/'
# 14b_augment_keypoints.py 를 돌렸으면 증강된 dataset_aug.csv 를 우선 사용,
# 없으면 원본 dataset.csv 를 그대로 사용함
_aug_file = f'{save_dir}dataset_aug.csv'
dataset_file = _aug_file if os.path.exists(_aug_file) else f'{save_dir}dataset.csv'
weight_file = f'{save_dir}model_weights.xgb'
classes_file = f'{save_dir}classes.json'
print(f"[학습 데이터] {dataset_file}")

# 데이터셋 로드
df = pd.read_csv(dataset_file)

# 자세 목록 로드
# classes.json이 기준이며 직접 관리하는 파일임. 클래스 번호 순서도 이 파일이 결정함.
if not os.path.exists(classes_file):
    print(f"Error: {classes_file} 가 없습니다. 자세 목록을 먼저 작성하세요.")
    exit()

with open(classes_file, encoding='utf-8') as fp:
    pose_classes = json.load(fp)
label_to_id = {name: i for i, name in enumerate(pose_classes)}

# 목록에 없는 라벨은 학습에서 제외
unknown = sorted(set(df['label']) - set(pose_classes))
if unknown:
    print(f"Warning: classes.json에 없는 라벨을 제외합니다 - {unknown}")
    df = df[df['label'].isin(pose_classes)]

# 목록에 있는데 데이터가 하나도 없으면 클래스 번호가 어긋나므로 중단
empty_poses = [name for name in pose_classes if name not in set(df['label'])]
if empty_poses:
    print(f"Error: 학습 데이터가 없는 자세가 있습니다 - {empty_poses}")
    exit()

# 특징(X)과 타겟(y) 변수 정의
X = df.drop(['label', 'image_name'], axis=1)  # 'label' 컬럼이 타겟 변수라고 가정
y = df['label'].map(label_to_id)  # 레이블을 0부터 시작하는 번호로 변환

print(f"자세 {len(pose_classes)}종: {pose_classes}")
print(df['label'].value_counts().to_string())

# 데이터를 학습 세트와 테스트 세트로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# XGBoost 분류기 생성 (자세 2종이면 이진 분류, 3종 이상이면 다중 분류)
if len(pose_classes) == 2:
    objective, eval_metric = 'binary:logistic', 'logloss'
else:
    objective, eval_metric = 'multi:softmax', 'mlogloss'
model = xgb.XGBClassifier(objective=objective, eval_metric=eval_metric)

# 학습 시작 시간 기록
start_time = time.time()
print("Training started...")

# 모델 학습
# 학습 과정을 보려면 eval_set을 넘겨야 함 (verbose는 기본값이 True라 eval_set만 주면 매 라운드가 출력됨)
model.fit(X_train, y_train,
          eval_set=[(X_train, y_train), (X_test, y_test)],
          verbose=True)

# 학습 종료 시간 기록
end_time = time.time()
print("Training finished...")

# 총 소요시간 계산
total_time = end_time - start_time
print(f"Total training time: {total_time:.2f} seconds")

# 테스트 세트에 대해 예측 수행
y_pred = model.predict(X_test)

# 정확도 평가
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# 학습된 모델 저장
model.save_model(weight_file)

print(f"Saved: {weight_file}")
