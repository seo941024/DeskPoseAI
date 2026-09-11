# 웹캠 또는 영상에서 얼굴을 찾아 나이와 성별을 추정해 화면에 표시하는 프로그램
# 얼굴 검출은 OpenCV Haar Cascade, 나이/성별 추정은 WideResNet(IMDB-WIKI 가중치)을 사용함

import cv2  # 영상 입출력, 얼굴 검출, 화면 표시
import os  # 파일 경로 조합
import numpy as np  # 얼굴 이미지 배열과 나이 기댓값 계산
from keras.models import Model  # 입력과 출력을 연결하는 함수형 모델
# WideResNet을 구성하는 층들(합성곱, 배치정규화, 활성화, 완전연결 등)
from keras.layers import Input, Activation, add, Dense, Flatten, Dropout, Conv2D, AveragePooling2D, BatchNormalization
from keras.regularizers import l2  # 가중치 감쇠(과적합 억제)
from keras import backend as K  # 채널 순서(channels_first/last) 확인
import argparse  # 커맨드라인에서 네트워크 깊이/폭 조정
from time import time  # FPS 계산
import tensorflow as tf  # GPU 메모리 설정

#=================================================================
#초기값 설정
#=================================================================
#실행 경로 설정 
# 경로 설정
# 현재 소스 파일 위치를 기준으로 절대 경로를 만든다.
# (어느 위치에서 실행해도 모델/영상 파일을 찾을 수 있도록 하기 위함)
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
Models_dir = os.path.abspath(os.path.join(script_dir, "../00_Models"))
Videos_dir = os.path.abspath(os.path.join(script_dir, "../00_Sample_Video"))

video_file = os.path.join(Videos_dir, "input","face.mp4" )
age_model_file  = os.path.join(Models_dir, "weights.28-3.73.hdf5" )


# NVIDIA GPU 설정
# 기본값은 GPU 메모리를 처음부터 전부 잡는 방식이라, 필요한 만큼만 늘려 쓰도록 바꾼다.
# 다른 GPU 프로그램과 함께 돌릴 때 메모리 부족을 피하기 위함
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)  # 동적 메모리 할당
        print(f"Using GPU: {gpus}")
    except RuntimeError as e:
        print(e)
        
# WideResNet 클래스 정의: 나이 및 성별 예측을 위한 신경망 모델
# 하나의 얼굴 이미지에서 성별과 나이를 동시에 예측하는 네트워크임.
# 출력이 두 갈래로 나뉘며, 성별은 2개(남/여), 나이는 101개(0~100세) 확률을 내놓음.
# depth(깊이)와 k(폭)로 크기를 조절하는 Wide Residual Network 구조임.
class WideResNet:
    def __init__(self, image_size, depth=16, k=8):
        # 네트워크 설정
        self._depth = depth
        self._k = k
        self._dropout_probability = 0  # 드롭아웃 확률 (기본값: 0)
        self._weight_decay = 0.0005
        self._use_bias = False
        self._weight_init = "he_normal"

        # 입력 데이터 형상 설정 (채널 우선 또는 마지막)
        if K.image_data_format() == "channels_first":
            self._channel_axis = 1
            self._input_shape = (3, image_size, image_size)
        else:
            self._channel_axis = -1
            self._input_shape = (image_size, image_size, 3)

    # Residual Block 하나를 만드는 함수를 돌려줌.
    # 입력을 우회해서 더해주는(shortcut) 구조라 층이 깊어져도 학습이 잘 됨.
    def _wide_basic(self, n_input_plane, n_output_plane, stride):
        # Residual Block 정의
        def f(net):
            conv_params = [[3, 3, stride, "same"], [3, 3, (1, 1), "same"]]
            n_bottleneck_plane = n_output_plane

            for i, v in enumerate(conv_params):
                if i == 0:
                    # 첫 번째 Conv2D 블록
                    if n_input_plane != n_output_plane:
                        net = BatchNormalization(axis=self._channel_axis)(net)
                        net = Activation("relu")(net)
                        convs = net
                    else:
                        convs = BatchNormalization(axis=self._channel_axis)(net)
                        convs = Activation("relu")(convs)

                    convs = Conv2D(
                        n_bottleneck_plane, kernel_size=(v[0], v[1]), strides=v[2],
                        padding=v[3], kernel_initializer=self._weight_init,
                        kernel_regularizer=l2(self._weight_decay), use_bias=self._use_bias
                    )(convs)
                else:
                    # 두 번째 Conv2D 블록
                    convs = BatchNormalization(axis=self._channel_axis)(convs)
                    convs = Activation("relu")(convs)
                    if self._dropout_probability > 0:
                        convs = Dropout(self._dropout_probability)(convs)
                    convs = Conv2D(
                        n_bottleneck_plane, kernel_size=(v[0], v[1]), strides=v[2],
                        padding=v[3], kernel_initializer=self._weight_init,
                        kernel_regularizer=l2(self._weight_decay), use_bias=self._use_bias
                    )(convs)

            # Shortcut 연결
            if n_input_plane != n_output_plane:
                shortcut = Conv2D(
                    n_output_plane, kernel_size=(1, 1), strides=stride, padding="same",
                    kernel_initializer=self._weight_init, kernel_regularizer=l2(self._weight_decay),
                    use_bias=self._use_bias
                )(net)
            else:
                shortcut = net

            return add([convs, shortcut])

        return f

    # 같은 Residual Block을 count개 쌓아 하나의 스테이지를 만듦
    def _layer(self, block, n_input_plane, n_output_plane, count, stride):
        # Residual Layer 정의
        def f(net):
            net = block(n_input_plane, n_output_plane, stride)(net)
            for i in range(2, int(count + 1)):
                net = block(n_output_plane, n_output_plane, stride=(1, 1))(net)
            return net

        return f

    # 실제 케라스 모델을 조립해서 돌려줌. WideResNet(...)()처럼 두 번 호출하는 형태임
    def __call__(self):
        # WideResNet 구조 생성
        assert ((self._depth - 4) % 6 == 0)
        # depth는 6의 배수 + 4 여야 함 (16, 22, 28 ...). n은 스테이지당 블록 수
        n = (self._depth - 4) / 6

        inputs = Input(shape=self._input_shape)
        n_stages = [16, 16 * self._k, 32 * self._k, 64 * self._k]

        # Residual 네트워크 생성
        conv1 = Conv2D(
            filters=n_stages[0], kernel_size=(3, 3), strides=(1, 1), padding="same",
            kernel_initializer=self._weight_init, kernel_regularizer=l2(self._weight_decay), use_bias=self._use_bias
        )(inputs)

        block_fn = self._wide_basic
        conv2 = self._layer(block_fn, n_stages[0], n_stages[1], n, (1, 1))(conv1)
        conv3 = self._layer(block_fn, n_stages[1], n_stages[2], n, (2, 2))(conv2)
        conv4 = self._layer(block_fn, n_stages[2], n_stages[3], n, (2, 2))(conv3)

        # 최종 출력
        batch_norm = BatchNormalization(axis=self._channel_axis)(conv4)
        relu = Activation("relu")(batch_norm)
        pool = AveragePooling2D(pool_size=(8, 8), strides=(1, 1), padding="same")(relu)
        flatten = Flatten()(pool)

        # 성별 출력: 2개 클래스(0=여성, 1=남성) 확률
        predictions_g = Dense(
            units=2, kernel_initializer=self._weight_init, use_bias=self._use_bias,
            kernel_regularizer=l2(self._weight_decay), activation="softmax"
        )(flatten)

        # 나이 출력: 0~100세 각각의 확률 101개
        predictions_a = Dense(
            units=101, kernel_initializer=self._weight_init, use_bias=self._use_bias,
            kernel_regularizer=l2(self._weight_decay), activation="softmax"
        )(flatten)

        return Model(inputs=inputs, outputs=[predictions_g, predictions_a])

# FaceCV 클래스 정의: 얼굴 검출 및 예측
# 카메라 영상에서 얼굴을 찾고 나이/성별을 예측해 화면에 그려주는 클래스
class FaceCV:
    WRN_WEIGHTS_PATH = age_model_file

    def __init__(self, depth=16, width=8, face_size=64):
        # WideResNet 모델 초기화
        self.face_size = face_size
        self.model = WideResNet(face_size, depth=depth, k=width)()

        # 가중치 파일 로드
        if not os.path.exists(self.WRN_WEIGHTS_PATH):
            raise FileNotFoundError(f"Weights file not found at {self.WRN_WEIGHTS_PATH}")
        self.model.load_weights(self.WRN_WEIGHTS_PATH)

        # OpenCV 내장 Haar Cascade 얼굴 검출기 초기화 (opencv 패키지에 xml 포함)
        cascade_file = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        self.detector = cv2.CascadeClassifier(cascade_file)
        if self.detector.empty():
            raise FileNotFoundError(f"Haar Cascade file not found at {cascade_file}")

    @staticmethod
    def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, thickness=2):
        # 이미지에 텍스트 레이블 표시
        size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        x, y = point
        cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
        cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)

    # 검출된 얼굴 영역을 여유(margin)를 두고 잘라 모델 입력 크기로 맞춤.
    # margin=40은 얼굴 크기의 40%만큼 사방으로 넓혀서 자른다는 뜻임(머리카락, 턱선 포함).
    def crop_face(self, img, section, margin=40, size=64):
        # 얼굴 영역 크롭 및 크기 조정
        img_h, img_w, _ = img.shape
        # 잘라낸 영역이 이미지 밖으로 나가지 않도록 max/min으로 잘라냄
        x, y, w, h = section
        margin = int(min(w, h) * margin / 100)
        x_a, y_a = max(0, x - margin), max(0, y - margin)
        x_b, y_b = min(img_w, x + w + margin), min(img_h, y + h + margin)

        cropped = img[y_a:y_b, x_a:x_b]
        resized_img = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)
        return resized_img, (x_a, y_a, x_b - x_a, y_b - y_a)

    def detect_face(self):
        # 웹캠에서 얼굴 검출 및 나이, 성별 예측
        
        video_capture = cv2.VideoCapture(0)
        #video_capture = cv2.VideoCapture(video_file)
        if not video_capture.isOpened():
            raise IOError("Cannot open webcam")

        # FPS 계산 변수 초기화
        frame_count = 0
        start_time = time()

        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Failed to grab frame")
                break

            # 얼굴 검출 비용은 픽셀 수에 비례하므로 절반 크기로 줄여서 처리함
            # (1280x720 기준 약 37ms -> 17ms). 이후 좌표와 화면 출력도 이 크기를 따름
            frame = cv2.resize(frame, None, fx=0.5, fy=0.5)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # scaleFactor  : 이미지를 10%씩 줄여가며 여러 크기의 얼굴을 탐색
            # minNeighbors : 클수록 오탐이 줄고 미검출이 늘어남
            # minSize      : 이보다 작은 영역은 얼굴로 보지 않음
            faces = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # 검출된 얼굴들을 한 배열에 모아 한 번에 예측함 (한 장씩 predict하면 느림)
            face_imgs = np.empty((len(faces), self.face_size, self.face_size, 3))
            for i, (x, y, w, h) in enumerate(faces):
                face_img, cropped = self.crop_face(frame, (x, y, w, h))
                face_imgs[i] = face_img
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)

            if len(face_imgs) > 0:
                # predict()는 대용량 배치를 위한 래퍼라 호출마다 준비 비용이 붙음.
                # 매 프레임 얼굴 몇 개만 넣는 용도에는 모델을 직접 호출하는 편이 빠름.
                # 대신 결과가 텐서로 나오므로 numpy로 바꿔서 사용함
                results = self.model(face_imgs, training=False)
                predicted_genders = results[0].numpy()
                # 나이 출력은 0~100세 확률 101개임. 확률에 나이를 곱해 더하면 기댓값(평균 나이)이 됨.
                # 예) 30세 확률 0.6, 31세 확률 0.4 -> 30*0.6 + 31*0.4 = 30.4세
                ages = np.arange(0, 101).reshape(101, 1)
                predicted_ages = results[1].numpy().dot(ages).flatten()

                for i, (x, y, w, h) in enumerate(faces):
                    # predicted_genders[i][0]은 여성일 확률. 0.5를 넘으면 Female로 표시함
                    label = f"{int(predicted_ages[i])}, {'Female' if predicted_genders[i][0] > 0.5 else 'Male'}"
                    self.draw_label(frame, (x, y), label)

            # FPS 계산: 매 프레임 업데이트
            frame_count += 1
            elapsed_time = time() - start_time
            if elapsed_time > 0:
                fps = frame_count / elapsed_time
            else:
                fps = 0

            # FPS 정보 출력
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Age and Gender Detection', frame)

            # 'q' 또는 ESC 키 입력 시 종료
            # waitKey는 5ms 대기하며 키 입력을 받음. 이 호출이 있어야 imshow 창이 갱신됨
            key = cv2.waitKey(5) & 0xFF
            if key == ord('q') or key == 27:
                break

        # 리소스 정리
        video_capture.release()
        cv2.destroyAllWindows()


def get_args():
    parser = argparse.ArgumentParser(
        description="웹캠에서 얼굴을 인식하고 나이와 성별을 추정합니다.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--depth", type=int, default=16, help="WideResNet 네트워크의 깊이")
    parser.add_argument("--width", type=int, default=8, help="WideResNet 네트워크의 폭")
    return parser.parse_args()

def main():
    args = get_args()
    face_cv = FaceCV(depth=args.depth, width=args.width)
    face_cv.detect_face()

if __name__ == "__main__":
    main()
