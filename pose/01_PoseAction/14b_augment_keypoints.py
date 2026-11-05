# 14.2D_Pose_Dataset_Save.py 이후 실행.
# dataset.csv (라벨 붙은 원본 키포인트)를 증강해서 dataset_aug.csv 를 만든다.
# 이미지가 아니라 좌표(x0..x16, y0..y16)만 다루므로 이미지를 더 모을 필요가 없음.
#
# 증강 방법 (원본 1개 -> 여러 개로 복제):
#   1) 좌우 반전  : x' = 1 - x, 좌/우 관절 쌍을 서로 바꿈 (거울에 비친 자세도 같은 라벨이므로 유효)
#   2) 위치·크기 지터 : 중심 기준으로 살짝 확대/축소 + 이동 (카메라 거리·위치 차이를 흉내)
#   3) 좌표 노이즈   : 관절 검출 오차를 흉내 낸 아주 작은 가우시안 노이즈
#
# 사용법:
#   python 14b_augment_keypoints.py --multiplier 6   (기본 6배 -> 자세당 20장이면 약 120장씩)
import argparse
import os

import numpy as np
import pandas as pd

# YOLO-Pose(COCO 17 keypoints) 좌우 관절 쌍. 반전 시 서로 교환해야 함
FLIP_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
save_dir = "./pose_img/person/"
dataset_path = f"{save_dir}dataset.csv"
out_path = f"{save_dir}dataset_aug.csv"


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--multiplier", type=int, default=6,
                     help="원본 1개당 만들 총 샘플 수 (원본 포함)")
    ap.add_argument("--jitter-std", type=float, default=0.015,
                     help="좌표 노이즈 표준편차 (정규화 좌표 기준)")
    ap.add_argument("--scale-range", type=float, default=0.12,
                     help="확대/축소 범위 (+-12%)")
    ap.add_argument("--shift-range", type=float, default=0.05,
                     help="중심 이동 범위 (+-5%)")
    return ap.parse_args()


def flip_row(xs, ys):
    xs = xs.copy()
    ys = ys.copy()
    xs = 1.0 - xs
    for a, b in FLIP_PAIRS:
        xs[a], xs[b] = xs[b], xs[a]
        ys[a], ys[b] = ys[b], ys[a]
    return xs, ys


def jitter_row(xs, ys, rng, jitter_std, scale_range, shift_range):
    valid = (xs > 0) | (ys > 0)  # 검출 안 된(0,0) 관절은 노이즈 대상에서 제외
    cx, cy = xs[valid].mean() if valid.any() else 0.5, ys[valid].mean() if valid.any() else 0.5

    scale = 1.0 + rng.uniform(-scale_range, scale_range)
    shift_x = rng.uniform(-shift_range, shift_range)
    shift_y = rng.uniform(-shift_range, shift_range)

    xs2 = xs.copy()
    ys2 = ys.copy()
    xs2[valid] = np.clip((xs[valid] - cx) * scale + cx + shift_x + rng.normal(0, jitter_std, valid.sum()), 0, 1)
    ys2[valid] = np.clip((ys[valid] - cy) * scale + cy + shift_y + rng.normal(0, jitter_std, valid.sum()), 0, 1)
    return xs2, ys2


def main():
    args = get_args()
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} 가 없습니다. 14.2D_Pose_Dataset_Save.py 를 먼저 실행하세요.")
        return

    df = pd.read_csv(dataset_path)
    x_cols = [f"x{i}" for i in range(17)]
    y_cols = [f"y{i}" for i in range(17)]
    rng = np.random.default_rng(42)

    rows = []
    for _, row in df.iterrows():
        xs = row[x_cols].to_numpy(dtype=float)
        ys = row[y_cols].to_numpy(dtype=float)
        label = row["label"]

        variants = [(xs, ys, "orig")]
        xs_f, ys_f = flip_row(xs, ys)
        variants.append((xs_f, ys_f, "flip"))

        rows.append(_to_row(row["image_name"], label, xs, ys, "orig"))
        rows.append(_to_row(row["image_name"], label, xs_f, ys_f, "flip"))

        # 목표 배수를 채울 때까지 orig/flip 을 번갈아 지터링해서 추가
        n_needed = max(0, args.multiplier - 2)
        for i in range(n_needed):
            base_xs, base_ys, tag = variants[i % 2]
            jx, jy = jitter_row(base_xs, base_ys, rng, args.jitter_std, args.scale_range, args.shift_range)
            rows.append(_to_row(row["image_name"], label, jx, jy, f"{tag}_jit{i}"))

    aug_df = pd.DataFrame(rows)
    aug_df.to_csv(out_path, index=False)

    print(f"[증강 완료] 원본 {len(df)}행 -> 증강 후 {len(aug_df)}행")
    print(aug_df["label"].value_counts().to_string())
    print(f"Saved: {out_path}")


def _to_row(image_name, label, xs, ys, tag):
    d = {"image_name": f"{image_name}#{tag}", "label": label}
    for i in range(17):
        d[f"x{i}"] = xs[i]
        d[f"y{i}"] = ys[i]
    return d


if __name__ == "__main__":
    main()
