# 두 번 이상 실행해서 모은 results/emotion_log_<backend>.csv 를 모아
# 백엔드별 평균 FPS는 콘솔 출력(각 실행 로그 참고)이고, 여기서는
# "같은 조건에서 감정 분포·검출 횟수가 백엔드마다 어떻게 다른가"를 비교한다.
#
# 사용법:
#   python Emotion_Detection.py --backend opencv
#   python Emotion_Detection.py --backend retinaface
#   python compare_backends.py
import glob
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
files = sorted(glob.glob(os.path.join(HERE, "results", "emotion_log_*.csv")))

if len(files) < 2:
    raise SystemExit(f"비교하려면 백엔드 2개 이상의 로그가 필요함. 현재 {len(files)}개: {files}")

dfs = [pd.read_csv(f) for f in files]
df = pd.concat(dfs, ignore_index=True)

print("=== 백엔드별 검출 횟수 ===")
print(df.groupby("backend").size().to_string())

print("\n=== 백엔드별 감정 분포(%) ===")
pct = (df.groupby(["backend", "emotion"]).size()
         .groupby(level=0).apply(lambda s: (s / s.sum() * 100).round(1)))
print(pct.to_string())

print("\n=== 백엔드별 평균 추정 나이 ===")
print(df.groupby("backend")["age"].mean().round(1).to_string())

out = os.path.join(HERE, "results", "backend_compare.csv")
df.to_csv(out, index=False, encoding="utf-8-sig")
print(f"\n[저장] {out}")
