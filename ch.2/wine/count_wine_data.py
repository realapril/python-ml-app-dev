import matplotlib.pyplot as plt
import pandas as pd

# 와인 데이터 읽어 들이기
# sep=";"는 ; 기준으로 파싱하겠다고 지정한것
wine = pd.read_csv("winequality-white.csv", sep=";", encoding="utf-8")

# 품질 데이터별로 그룹을 나누고 수 세어보기
count_data = wine.groupby('quality')["quality"].count()
print(count_data)

# 수를 그래프로 그리기
count_data.plot()
# 출력 그래프 저장
plt.savefig("wine-count-plt.png")
plt.show()

# 불균형 데이터 : 데이터 수의 분포 차가 큰 데이터
