import matplotlib.pyplot as plt
import pandas as pd

# CSV 파일 읽기
df = pd.read_csv("tem10y.csv", encoding="utf-8")

# 월별 평균 구하기
g = df.groupby(['월'])['기온']
gg = g.sum() / g.count()

# 결과 출력
print(gg)
gg.plot()
plt.savefig("tem-month-avg.png")
plt.show()

