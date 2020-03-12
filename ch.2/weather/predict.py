from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#기온예측프로그램. 6일간의 최근 과거 데이터를 입력하면 다음날의 기온을 예측한다.

# 기온 데이터 읽어 들이기
df = pd.read_csv('tem10y.csv', encoding="utf-8")

# 데이터를 학습 전용과 테스트 전용으로 분리
train_year = (df["연"] <= 2015)
test_year = (df["연"] >= 2016)
interval = 6

# 과거 6일의 데이터를 기반으로 학습할 데이터 만듬
def make_data(data):
    x = [] # 학습 데이터(6일간의 데이터)
    y = [] # 결과(7일째 데이터)
    temps = list(data["기온"])
    for i in range(len(temps)):
        if i < interval: continue
        y.append(temps[i])
        xa = []
        for p in range(interval):
            d = i + p - interval
            xa.append(temps[d])
        x.append(xa)
    return (x, y)

train_x, train_y = make_data(df[train_year]) #학습전용 리스트
test_x, test_y = make_data(df[test_year]) #테스트 전용 리스트

# 직선 회귀 분석
lr = LinearRegression(normalize=True)
lr.fit(train_x, train_y) # 학습하기
pre_y = lr.predict(test_x) # 예측하기

# 결과를 그래프로 그리기
plt.figure(figsize=(10, 6), dpi=100)
plt.plot(test_y, c='r')
plt.plot(pre_y, c='b')
plt.savefig('tenki-kion-lr.png')
plt.show()



# pre_y - test_y

diff_y = abs(pre_y - test_y)
print("average=", sum(diff_y)/ len(diff_y)) #평균 1.664 의 오차
print("max=", max(diff_y)) #최대 8.47도 오차
