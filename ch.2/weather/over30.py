import matplotlib.pyplot as plt
import pandas as pd
 
# 파일 읽어 들이기
df = pd.read_csv('tem10y.csv', encoding="utf-8")

# 온도가 30도를 넘는 데이터 확인
hot_bool = (df["기온"] > 30)  #True/False 반환
# 데이터 추출하기 
hot = df[hot_bool] #True 행만 추출해 hot 라는 데이터 만듬. 자료 형식은 DataFrame 

# 연별로 세기
cnt = hot.groupby(["연"])["연"].count()

# 출력
print(cnt)
cnt.plot()
plt.savefig("tem-over30.png")
plt.show()
