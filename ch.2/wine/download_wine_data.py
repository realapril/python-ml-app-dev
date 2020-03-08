# 화이트와인 데이터를 csv로 저장한다.
from urllib.request import urlretrieve
url = "https://archive.ics.uci.edu" + \
      "/ml/machine-learning-databases/wine-quality" + \
      "/winequality-white.csv"
savepath = "winequality-white.csv"
urlretrieve(url, savepath)
print("saved")

# 읽어온 내용 불러온다
csv = pd.read_csv(savepath, encoding="utf-8")
# csv 내용 화면에 출력
csv


