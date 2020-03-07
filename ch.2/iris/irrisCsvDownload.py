import urllib.request as req
import pandas as pd

#직접 url에 방문해 csv를 컴퓨터에 저장해도 된다.

#파일다운로드
url = "https://raw.githubusercontent.com/pandas-dev/pandas/master/pandas/tests/data/iris.csv"
savefile = "iris.csv"
req.urlretrieve(url, savefile)
print("saved")

#다운한 파일 읽어옴
csv = pd.read_csv(savefile, encoding="utf-8")
#csv 내용 화면에 출력
csv