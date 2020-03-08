import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# 데이터 읽어 들이기
# sep=";"는 ; 기준으로 파싱하겠다고 지정한것
wine = pd.read_csv("winequality-white.csv", sep=";", encoding="utf-8")

# 데이터를 레이블과 데이터로 분리하기
y = wine["quality"]
x = wine.drop("quality", axis=1) #퀄리티 칼럼을 드랍하고 나머지 전체 

# 학습 전용과 테스트 전용으로 분리하기
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# 학습하기
# 랜덤포레스트는 앙상블 학습이다.
# 앙상블 학습:머신러닝에서 여러개의 분류 트리 모델을 학습시켜 그 모델들의 예측결과들을 기반으로 다수결로 투표를 해 결과를 낸다.
#          하나의 분류 트리 모델은 정답율이 낮지만 앙상블 학습을 거치면 정답율이 높아진다.
model = RandomForestClassifier() 
model.fit(x_train, y_train)

# 평가하기
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred)) #결과: UndefinedMetricWarning: Precision and F-score are ill-defined
print("정답률=", accuracy_score(y_test, y_pred)) #데이터 랜덤하게 학습/테스트용으로 분류하는데 이에 따라 정답율 0.61~0.67사이 나옴
