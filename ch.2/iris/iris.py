import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 붓꽃 데이터 읽기
iris_data = pd.read_csv("iris.csv", encoding="utf-8")

# 붓꽃 데이터를 레이블과 입력 데이터로 분리
y = iris_data.loc[:,"Name"]
x = iris_data.loc[:,["SepalLength","SepalWidth","PetalLength","PetalWidth"]]

# 붓꽃 데이터를 학습 전용과 테스트 전용 분리
# 20%는 테스트할 데이터, 80%는 학습용 데이터
# shuffle는 학습 편향 막기 위해서, True 가 디폴트이기 때문에 꼭 선언 안해도됨.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, train_size = 0.8, shuffle = True)

# 학습기 생성
clf = SVC()
# 학습 시작
clf.fit(x_train, y_train)

# 평가
y_pred = clf.predict(x_test)
print("정답률 = " , accuracy_score(y_test, y_pred))  #0.9666666666666667