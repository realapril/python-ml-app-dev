# 목표 1: 최적의 알고리즘이 맞는지 검증*
# 목표 2: 데이터 분류 편향이 발생하지 않았는지 확인

# *classifier 계열 모든 알고리즘으로 테스트 해본다.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
from sklearn.utils.testing import all_estimators

import warnings
warnings.filterwarnings('ignore')

# 붓꽃 데이터 읽음
iris_data = pd.read_csv("iris.csv", encoding="utf-8")

# 붓꽃 데이터를 레이블과 입력 데이터로 분리하기 
y = iris_data.loc[:,"Name"]
x = iris_data.loc[:,["SepalLength","SepalWidth","PetalLength","PetalWidth"]]

# 학습 전용과 테스트 전용 분리하기 
warnings.filterwarnings('ignore')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, train_size = 0.8, shuffle = True)

# classifier 계열 모든 알고리즘으로 추출
warnings.filterwarnings('ignore')
allAlgorithms = all_estimators(type_filter="classifier")

for(name, algorithm) in allAlgorithms:
    # 각 알고리즘 객체 생성. 옵션 생성도 여기서 가능
    clf = algorithm()

    # 학습하고 평가
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(name,"의 정답률 = " , accuracy_score(y_test, y_pred))



# AdaBoostClassifier 의 정답률 =  0.9
# BaggingClassifier 의 정답률 =  0.9
# BernoulliNB 의 정답률 =  0.3
# CalibratedClassifierCV 의 정답률 =  0.8
# CategoricalNB 의 정답률 =  0.8666666666666667    