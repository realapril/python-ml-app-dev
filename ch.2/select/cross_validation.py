# 목표 1: 최적의 알고리즘이 맞는지 검증
# 목표 2: 데이터 분류 편향이 발생하지 않았는지 확인*

# * Cross Validation: 학습 데이터가 적은 경우 평가의 신뢰성을 올리는 방법. 
#                   K분할 밸리데이션: 데이터를 k 개 그룹으로 분할하고, k-1 개를 학습 전용, 남은 1개를 평가 전용으로 사용해 평가 -> k번 반복

import pandas as pd
from sklearn.utils.testing import all_estimators
from sklearn.model_selection import KFold
import warnings
from sklearn.model_selection import cross_val_score

# 붓꽃 데이터 읽어 들이기
iris_data = pd.read_csv("iris.csv", encoding="utf-8")

# 붓꽃 데이터를 레이블과 입력 데이터로 분리하기 
y = iris_data.loc[:,"Name"]
x = iris_data.loc[:,["SepalLength","SepalWidth","PetalLength","PetalWidth"]]

# classifier 알고리즘 모두 추출하기
warnings.filterwarnings('ignore')
allAlgorithms = all_estimators(type_filter="classifier")

# K-분할 크로스 밸리데이션 전용 객체 
kfold_cv = KFold(n_splits=5, shuffle=True) #5분할 validation

for(name, algorithm) in allAlgorithms:
    # 각 알고리즘 객체 생성하기
    clf = algorithm()

    # score 메서드를 가진 클래스를 대상
    if hasattr(clf,"score"):
        
        # 크로스 밸리데이션
        scores = cross_val_score(clf, x, y, cv=kfold_cv)
        print(name,"의 정답률=")
        print(scores)


# AdaBoostClassifier 의 정답률=
# [0.96666667 0.93333333 0.93333333 0.83333333 0.93333333]
# BaggingClassifier 의 정답률=
# [0.96666667 0.96666667 0.96666667 0.93333333 0.96666667]
# BernoulliNB 의 정답률=
# [0.23333333 0.26666667 0.26666667 0.3        0.3       ]
# CalibratedClassifierCV 의 정답률=
# [0.96666667 0.8        0.83333333 0.96666667 0.86666667]
# CategoricalNB 의 정답률=
# [0.86666667 1.         0.86666667 0.96666667 0.93333333]        