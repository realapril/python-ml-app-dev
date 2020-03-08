from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# 학습 전용 데이터와 결과 세팅
# X , Y
# 특성: 입력 변수입니다(단순 선형 회귀의 x 변수)
learn_data = [[0,0], [1,0], [0,1], [1,1]]
# X and Y
# 라벨: 예측하는 항목(단순 선형 회귀의 y 변수).
learn_label = [0, 0, 0, 1]

# 알고리즘 지정 (LinierSVC) scikit-leanr algorithm cheat sheet 참조
clf = LinearSVC()

# 학습전용데이터로 결과 학습하기  
# 학습 <-> 추론
# 학습은 모델을 만들거나 배우는 것을 의미한다. 즉 라벨이 있는 예를 모델에 보여 주고, 모델이 특성과 라벨의 관계를 점차적으로 학습한다.
# 추론은 학습된 모델을 라벨이 없는 예에 적용하는 것을 의미한다. 즉 학습된 모델을 사용하여 유용한 예측(y')을 한다.
# 앞부분 예제는 모두 라벨이 있기때문에 학습을 시키는것이다.
clf.fit(learn_data, learn_label)

# 예측할 테스트 데이터 입력 
test_data = [[0,0], [1,0], [0,1], [1,1]]
test_label = clf.predict(test_data)

# 예측 결과 평가하기
# 회귀<->분류. 이 예제는 분류이다
# 회귀 모델은 연속적인 값을 예측. 예시: 캘리포니아의 주택 가격
# 분류 모델은 불연속적인 값을 예측. 예시: 이미지가 강아지, 고양이 또는 햄스터의 이미지인가요?
print(test_data , "의 예측 결과: " ,  test_label)
print("정답률 = " , accuracy_score([0, 0, 0, 1], test_label))
