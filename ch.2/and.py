from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# 학습 전용 데이터와 결과 세팅
# X , Y
learn_data = [[0,0], [1,0], [0,1], [1,1]]
# X and Y
learn_label = [0, 0, 0, 1]

# 알고리즘 지정 (LinierSVC) scikit-leanr algorithm cheat sheet 참조
clf = LinearSVC()

# 학습전용데이터와결과학습하기  
clf.fit(learn_data, learn_label)

# 예측할 테스트 데이터 입력 
test_data = [[0,0], [1,0], [0,1], [1,1]]
test_label = clf.predict(test_data)

# 예측 결과 평가하기
print(test_data , "의 예측 결과: " ,  test_label)
print("정답률 = " , accuracy_score([0, 0, 0, 1], test_label))
