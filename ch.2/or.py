#from sklearn.svm import LinearSVC #KNeighborsClassifier로 변경함
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 학습 전용 데이터와 결과 세팅
# X , Y
learn_data = [[0,0], [1,0], [0,1], [1,1]]
# X xor Y
learn_label = [0, 1, 1, 0]  

# 알고리즘 지정하기(LinierSVC, KNeighborsClassifier)
# clf = LinearSVC() # 정답율이 아주 낮음. 변경한다.
clf = KNeighborsClassifier(n_neighbors = 1) #테스트 세트 안의 변수가 2개이므로 가까운 neighbor은 1개다.

# 학습전용데이터와결과학습하기 
clf.fit(learn_data, learn_label)

# 테스트 데이터로 예측하기
test_data = [[0,0], [1,0], [0,1], [1,1]]
test_label = clf.predict(test_data)

# 테스트 결과 평가하기
print(test_data , "의 예측 결과: " ,  test_label)
print("정답률 = " , accuracy_score([0, 1, 1, 0], test_label)) 