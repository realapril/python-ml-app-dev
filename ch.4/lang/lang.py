#언어 판정

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Unicode 코드 포인트로 출현 빈도 판정
def count_codePoint(str):
    # Unicode 코드 포인트를 저장할 배열 준비
    counter = np.zeros(65535) # 매서드 요소 수를 지정하면 모든 요소가 0인 배열을 반환한다.

    for i in range(len(str)):
        # 각 문자를 Unicode 코드 포인트로 변환
        code_point = ord(str[i])
        if code_point > 65535 :
            continue
        # 출현 횟수 세기
        counter[code_point] += 1

    # 각 요소를 문자 수로 나눠 정규화
    counter = counter/len(str)
    return counter

# 학습 전용 데이터 준비하기
ko_str = '이것은 한국어 문장입니다.'
ja_str = 'これは日本語の文章です。'
en_str = 'This is English Sentences.'

x_train = [count_codePoint(ko_str),count_codePoint(ja_str),count_codePoint(en_str)]
y_train = ['ko','ja','en']

# 학습
clf = GaussianNB() 
clf.fit(x_train, y_train)

# 평가 전용 데이터 준비하기
ko_test_str = '안녕하세요'
ja_test_str = 'こんにちは'
en_test_str = 'Hello'

x_test = [count_codePoint(en_test_str),count_codePoint(ja_test_str),count_codePoint(ko_test_str)]
y_test = ['en', 'ja', 'ko']

# 평가하기 
y_pred = clf.predict(x_test)
print(y_pred)
print("정답률 = " , accuracy_score(y_test, y_pred))  #0.66666666666