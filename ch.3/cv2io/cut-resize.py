import matplotlib.pyplot as plt
import cv2

# 이미지 읽어 들이기
img = cv2.imread("test.png")
# 이미지 자르기
im2 = img[150:450, 150:450]
# 이미지 크기 변경하기
im2 = cv2.resize(im2, (400, 400)) # 튜플(tuple)은 셀수있는 수량의 순서있는 열거. 리스트와 비슷하나 한번 만들면 순서를 바꿀 수 없다.
# 크기 변경한 이미지 저장하기
cv2.imwrite("cut-resize.png", im2)

# 이미지 출력하기
plt.imshow(cv2.cvtColor(im2, cv2.COLOR_BGR2RGB))
plt.show()


