import matplotlib.pyplot as plt
import cv2

# 이미지 읽음
img = cv2.imread("test.png")
# 이미지 크기 변경
im2 = cv2.resize(img, (600, 300))
# 크기 변경한 이미지 저장
cv2.imwrite("out-resize.png", im2)

# 이미지 출력
plt.imshow(cv2.cvtColor(im2, cv2.COLOR_BGR2RGB))
plt.show()
