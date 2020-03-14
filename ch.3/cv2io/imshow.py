# 다운로드한 이미지 출력하기
import matplotlib.pyplot as plt
import cv2

img = cv2.imread("test.png") # download_imread.py 에서 png로 저장했으니 png로 불러야함

plt.axis("off") # 꼭 plt imshow 보다 상단에서 호출해야 적용됨
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

