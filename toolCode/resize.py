import cv2
import os

### 원본 폴더 경로 작성
path = 'C:/turtlebot/line_data/_label_image/'

### 리사이즈 이미지를 저장할 폴더 경로 작성
save = 'C:/turtlebot/line_data/label_image/'

pixel_X=320
pixel_Y=240

img_list=os.listdir(path)
# print(img_list)

for i in range(len(os.listdir(path))):
    img=cv2.imread(path+img_list[i],cv2.IMREAD_COLOR)
    
    # 1. general or big
    # dst = cv2.resize(img, dsize=(pixel_X, pixel_Y), interpolation=cv2.INTER_LINEAR)

    # 2. small
    dst = cv2.resize(img, dsize=(pixel_X, pixel_Y), interpolation=cv2.INTER_AREA)
    cv2.imwrite(save+img_list[i],dst)
