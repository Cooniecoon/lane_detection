import cv2
import os

### 원본 폴더 경로 작성
path = 'C:/turtlebot/line_data/data_image/'

### 리사이즈 이미지를 저장할 폴더 경로 작성
save = 'C:/turtlebot/line_data/data_image/'


img_list=os.listdir(path)
# print(img_list)

for i in range(len(os.listdir(path))):
    img=cv2.imread(path+img_list[i],cv2.IMREAD_COLOR)

    flipHorizontal = cv2.flip(img, 1)

    cv2.imwrite(save+'flip_'+img_list[i],flipHorizontal)


###########################################################################################
###########################################################################################

### 원본 폴더 경로 작성
path_ = 'C:/turtlebot/line_data/label_image/'

### 리사이즈 이미지를 저장할 폴더 경로 작성
save_ = 'C:/turtlebot/line_data/label_image/'

img_list=os.listdir(path_)
# print(img_list)

for i in range(len(os.listdir(path_))):
    img=cv2.imread(path_+img_list[i],cv2.IMREAD_COLOR)

    # img90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) # 시계방향으로 90도 회전
    # img180 = cv2.rotate(img, cv2.ROTATE_180) # 180도 회전
    # img270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    flipHorizontal = cv2.flip(img, 1)

    cv2.imwrite(save_+'flip_'+img_list[i],flipHorizontal)
