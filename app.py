import cv2
import numpy as np
import random
import mediapipe as mp

# 读取图片
def readImage():
    img = cv2.imread('iron man.jpg')
    img = cv2.resize(img, (300, 300))
    cv2.imshow('img', img)
    cv2.waitKey(0)

# 调用函数
# readVideo()

# 读取视频下一帧
def readVideoNextFrame():
    cap = cv2.VideoCapture('sunrise.mp4')
    ret, nextFrame = cap.read()
    if ret:
        cv2.imshow('video', nextFrame)
    cv2.waitKeyEx(0)

# 读取视频
def readVideo():
    cap = cv2.VideoCapture('sunrise.mp4')
    while True:
        flag, nextFrame = cap.read()
        if flag:
            cv2.imshow('video', nextFrame)
        else:
            break
        # 按键q退出
        if cv2.waitKeyEx(10) == ord('q'):
            break

# 读取摄像头
def readCamera():
    cap = cv2.VideoCapture(0)
    while True:
        flag, nextFrame = cap.read()
        if flag:
            cv2.imshow('video', nextFrame)
        else:
            break
        # 按键q退出
        if cv2.waitKeyEx(10) == ord('q'):
            break

# 图片格式
def imageType():
    img = cv2.imread('iron man.jpg')
    print(img.shape)
    print(img)

def creatImage():
    img = np.empty((300, 300, 3), np.uint8)
    for row in range(300):
        for col in range(300):
            img[row][col] = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
    cv2.imshow('img', img)
    cv2.waitKey(0)

# 切割图片
def splitImage():
    image = cv2.imread('iron man.jpg')
    newImage = image[: 100,: 100]
    cv2.imshow('image', image)
    cv2.imshow('newImage', newImage)
    cv2.waitKey(0)

# 改变图片大小、灰阶
def resizeImage():
    img = cv2.imread('iron man.jpg')
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    # 灰阶
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 高斯模糊
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    # 边缘
    canny = cv2.Canny(img, 150, 200)
    # 膨胀边缘
    kernel = np.ones((3, 3), np.uint8)
    dilate = cv2.dilate(canny, kernel, iterations=3)
    erode = cv2.erode(dilate, kernel, iterations=1)
    cv2.imshow('img', img)
    cv2.imshow('gray', gray)
    cv2.imshow('blur', blur)
    cv2.imshow('canny', canny)
    cv2.imshow('dilate', dilate)
    cv2.imshow('erode', erode)
    cv2.waitKey(0)

# 画图
def paint():
    img = np.zeros((600, 600, 3), np.uint8)
    cv2.line(img, (0, 0), (400, 300), (255, 0, 0), 1)
    cv2.imshow('img', img)
    cv2.waitKey(0)

# 侦测颜色
def readColor():
    img = cv2.imread('iron man.jpg')
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # lower = np.array(10, 10, 10)
    # upper = np.array(11, 11, 10)
    # mask = cv2.inRange(hsv, lower, upper)
    # result = cv2.bitwise_and(img, img, mask = mask)
    cv2.imshow('img', img)
    cv2.imshow('hsv', hsv)
    # cv2.imshow('result', result)
    cv2.waitKey(0)

# 轮廓检测
def readShap():
    img = cv2.imread('iron man.jpg')
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    imgFarke = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(img, 150, 200)
    contours, hierarchy =  cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        # 画出轮廓
        cv2.drawContours(imgFarke, cnt, -1, (255, 0, 0), 4)
    cv2.imshow('img', img)
    cv2.imshow('canny', canny)
    cv2.imshow('imgFarke', imgFarke)
    cv2.waitKey(0)

# 人脸识别
def readFace():
    img = cv2.imread('iron man.jpg')
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalcatface.xml')
    # 图片，缩小倍数，侦测到的次数
    faceRect = faceCascade.detectMultiScale(gray, 1.1, 3)
    print(len(faceRect))
    for (x, y, w, h) in faceRect:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow('img', img)
    cv2.waitKey(0)

# 手部追踪
def detectHand():
    cap = cv2.VideoCapture(0)
    # 使用手部模型
    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils
    while True:
        ret, img = cap.read()
        if ret:
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(imgRGB)
            # print(result.multi_hand_landmarks)
            if result.multi_hand_landmarks:
                for handLms in result.multi_hand_landmarks:
                    # 划线
                    mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
                    for i, lm in enumerate(handLms.landmark):
                        # 打印手部坐标
                        print(i, lm.x, lm.y)
            cv2.imshow('img', img)
        if cv2.waitKey(1) == ord('q'):
            break

detectHand()