
import cv2
import os
import matplotlib.pyplot as plt


RootFileName = 'HED'
listVideoName = ['MachineBelt2']
RootPath = os.path.abspath(os.path.dirname(os.getcwd()))

if not os.path.exists(RootPath + '/' + RootFileName):
    os.mkdir(RootPath + '/' + RootFileName)

j = 0
for video in listVideoName:
    pVideo = cv2.VideoCapture(RootPath + "//TrainDataSet//Machine//%s.mp4" % video)
    if not os.path.exists(RootPath + '//TrainDataSet//%s//%s' % (RootFileName, video)):
        os.mkdir(RootPath + '//TrainDataSet//%s//%s' % (RootFileName, video))
        os.mkdir(RootPath + '//TrainDataSet//%s//%s//Train' % (RootFileName, video))
        os.mkdir(RootPath + '//TrainDataSet//%s//%s//Test' % (RootFileName, video))
    else:
        continue
    i = 0
    iNum = 0
    imgPath = ""
    # hog = cv2.HOGDescriptor()
    strStatus = 'Train'
    while True:
        ret, frame = pVideo.read()
        if not ret:
            break
        '''
        frameTopLeft = frame[0: 1120, 0: 1120]
        frameTopRight = frame[0: 1120, 1120 + 160: 2400]#1120 - 320: 1921]
        frameTopLeft = cv2.resize(frameTopLeft, (224, 224))
        frameTopRight = cv2.resize(frameTopRight, (224, 224))
        '''
        # frameBottomLeft = frame[1120 - 320: 1920, 0: 1120]
        # frameBottomRight = frame[1120 - 320: 1920, 1120 + 320: 2561]
        # frame = frame[200: 1480, 0:2560]
        # frame = cv2.resize(frame, (256, 128))
        #frame = frame[800: 1440, 850: 1490]
        #frame = cv2.resize(frame, (64, 64))
        frame = frame[0: 1480, 340: 1820]
        frame = cv2.resize(frame, (224, 224))
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # frame = cv2.GaussianBlur(frame, (5, 5), 0)
        # frame = cv2.Canny(frame, 100, 200)
        # plt.imshow(frame)
        # plt.show()
        # 帧率相关
        # if i % 25 == 0:
        if i % 720 == 0:
            imgPath = RootPath + "/TrainDataSet/%s/%s/%s/%s.jpg" % (RootFileName, video, strStatus, str(j + 318))
            # imgPath1 = RootPath + "/TrainDataSet/%s/%s/%s/%s.jpg" % (RootFileName, video, strStatus, str(2 * j))
            # imgPath2 = RootPath + "/TrainDataSet/%s/%s/%s/%s.jpg" % (RootFileName, video, strStatus, str(2 * j + 1))
            # imgPath3 = RootPath + "/TrainDataSet/%s/%s/%s/%s.jpg" % (RootFileName, video, strStatus, str(j+2))
            # imgPath4 = RootPath + "/TrainDataSet/%s/%s/%s/%s.jpg" % (RootFileName, video, strStatus, str(j+3))
            cv2.imwrite(imgPath, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            # cv2.imwrite(imgPath1, frameTopLeft, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            # cv2.imwrite(imgPath2, frameTopRight, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            # cv2.imwrite(imgPath3, frameBottomLeft, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            # cv2.imwrite(imgPath4, frameBottomRight, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            j += 1
            iNum += 1
        if iNum >= 1799:
            strStatus = 'Test'
        i += 1
