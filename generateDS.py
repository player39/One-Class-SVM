
import tensorflow as tf
from tensorflow.python.keras.preprocessing import image
import os
import numpy as np
import matplotlib.pyplot as plt

# 项目中放置TFRecord文件的路径
IMGPath = os.getcwd()
# 上级目录
RootPath = os.path.abspath(os.path.dirname(os.getcwd()))
# positive(Normal) Image Path
NormalPath = RootPath + '/TrainDataSet/FastTrainFrame/Normal'
# negative(ANormal) Image Path
ANormalPath = RootPath + '/TrainDataSet/FastTrainFrame/ANormal'


def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def generateDS(imageString, strDSType):
    tensorIMG = tf.image.decode_jpeg(imageString)
    pIMG = tf.transpose(tensorIMG, perm=[0, 1, 2])
     #tensorIMG.transpose(perm=[1, 0, 2])
    # pIMG = image.img_to_array(pIMG)
    # tensorIMG.swapaxes(1, 0)
    pIMG = image.array_to_img(pIMG).resize((256, 144))
    #pIMG = image.img_to_array(pIMG)
    # pIMG = tf.transpose(pIMG, perm=[1, 0, 2])
    '''
    plt.figure(num='Input')
    plt.subplot(1, 1, 1)
    plt.imshow(pIMG)
    plt.show(pIMG)
    '''
    tensorIMG = image.img_to_array(pIMG)
    pIMG = tf.image.encode_jpeg(tensorIMG)
    feature = {
        'train_image_raw': _bytes_feature(pIMG)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def generateTestDS(imageString, strDSType, iLabel):
    tensorIMG = tf.image.decode_jpeg(imageString)
    pIMG = image.array_to_img(tensorIMG).resize((155, 235))
    tensorIMG = image.img_to_array(pIMG)
    pIMG = tf.image.encode_jpeg(tensorIMG)
    feature = {
        strDSType: _bytes_feature(pIMG),
        'label': _int64_feature(iLabel)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def generateTestDSOnly():
    recordTestPath = IMGPath + '/TrainData/Test/ImageTrainTest2.tfrecord'
    listNormalIMG = os.listdir(NormalPath)
    with tf.io.TFRecordWriter(recordTestPath) as writer:
        #创建测试集TFRecord, 由于是OC-SVM因此将异常数据一并加入
        #recordPathTest = IMGPath + '/TFRecord/ImageTest.tfrecord'
        listANormalIMG = os.listdir(ANormalPath)
        #先将剩余的正常数据放入TFRecord
        i = 0
        while i < len(listNormalIMG):
            imagePathNormal = NormalPath + '/' + listNormalIMG[i]
            imgNormal = open(imagePathNormal, 'rb').read()
            tfExample = generateTestDS(imgNormal, 'test_image_raw', 1)
            writer.write(tfExample.SerializeToString())
            i += 1

        #取异常数据放入TFRecord
        for strImageName in listANormalIMG:
            imagePathANormal = ANormalPath + '/' + strImageName
            imgANormal = open(imagePathANormal, 'rb').read()
            tfExample = generateTestDS(imgANormal, 'test_image_raw', -1)
            writer.write(tfExample.SerializeToString())


def generateTrainAndTestDS():
    recordTrainPath = IMGPath + '/TrainData/Train/MyMiniTrain.tfrecord'
    with tf.io.TFRecordWriter(recordTrainPath) as writer:
        listNormalIMG = os.listdir(NormalPath)
        # 按8 : 2 分割 2成测试集
        iLen = len(listNormalIMG) if len(listNormalIMG) > 1250 else 1250
        iTrain = int(iLen / 10) * 8
        iTest = iLen - iTrain
        #创建训练集TFRecord

        for i in range(iTrain):
            imagePathNormal = NormalPath + '/' + listNormalIMG[i]
            imgNormal = open(imagePathNormal, 'rb').read()
            tfExample = generateDS(imgNormal, 'train_image_raw')
            writer.write(tfExample.SerializeToString())
        iBreakPoint = i
    '''
    recordTestPath = IMGPath + '/TrainData/Test/ImageTrainTest23.tfrecord'
    with tf.io.TFRecordWriter(recordTestPath) as writer:
        #创建测试集TFRecord, 由于是OC-SVM因此将异常数据一并加入
        #recordPathTest = IMGPath + '/TFRecord/ImageTest.tfrecord'
        listANormalIMG = os.listdir(ANormalPath)
        #先将剩余的正常数据放入TFRecord
        while iBreakPoint < len(listNormalIMG):
            imagePathNormal = NormalPath + '/' + listNormalIMG[iBreakPoint]
            imgNormal = open(imagePathNormal, 'rb').read()
            tfExample = generateTestDS(imgNormal, 'test_image_raw', 1)
            writer.write(tfExample.SerializeToString())
            iBreakPoint += 1

        #取异常数据放入TFRecord
        for strImageName in listANormalIMG:
            imagePathANormal = ANormalPath + '/' + strImageName
            imgANormal = open(imagePathANormal, 'rb').read()
            tfExample = generateTestDS(imgANormal, 'test_image_raw', -1)
            writer.write(tfExample.SerializeToString())
    '''

generateTrainAndTestDS()
