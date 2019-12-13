
import tensorflow as tf
import os


class jyDataSet:
    def __init__(self):
        strTrainFile = 'TrainData//Train//ImageTrainTrain.tfrecord'
        strTestFile = 'TrainData//Test//ImageTrainTest.tfrecord'
        RootPath = os.path.abspath(os.path.dirname(os.getcwd()))
        # positive(Normal) Image Path
        NormalPath = RootPath + '/TrainDataSet/FastTrainFrame/Normal'
        # negative(ANormal) Image Path
        ANormalPath = RootPath + '/TrainDataSet/FastTrainFrame/ANormal'
        # 获取训练数据集长度
        listTrainName = os.listdir(NormalPath)
        # 由于是8 : 2分割
        iLenAll = len(listTrainName)
        self.__iLenTrain = int(iLenAll / 10) * 8
        iLenTest = iLenAll - self.__iLenTrain
        iLenANormalTest = len(os.listdir(ANormalPath))
        self.__iLenTest = iLenTest + iLenANormalTest

        # 训练数据集
        self.__iRepeatTimes = 1
        self.__iShuffleBuf = self.__iLenTrain * self.__iRepeatTimes
        self.__iSeed = 10
        self.pTrainTFDS = tf.data.TFRecordDataset(strTrainFile). \
            shuffle(buffer_size=self.__iShuffleBuf, seed=self.__iSeed). \
            map(self.trainMapFun)
        # 转换成numpy格式
        self.pTrainTFDS = self.pTrainTFDS.take(self.__iLenTrain).batch(self.__iLenTrain)
        for tensor in self.pTrainTFDS:
            self.pTrainTFDS = tensor['train_image_raw'].numpy()

        # 测试数据集
        self.pTestTFDS = tf.data.TFRecordDataset(strTestFile). \
            shuffle(buffer_size=self.__iShuffleBuf, seed=self.__iSeed). \
            map(self.testMapFun)
        # Another
        strTestFile2 = 'TrainData//Test//ImageTrainTest2.tfrecord'
        self.pAnotherTest = tf.data.TFRecordDataset(strTestFile2). \
            shuffle(buffer_size=self.__iShuffleBuf, seed=self.__iSeed). \
            map(self.testMapFun)
        self.pAnotherTest = self.pAnotherTest.take(985).batch(985)
        for tensor in self.pAnotherTest:
            self.pAnotherTFDS = tensor['test_image_raw'].numpy()
            self.pAnotherLabel = tensor['label'].numpy()
        # 转换成numpy格式
        self.pTestTFDS = self.pTestTFDS.take(self.__iLenTest).batch(self.__iLenTest)
        for tensor in self.pTestTFDS:
            self.pTestTFDS = tensor['test_image_raw'].numpy()
            self.pTestLabel = tensor['label'].numpy()

    def trainMapFun(self, exampleProto):
        self.__iRepeatTimes = 1
        features = {
            'train_image_raw': tf.io.FixedLenFeature([], tf.string)
        }
        parsedFeatures = tf.io.parse_single_example(exampleProto, features)
        tensorIMG = tf.image.decode_jpeg(parsedFeatures['train_image_raw'])
        tensorIMG = tf.cast(tensorIMG, tf.float32)
        tensorIMG /= 255.0
        parsedFeatures['train_image_raw'] = tensorIMG
        return parsedFeatures

    def testMapFun(self, exampleProto):
        self.__iRepeatTimes = 1
        features = {
            'test_image_raw': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64)
        }
        parsedFeatures = tf.io.parse_single_example(exampleProto, features)
        tensorIMG = tf.image.decode_jpeg(parsedFeatures['test_image_raw'])
        tensorIMG = tf.cast(tensorIMG, tf.float32)
        tensorIMG /= 255.0
        parsedFeatures['test_image_raw'] = tensorIMG
        return parsedFeatures
