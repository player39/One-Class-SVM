
from sklearn.svm.classes import OneClassSVM
from sklearn.decomposition.pca import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Model
from TwoLayers import jyModel
from DataSet import jyDataSet
import os
import datetime
import tensorflow as tf
import numpy as np

RootPath = os.path.abspath(os.path.dirname(os.getcwd()))

#多轮预测


def predictValue(pModel, iTimes, pTrainDS, pTestDS, pTestLabel):
    for i in range(iTimes):
        print('-------------------%s--------------------' % (str(i)))
        ptrain = pModel.predict(pTrainDS)
        ptest = pModel.predict(pTestDS)
        #l = list()
        iRight = 0
        for i in range(len(ptrain)):
            if ptrain[i] - 1 == 0:
                iRight += 1
        print(iRight / len(ptrain))
        iRight2 = 0
        # Label为正常时出错概率
        iNormalMis = 0
        # Label为异常时出错概率
        iANormalMis = 0
        for k in range(len(ptest)):
            if ptest[k] - pTestLabel[k] == 0:
                iRight2 += 1
            else:
                if pTestLabel[k] == 1:
                    iNormalMis += 1
                if pTestLabel[k] == -1:
                    iANormalMis += 1
                #print(ptest[k], self.__pTestLabel[k])
        iMistake = len(ptest) - iRight2
        print('Normal: %s, Num: %s' % (str(iNormalMis / iMistake), iNormalMis))
        print('ANormal: %s, Num: %s' % (str(iANormalMis / iMistake), iANormalMis))
        print(iRight2 / len(ptest))
        print('-------------------%s--------------------' % (str(i)))


def preProcessingCNNModel(pCNNModel, strWeightPath, strFeatureLayerName):
    latest = tf.train.latest_checkpoint(strWeightPath)
    #载入权重
    pCNNModel.pModel.load_weights(latest)
    #去掉全接层获取特征向量
    newModel = Model(inputs=pCNNModel.pModel.input, outputs=pCNNModel.pModel.get_layer(strFeatureLayerName).output)
    return newModel


class jyAllFreezeLayer:
    def __init__(self, pCNNModel, pDataSet):
        self.__tupleIMGShape = (224, 224, 3)

        self.__pResNet50 = ResNet50(weights='imagenet', include_top=False,
                                    input_shape=self.__tupleIMGShape, pooling='avg')
        #self.__pResNet50.summary()

        self.__pCNNModel = pCNNModel
        self.__pTrainDS = pDataSet.pTrainTFDS
        self.__pTestDS = pDataSet.pTestTFDS
        self.__pTestLabel = pDataSet.pTestLabel
        self.__pAnotherDS = pDataSet.pAnotherTFDS
        self.__pAnotherLabelDS = pDataSet.pAnotherLabel
        #test = self.__pTrainDS[0]
        #import numpy as np
        #test = np.expand_dims(test, axis=0)
        #print(self.__pResNet50.predict(self.__pTrainDS[0]))
        #模型保存路径
        self.__strSVMModelPath = os.path.abspath(os.path.dirname(__file__)) + '\\Model'

        # Hyper Parameter
        # PCA处理后特征将低到此维数 100
        self.__iDimensions = 100
        # nu训练集中负样本比例0.003
        self.__fNu = 0.0025
        # gamma参数在 Gaussian kernel中位于分母位置，选取较大会导致方差很小 导致样本在支持向量样本附近才有高相似度，
        # 图像表现的过于陡峭 对未知样本分类效果不好0.015
        self.__pOCSVM = OneClassSVM(kernel='linear', nu=self.__fNu)#, gamma=0.0135)

    def extractFeature(self):
        self.__pTrainDS = self.__pResNet50.predict(self.__pTrainDS)
        self.__pAnotherDS = self.__pResNet50.predict(self.__pAnotherDS)
        if self.__pTestDS is not None:
            self.__pTestDS = self.__pResNet50.predict(self.__pTestDS)
        #标准化features数值, 便于在PCA算法中计算方差与协方差
        standard = StandardScaler()
        standard.fit(self.__pTrainDS)

        self.__pTrainDS = standard.transform(self.__pTrainDS)
        self.__pAnotherDS = standard.transform(self.__pAnotherDS)
        if self.__pTestDS is not None:
            self.__pTestDS = standard.transform(self.__pTestDS)
        #使用PCA算法对feature进行降维处理, n_components表示降低到多少维度, whiten白化处理 降低冗余性(之后再详细看一下)
        pca = PCA(n_components=self.__iDimensions, whiten=True)
        pca.fit(self.__pTrainDS)
        self.__pTrainDS = pca.transform(self.__pTrainDS)
        self.__pAnotherDS = pca.transform(self.__pAnotherDS)
        if self.__pTestDS is not None:
            self.__pTestDS = pca.transform(self.__pTestDS)
        # OC-SVM
        self.__pOCSVM.fit(self.__pTrainDS)
        # 预测
        predictValue(self.__pOCSVM, 1, self.__pTrainDS, self.__pTestDS, self.__pTestLabel)
        predictValue(self.__pOCSVM, 1, self.__pTrainDS, self.__pAnotherDS, self.__pAnotherLabelDS)
        # 保存模型
        strSavePath = self.__strSVMModelPath + '\\' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        os.makedirs(strSavePath)
        joblib.dump(self.__pOCSVM, strSavePath + '\\' + 'OC_SVM.m')


pModel = jyModel()
pDS = jyDataSet()
strWeightPath = RootPath + '/TensorflowTest/Log/ThreeLayers/'
pNewModel = preProcessingCNNModel(pModel, strWeightPath, 'global_average_pooling2d')
pCNN_SVM = jyAllFreezeLayer(pNewModel, pDS)
pCNN_SVM.extractFeature()
