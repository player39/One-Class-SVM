
import os
import tensorflow as tf
from tensorflow.python.keras import layers, models
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import SGD, adam
import datetime


class jyModel(object):
    def __init__(self):
        self.pModel = Sequential()

        self.pModel.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(244, 244, 3)))
        self.pModel.add(layers.MaxPool2D(2, 2))

        #self.pModel.add(layers.Conv2D(64, (3, 3), activation='relu'))
        #self.pModel.add(layers.MaxPool2D(2, 2))

        self.pModel.add(layers.Conv2D(64, (3, 3), activation='relu'))

        self.pModel.add(layers.GlobalAveragePooling2D())
        self.pModel.add(layers.Dense(1, activation='sigmoid'))

        self.__strSavePath = os.path.abspath(os.path.dirname(__file__)) + '//Log//ThreeLayers//cp-{epoch:04d}.ckpt'
        self.__pSaveModel = ModelCheckpoint(self.__strSavePath, save_weights_only=True, verbose=1, period=10)
        #定义梯度下降函数
        pOptimizer = SGD(lr=0.001)
        pAdam = adam()

        self.pModel.compile(optimizer='adam',
                          loss='binary_crossentropy',
                          metrics=['accuracy'])
        self.pModel.summary()

    def trainStart(self, pX, pY, tupleValidDS):
        #with tf.device('/GPU:0'):
        log_dir = os.path.join(
            "Log",
            "ThreeLayers",
            datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        ) #+ #datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        pTensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)
        self.pModel.fit(pX, pY,
                        epochs=800, batch_size=64,
                        validation_data=tupleValidDS, callbacks=[self.__pSaveModel, pTensorboard])

    def evaluateModel(self, pTestX, pTestY, iTestLen):
        testLoss, testAccuracy = self.pModel.evaluate(pTestX, pTestY)
        print("Loss: %.4f, Accuracy: %.4f, Test Image Num: %d" % (testLoss, testAccuracy, iTestLen))
