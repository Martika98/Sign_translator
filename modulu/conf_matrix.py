"""
https://github.com/FrederikSchorr/sign-language

Train a pre-trained I3D convolutional network to classify videos
"""

import os
import glob
import time
import sys

import numpy as np
import pandas as pd

from tensorflow import keras
import tensorflow.compat.v1.keras.backend as K

from datagenerator import VideoClasses, FramesGenerator, FeaturesGenerator
from feature import features_3D_predict_generator
from model_i3d import Inception_Inflated3d, add_i3d_top
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)

from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix


def layers_freeze(keModel: keras.Model) -> keras.Model:
    print("Freeze all %d layers in Model %s" % (len(keModel.layers), keModel.name))
    for layer in keModel.layers:
        layer.trainable = False

    return keModel


def layers_unfreeze(keModel: keras.Model) -> keras.Model:
    print("Unfreeze all %d layers in Model %s" % (len(keModel.layers), keModel.name))
    for layer in keModel.layers:
        layer.trainable = True

    return keModel


def count_params(keModel: keras.Model):
    trainable_count = int(
        np.sum([K.count_params(p) for p in set(keModel.trainable_weights)]))
    non_trainable_count = int(
        np.sum([K.count_params(p) for p in set(keModel.non_trainable_weights)]))

    print('Total params: {:,}'.format(trainable_count + non_trainable_count))
    print('Trainable params: {:,}'.format(trainable_count))
    print('Non-trainable params: {:,}'.format(non_trainable_count))

    return


def train_I3D_oflow_end2end(diVideoSet):

    num_of_test_samples = 356

    # directories
    sFolder = "%03d-%d" % (diVideoSet["nClasses"], diVideoSet["nFramesNorm"])
    sClassFile = "data-set/%s/%03d/class.csv" % (diVideoSet["sName"], diVideoSet["nClasses"])
    sOflowDir = "data-temp/%s/%s/oflow" % (diVideoSet["sName"], sFolder)
    # sOflowFeatureDir = "data-temp/%s/%s/oflow-i3d"%(diVideoSet["sName"], sFolder)
    sImageFeatureDir = "data-temp/%s/%s/image-i3d"%(diVideoSet["sName"], sFolder)
    sImageDir = "data-temp/%s/%s/image" % (diVideoSet["sName"], sFolder)

    sModelDir = "model"

    diTrainTop = {
        "fLearn": 1e-3,
        "nEpochs": 3}

    diTrainAll = {
        "fLearn": 1e-4,
        "nEpochs": 17}

    nBatchSize = 4

    print("\nStarting I3D end2end training ...")
    print(os.getcwd())

    sModelPath = "model/final_model.h5"
    #keModel = keras.models.load_model(sModelPath)

    keModel = keras.models.load_model(sModelPath)
    print(keModel.input_shape)
    oClasses = VideoClasses(sClassFile)

    # chuj = sOflowDir + "/val"
    #
    # print(f" HALO TUTAJ DRUKOWANKO {features_3D_predict_generator(chuj, sImageFeatureDir, keModel, nBatchSize)}")


    # Load training data
    # genFramesTrain = FramesGenerator(sOflowDir + "/train", nBatchSize,
    #                                  diVideoSet["nFramesNorm"], 224, 224, 2, oClasses.liClasses)
    # genFramesVal = FramesGenerator(sOflowDir + "/val", nBatchSize,
    #                                diVideoSet["nFramesNorm"], 224, 224, 2, oClasses.liClasses)
    #
    #
    #
    #
    print("Load model %s ..." % sModelPath)


    # Load video features
    genFeatures = FeaturesGenerator(sImageFeatureDir, nBatchSize,
                                    keModel.input_shape[1:], oClasses.liClasses, bShuffle=False)
    if genFeatures.nSamples == 0: raise ValueError("No feature files detected, prediction stopped")

    # predict
    arProba = keModel.predict_generator(
        generator=genFeatures,
        workers=1,
        use_multiprocessing=False,
        verbose=1)
    if arProba.shape[0] != genFeatures.nSamples: raise ValueError("Unexpected number of predictions")

    arPred = arProba.argmax(axis=1)
    liLabels = list(genFeatures.dfSamples.sLabel)
    for i in range(len(liLabels)):
        liLabels[i] = int(liLabels[i][3])-1
    #
    #
    #
    print(f"Predsy z modelu : {arPred}")
    print(f"GroundTruth {liLabels}")
    print('Confusion Matrix')
    print(confusion_matrix(liLabels, arPred))
    print('Classification Report')
    target_names = ['Dom', 'Motyl', 'Plywac', 'Patrzec']
    print(classification_report(liLabels, arPred, target_names=target_names))

    return


if __name__ == '__main__':
    """diVideoSet = {"sName" : "ledasila",
        "nClasses" : 21,   # number of classes
        "nFramesNorm" : 40,    # number of frames per video
        "nMinDim" : 240,   # smaller dimension of saved video-frames
        "tuShape" : (288, 352), # height, width
        "nFpsAvg" : 25,
        "nFramesAvg" : 75,
        "fDurationAvg" : 3.0} # seconds
    """
    diVideoSet = {"sName" : "chalearn",
    "nClasses" : 4,   # number of classes
    "nFramesNorm" : 70,    # number of frames per video
    "nMinDim" : 240,   # smaller dimension of saved video-frames
    "tuShape" : (240, 320), # height, width
    "nFpsAvg" : 10,
    "nFramesAvg" : 50,
    "fDurationAvg" : 4.0}  # seconds

    train_I3D_oflow_end2end(diVideoSet)