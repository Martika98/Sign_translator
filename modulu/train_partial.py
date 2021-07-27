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

from datagenerator import VideoClasses, FramesGenerator
from model_i3d import Inception_Inflated3d, add_i3d_top
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)


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
    """
    * Loads pretrained I3D model,
    * reads optical flow data generated from training videos,
    * adjusts top-layers adequately for video data,
    * trains only news top-layers,
    * then fine-tunes entire neural network,
    * saves logs and models to disc.
    """
    train_data_path = 'F://data//Train'
    test_data_path = 'F://data//Validation'
    img_rows = 150
    img_cols = 150
    epochs = 30
    num_of_train_samples = 3000
    num_of_test_samples = 600

    # directories
    sFolder = "%03d-%d" % (diVideoSet["nClasses"], diVideoSet["nFramesNorm"])
    sClassFile = "data-set/%s/%03d/class.csv" % (diVideoSet["sName"], diVideoSet["nClasses"])
    # sVideoDir        = "data-set/%s/%03d"%(diVideoSet["sName"], diVideoSet["nClasses"])
    # sImageDir        = "data-temp/%s/%s/image"%(diVideoSet["sName"], sFolder)
    # sImageFeatureDir = "data-temp/%s/%s/image-i3d"%(diVideoSet["sName"], sFolder)
    sOflowDir = "data-temp/%s/%s/oflow" % (diVideoSet["sName"], sFolder)
    # sOflowFeatureDir = "data-temp/%s/%s/oflow-i3d"%(diVideoSet["sName"], sFolder)

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

    sModelPath = "model/20210715-1211-chalearn004-oflow-i3d-above-last.h5"
    keModel = keras.models.load_model(sModelPath)

    # read the ChaLearn classes
    oClasses = VideoClasses(sClassFile)

    # Load training data
    genFramesTrain = FramesGenerator(sOflowDir + "/train", nBatchSize,
                                     diVideoSet["nFramesNorm"], 224, 224, 2, oClasses.liClasses)
    genFramesVal = FramesGenerator(sOflowDir + "/val", nBatchSize,
                                   diVideoSet["nFramesNorm"], 224, 224, 2, oClasses.liClasses)

    # Load pretrained i3d model and adjust top layer

    # Prep logging
    sLog = time.strftime("%Y%m%d-%H%M", time.gmtime()) + \
           "-%s%03d-oflow-i3d" % (diVideoSet["sName"], diVideoSet["nClasses"])

    # Helper: Save results
    csv_logger = keras.callbacks.CSVLogger("log/" + sLog + "-acc.csv", append=True)

    cpTopLast = keras.callbacks.ModelCheckpoint(filepath='model' + "/" + sLog + "-above-last.h5", verbose=0)
    cpTopBest = keras.callbacks.ModelCheckpoint(filepath='model' + "/" + sLog + "-above-best.h5",
                                                verbose=1, save_best_only=True)
    cpAllLast = keras.callbacks.ModelCheckpoint(filepath='model' + "/" + sLog + "-entire-last.h5", verbose=0)
    cpAllBest = keras.callbacks.ModelCheckpoint(filepath='model' + "/" + sLog + "-entire-best.h5",
                                                verbose=1, save_best_only=True)

    # Helper: Save the model

    # Fit top layers

    # Fit entire I3D model

    keModel.fit_generator(
        generator=genFramesTrain,
        validation_data=genFramesVal,
        epochs=diTrainAll["nEpochs"],
        workers=4,
        use_multiprocessing=True,
        max_queue_size=8,
        verbose=1,
        callbacks=[csv_logger, cpAllLast, cpAllBest])

    Y_pred1 = keModel.predict_generator(genFramesVal, num_of_test_samples // nBatchSize + 1)
    y_pred = np.argmax(Y_pred1, axis=1)
    print('Confusion Matrix')
    print(confusion_matrix(genFramesVal.liClasses, y_pred))
    print('Classification Report')
    target_names = ['Dom', 'Motyl', 'Plywac', 'Patrzec']
    print(classification_report(genFramesVal.liClasses, y_pred, target_names=target_names))

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