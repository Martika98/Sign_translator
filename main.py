from model_i3d import I3D_load
from predict import probability2label
import os
import numpy as np
from opticalflow import frames2flows, flows2colorimages, flows2file, flows_add_third_channel
from datagenerator import VideoClasses
from frame import video2frames, images_normalize, frames_downsample, images_crop
import cv2
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont, QPixmap, QImage
from PyQt5.QtWidgets import QVBoxLayout, QWidget, QLabel, QSpacerItem, QSizePolicy
import time
from PyQt5 import QtGui, QtCore
import keyboard
import sys
from PyQt5.QtWidgets import QApplication
import threading


class Gui(QWidget):
    def __init__(self):
        super().__init__()

        self.cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.frame = None

        self.setWindowTitle("Translator")
        # self.setGeometry(0, 0, 1800, 800)

        # setup widget for displaying camera images
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        self.frame_widget = QLabel()
        self.frame_widget.setAlignment(QtCore.Qt.AlignCenter)
        main_layout.addWidget(self.frame_widget)
        self.setLayout(main_layout)

        # setup overlaying widgets
        frame_layout = QVBoxLayout()
        # frame_layout.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel()
        self.label.setWordWrap(True)
        self.label.setFont(QFont('Helvetica', 50, weight=12))
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setStyleSheet("QLabel {color : white; background: rgba(0, 0, 0, 40)}")
        frame_layout.addWidget(self.label)
        self.spacer = QSpacerItem(20, 40, hPolicy=QSizePolicy.Minimum, vPolicy=QSizePolicy.Expanding)
        frame_layout.addSpacerItem(self.spacer)
        self.frame_widget.setLayout(frame_layout)
        #flags = QtCore.Qt.WindowFlags(QtCore.Qt.FramelessWindowHint)
        # flags = QtCore.Qt.WindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint)
        #self.setWindowFlags(flags)
        self.show()
        #self.setFocus()

        # refresh frame every 1000 / 30 ms
        self.timer = QTimer()
        self.timer.timeout.connect(self.change_frame)
        self.timer.start(1000 / 60)


    def change_frame(self):
        _, frame = self.cam.read()
        self.frame = frame
        frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame = QImage(frame.data,
                       frame.shape[1],
                       frame.shape[0],
                       frame.shape[1]*3,  # delete this
                       QImage.Format_RGB888)
        frame = QPixmap.fromImage(frame)
        frame = frame.scaled(self.frame_widget.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.frame_widget.setPixmap(frame)
        self.update()

    def set_label(self, label):
        self.label.setText(label)


class Translator(threading.Thread):
    def __init__(self, gui):
        super().__init__()
        self.gui = gui


    def run(self):
        # dataset
        diVideoSet = {"sName": "chalearn",
                      "nClasses": 4,  # number of classes
                      "nFramesNorm": 70,  # number of frames per video
                      "nMinDim": 240,  # smaller dimension of saved video-frames
                      "tuShape": (240, 320),  # height, width
                      "nFpsAvg": 10,
                      "nFramesAvg": 50,
                      "fDurationAvg": 5.0}  # seconds

        # files
        sClassFile = "data-set/%s/%03d/class.csv" % (diVideoSet["sName"], diVideoSet["nClasses"])
        sVideoDir = "data-set/%s/%03d" % (diVideoSet["sName"], diVideoSet["nClasses"])

        print("\nStarting gesture recognition live demo ... ")
        print(os.getcwd())
        print(diVideoSet)

        # load label description
        oClasses = VideoClasses(sClassFile)

        sModelFile = "model/20210715-1322-chalearn004-oflow-i3d-entire-best.h5"
        h, w = 224, 224
        keI3D = I3D_load(sModelFile, diVideoSet["nFramesNorm"], (h, w, 2), oClasses.nClasses)

        # liVideosDebug = glob.glob(sVideoDir + "/train/*/*.*")
        sResults = ""
        self.gui.set_label("Naciśnij spację aby rozpocząć")
        # loop over action states
        while True:
            # start!
            if keyboard.is_pressed(' '):
                self.gui.set_label("Zaczynam nagrywanie")
                time.sleep(1)
                frame_list = []
                start = time.time()
                now = start
                while (now - start) < diVideoSet["fDurationAvg"]:
                    sec_ = 5 - int(now - start)
                    self.gui.set_label(f"{sec_}s")
                    now = time.time()
                    frame = self.gui.frame
                    frame = cv2.resize(frame, (320,240))
                    frame_list.append(frame)
                    time.sleep(0.05)
                self.gui.set_label("Przetwarzanie...")
                ar_frames = np.array(frame_list)

                ar_frames = images_crop(ar_frames, h, w)
                ar_frames = frames_downsample(ar_frames, diVideoSet["nFramesNorm"])

                # Translate frames to flows - these are already scaled between [-1.0, 1.0]
                print("Calculate optical flow on %d frames ..." % len(ar_frames))
                ar_flows = frames2flows(ar_frames, bThirdChannel=False, bShow=True)

                # predict video from flows
                print("Predict video with %s ..." % (keI3D.name))
                arX = np.expand_dims(ar_flows, axis=0)
                arProbas = keI3D.predict(arX, verbose=1)[0]
                nLabel, sLabel, fProba = probability2label(arProbas, oClasses, nTop=3)

                sResults = "Sign: %s (%.0f%%)" % (sLabel, fProba * 100.)
                self.gui.set_label(sLabel)
                print(sResults)
                time.sleep(3)
                self.gui.set_label("Naciśnij spację aby rozpocząć")

            # quit
            elif keyboard.is_pressed('q'):
                break

        return


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = Gui()
    translator_ = Translator(gui)
    translator_.start()
    app.exec_()
