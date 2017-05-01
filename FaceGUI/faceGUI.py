# -*- coding: utf-8 -*-
from PyQt5.QtWidgets import (QWidget, QLabel, QLineEdit, QApplication, QVBoxLayout, QPushButton)
from PyQt5 import QtGui, QtCore
import cv2
import dlib
import numpy as np
import pickle

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/home/malov/PycharmProjects/object_detection/shape_predictor_68_face_landmarks.dat')
INNER_EYES_AND_BOTTOM_LIP = [39, 42, 57]
TEMPLATE = np.float32([
    (0.0792396913815, 0.339223741112), (0.0829219487236, 0.456955367943),
    (0.0967927109165, 0.575648016728), (0.122141515615, 0.691921601066),
    (0.168687863544, 0.800341263616), (0.239789390707, 0.895732504778),
    (0.325662452515, 0.977068762493), (0.422318282013, 1.04329000149),
    (0.531777802068, 1.06080371126), (0.641296298053, 1.03981924107),
    (0.738105872266, 0.972268833998), (0.824444363295, 0.889624082279),
    (0.894792677532, 0.792494155836), (0.939395486253, 0.681546643421),
    (0.96111933829, 0.562238253072), (0.970579841181, 0.441758925744),
    (0.971193274221, 0.322118743967), (0.163846223133, 0.249151738053),
    (0.21780354657, 0.204255863861), (0.291299351124, 0.192367318323),
    (0.367460241458, 0.203582210627), (0.4392945113, 0.233135599851),
    (0.586445962425, 0.228141644834), (0.660152671635, 0.195923841854),
    (0.737466449096, 0.182360984545), (0.813236546239, 0.192828009114),
    (0.8707571886, 0.235293377042), (0.51534533827, 0.31863546193),
    (0.516221448289, 0.396200446263), (0.517118861835, 0.473797687758),
    (0.51816430343, 0.553157797772), (0.433701156035, 0.604054457668),
    (0.475501237769, 0.62076344024), (0.520712933176, 0.634268222208),
    (0.565874114041, 0.618796581487), (0.607054002672, 0.60157671656),
    (0.252418718401, 0.331052263829), (0.298663015648, 0.302646354002),
    (0.355749724218, 0.303020650651), (0.403718978315, 0.33867711083),
    (0.352507175597, 0.349987615384), (0.296791759886, 0.350478978225),
    (0.631326076346, 0.334136672344), (0.679073381078, 0.29645404267),
    (0.73597236153, 0.294721285802), (0.782865376271, 0.321305281656),
    (0.740312274764, 0.341849376713), (0.68499850091, 0.343734332172),
    (0.353167761422, 0.746189164237), (0.414587777921, 0.719053835073),
    (0.477677654595, 0.706835892494), (0.522732900812, 0.717092275768),
    (0.569832064287, 0.705414478982), (0.635195811927, 0.71565572516),
    (0.69951672331, 0.739419187253), (0.639447159575, 0.805236879972),
    (0.576410514055, 0.835436670169), (0.525398405766, 0.841706377792),
    (0.47641545769, 0.837505914975), (0.41379548902, 0.810045601727),
    (0.380084785646, 0.749979603086), (0.477955996282, 0.74513234612),
    (0.523389793327, 0.748924302636), (0.571057789237, 0.74332894691),
    (0.672409137852, 0.744177032192), (0.572539621444, 0.776609286626),
    (0.5240106503, 0.783370783245), (0.477561227414, 0.778476346951)])

TPL_MIN, TPL_MAX = np.min(TEMPLATE, axis=0), np.max(TEMPLATE, axis=0)
MINMAX_TEMPLATE = (TEMPLATE - TPL_MIN) / (TPL_MAX - TPL_MIN)

class QtCapture(QWidget):
    def __init__(self, *args):
        super(QWidget, self).__init__()

        self.fps = 24
        self.cap = cv2.VideoCapture(*args)

        self.video_frame = QLabel()
        lay = QVBoxLayout()
        # lay.setContentsMargins(0)
        lay.addWidget(self.video_frame)
        self.setLayout(lay)

    def setFPS(self, fps):
        self.fps = fps

    def nextFrameSlot(self):
        #
        # def _get_aligned_frame(frame):
        #     detection = detector(frame, 1)
        #     bb = max(detection, key=lambda rect: rect.width() * rect.height())
        #     points = predictor(frame, bb)
        #     landmarks = list(map(lambda p: (p.x, p.y), points.parts()))
        #     np_landmarks = np.float32(landmarks)
        #     np_landmarks_indices = np.array(INNER_EYES_AND_BOTTOM_LIP)
        #     affine_transform = cv2.getAffineTransform(np_landmarks[np_landmarks_indices],
        #                                               96 * MINMAX_TEMPLATE[np_landmarks_indices])
        #     return cv2.warpAffine(frame, affine_transform, (96, 96))
        #
        def _infer(frame):
            with open('/home/malov/openface/generated-embeddings/classifier.pkl', 'r') as f:
                (le, clf) = pickle.load(f)
            reps = []
            frame1 = _get_aligned_frame(frame)
            win = dlib.image_window()
            win.clear_overlay()
            win.set_image(frame)
            dets = detector(frame, 1)
            for k, d in enumerate(dets):
                shape = predictor(frame1, d)
                win.clear_overlay()
                win.add_overlay(d)
                win.add_overlay(shape)
                face_descriptor = recognizer.compute_face_descriptor(frame1, shape)
                face_info = np.array(face_descriptor)  # 128 characters
                reps.append(face_info)
            reps = np.array(reps)
            predictions = clf.predict_proba(reps)
            max_prob = np.argmax(predictions)
            person = le.inverse_transform(max_prob)
            confidence = predictions[max_prob]
            return '{} (с вероятностью {:.2f})'.format(person, confidence)

        ret, frame = self.cap.read()
        # OpenCV yields frames in BGR format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = QtGui.QImage(frame, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(img)
        self.video_frame.setPixmap(pix)



    def start(self):
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.nextFrameSlot)
        self.timer.start(1000./self.fps)

    def stop(self):
        self.timer.stop()

    def deleteLater(self):
        self.cap.release()
        super(QWidget, self).deleteLater()

class ControlWindow(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        self.capture = None

        self.start_button = QPushButton('Start')
        self.start_button.clicked.connect(self.startCapture)
        self.quit_button = QPushButton('End')
        self.quit_button.clicked.connect(self.endCapture)
        self.end_button = QPushButton('Stop')
        self.session_button = QPushButton('Распознать')
        self.session_button.clicked.connect(self.personRecognition)

        vbox = QVBoxLayout(self)
        vbox.addWidget(self.start_button)
        vbox.addWidget(self.end_button)
        vbox.addWidget(self.quit_button)
        vbox.addWidget(self.session_button)
        self.setLayout(vbox)
        self.setWindowTitle('Control Panel')
        self.setGeometry(100,100,200,200)
        self.show()

    def startCapture(self):
        if not self.capture:
            self.capture = QtCapture(0)
            self.end_button.clicked.connect(self.capture.stop)
            # self.capture.setFPS(1)
            self.capture.setParent(self)
            self.capture.setWindowFlags(QtCore.Qt.Tool)
        self.capture.start()
        self.capture.show()

    def personRecognition(self):
        # if not self.capture:
        #     self.capture = QtCapture(0)
        #     self.end_button.clicked.connect(self.capture.stop)
        #     # self.capture.setFPS(1)
        #     self.capture.setParent(self)
        #     self.capture.setWindowFlags(QtCore.Qt.Tool)
        # self.capture.start()
        # self.capture.show()
        with open('/home/malov/openface/generated-embeddings/classifier.pkl', 'r') as f:

            (self.le, self.clf) = pickle.load(f)
        self.reps = []
        self.frame1 = _get_aligned_frame(frame)
        self.win = dlib.image_window()
        self.win.clear_overlay()
        self.win.set_image(frame)
        self.dets = detector(frame, 1)
        for k, d in enumerate(self.dets):
            self.shape = predictor(frame1, d)
            self.win.clear_overlay()
            self.win.add_overlay(d)
            self.win.add_overlay(shape)
            self.face_descriptor = recognizer.compute_face_descriptor(self.frame1, self.shape)
            self.face_info = np.array(face_descriptor)  # 128 characters
            self.reps.append(face_info)
        self.reps = np.array(reps)
        self.predictions = clf.predict_proba(reps)
        self.max_prob = np.argmax(predictions)
        self. person = le.inverse_transform(max_prob)
        self.confidence = predictions[max_prob]
        return '{} (с вероятностью {:.2f})'.format(self.person, self.confidence)

    def endCapture(self):
        self.capture.deleteLater()
        self.capture = None


if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    window = ControlWindow()
    sys.exit(app.exec_())