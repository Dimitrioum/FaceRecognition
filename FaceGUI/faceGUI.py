# -*- coding: utf-8 -*-
"""
GUI for user interaction with face recognition system, implemented into the working terminal
"""

from PyQt5.QtWidgets import QWidget, QLabel, QApplication, QVBoxLayout, QPushButton
from PyQt5 import QtGui, QtCore
import cv2
from ImageProcessing.image_processing import get_face_shape_coordinates, get_128_face_chars
from Classifier.classifier import Classifier
from Constants.constants import CLASSIFIER_PATH
import pickle
import numpy as np
from UserProfile.UserProfile import make_user_profile


class QtCapture(QWidget):
    """
    Control Panel for user interaction with face recognition system
    """
    def __init__(self, *args):
        super(QWidget, self).__init__()
        self.fps = 30
        self.video_frame = QLabel()
        self.cap = cv2.VideoCapture(*args)
        self.user_classifier = Classifier()
        lay = QVBoxLayout()
        # lay.setContentsMargins(0)
        lay.addWidget(self.video_frame)
        self.setLayout(lay)
        self.label_encoder = None
        self.classifier = None
        self.person_name = None
        self.confidence = None

    def set_face_shape_contour(self, frame, face_shape_coordinates):
        """
        Set 68-point face-shape at the frame with user face
        :param frame: RGB-frame from the camera
        :type frame: np.array
        :param face_shape_coordinates: coordinates of face shape, detected on the frame
        :type face_shape_coordinates: np.array
        """
        for (x, y) in face_shape_coordinates:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    def set_person_name(self, frame, person_name, confidence):
        """
        Set inferred person name on the frame with detected face
        :param frame: RGB-frame from the camera
        :type frame: np.array
        :param person_name: name of the user, whose face was detected and inferred by the classifier
        :type: str
        :param confidence: percentage of successful inference 
        :type confidence: float
        """
        x = get_face_shape_coordinates(frame)[0][0]
        y = get_face_shape_coordinates(frame)[0][1]
        cv2.putText(frame, '{}-{}%'.format(person_name, confidence * 100), (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    def user_infer(self, frame):
        """
        Recognition of the detected user face on the frame from the camera
        :param frame: frame from the camera
        :type frame: np.array
        :return: inferred person name and confidence of inference
        """
        try:
            face_128_chars = np.array(get_128_face_chars(frame))
            predictions = self.classifier.predict_proba(face_128_chars.reshape(1, -1)).ravel()
            max_prob = np.argmax(predictions)
            self.person_name = self.label_encoder.inverse_transform(max_prob)
            self.confidence = predictions[max_prob]
        except:
            pass

    def next_frame_slot(self):
        """
        Getting, processing and demonstration the frame from camera
        """
        _, frame = self.cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            self.user_infer(frame)
            face_shape_coordinates = get_face_shape_coordinates(frame)
            self.set_face_shape_contour(frame, face_shape_coordinates)
            # self.set_person_name(frame, self.person_name, self.confidence)
            print self.person_name, self.confidence
            img = QtGui.QImage(frame, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(img)
            self.video_frame.setPixmap(pix)
        except TypeError:
            print 'Блииже к камере'
            img = QtGui.QImage(frame, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(img)
            self.video_frame.setPixmap(pix)

    def start(self):
        """
        Starts the terminal session
        """
        self.timer = QtCore.QTimer()
        with open(CLASSIFIER_PATH, 'r') as clf:
            self.label_encoder, self.classifier = pickle.load(clf)
        self.timer.timeout.connect(self.next_frame_slot)
        self.timer.start(1000. / self.fps)

    def stop(self):
        """
        Stops the terminal session
        """
        self.timer.stop()

    def deleteLater(self):
        """
        Stops getting the frames from the camera 
        """
        self.cap.release()
        super(QWidget, self).deleteLater()


class ControlWindow(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        self.capture = None
        self.person_name = None
        self.user = None
        self.face_128_chars = []
        self.start_button = QPushButton('Начало сессии')
        self.start_button.clicked.connect(self.start_capture)
        self.quit_button = QPushButton('Закончить сессию')
        self.quit_button.clicked.connect(self.end_capture)
        self.session_button = QPushButton('Добавить пользователя')
        self.session_button.clicked.connect(self.add_user)

        vbox = QVBoxLayout(self)
        vbox.addWidget(self.session_button)
        vbox.addWidget(self.start_button)
        vbox.addWidget(self.quit_button)
        self.setLayout(vbox)
        self.setWindowTitle('Панель управления')
        self.setGeometry(100, 100, 200, 200)
        self.show()

    def add_user(self):
        """
        Starts session of adding new user
        """
        self.person_name = raw_input('Введите ФИО в формате:  \'familiya-imya\'\n')
        make_user_profile(self.person_name)
        if not self.capture:
            self.capture = QtCapture(0)
            self.capture.user_classifier.train_classifier()
            # print(self.capture.user_classifier.classifier, self.capture.user_classifier.label_encoder)
            print 'Пользователь успешно добавлен'

    def start_capture(self):
        """
        Starts demonstrating frames from the camera
        """
        if not self.capture:
            self.capture = QtCapture(0)
            # self.capture.setFPS(1)
            self.capture.user_classifier = Classifier()
            self.capture.setParent(self)
            self.capture.setWindowFlags(QtCore.Qt.Tool)
        self.capture.start()
        self.capture.show()

    def end_capture(self):
        """
        Ends demonstrating frames from the camera 
        """
        self.capture.deleteLater()
        self.capture = None


if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    window = ControlWindow()
    sys.exit(app.exec_())
