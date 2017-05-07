# -*- coding: utf-8 -*-
from PyQt5.QtWidgets import QWidget, QLabel, QApplication, QVBoxLayout, QPushButton
from PyQt5 import QtGui, QtCore
import cv2
import os
from ImageProcessing.image_processing import _get_face_shape_coordinates
from Classifier.classifier import Classifier
from Constants.constants import WORKING_DIRECTORY
import datetime
from time import sleep


class QtCapture(QWidget):
    def __init__(self):
        super(QWidget, self).__init__()
        self.fps = 30
        # self.cap = cv2.VideoCapture(*args)
        # self.cap.release()
        # self.cap.open(0)
        self.user_classifier = None
        self.video_frame = QLabel()
        lay = QVBoxLayout()
        # lay.setContentsMargins(0)
        lay.addWidget(self.video_frame)
        self.setLayout(lay)

    def set_face_shape_contour(self, frame, face_shape_coordinates):
        for (x, y) in face_shape_coordinates:
            return cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    def set_person_name(self, frame, person_name, confidence):
        x = _get_face_shape_coordinates(frame)[0][0]
        y = _get_face_shape_coordinates(frame)[0][1]
        return cv2.putText(frame, '{}-{}%'.format(person_name, confidence * 100), (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    def nextFrameSlot(self):
        self.ret, self.frame = cv2.VideoCapture(0).read()
        frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        face_shape_coordinates = _get_face_shape_coordinates(frame)
        person_name, confidence = self.user_classifier._infer(frame)
        self.set_face_shape_contour(frame, face_shape_coordinates)
        self.set_person_name(frame, person_name, confidence)
        img = QtGui.QImage(frame, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(img)
        self.video_frame.setPixmap(pix)

    def start(self):
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.nextFrameSlot)
        self.timer.start(1000. / self.fps)

    def stop(self):
        self.timer.stop()

    def deleteLater(self):
        self.cap.release()
        super(QWidget, self).deleteLater()


class ControlWindow(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        self.capture = None
        self.face_128_chars = []
        self.start_button = QPushButton('Начало сессии')
        self.start_button.clicked.connect(self.startCapture)
        self.quit_button = QPushButton('Закончить сессию')
        self.quit_button.clicked.connect(self.endCapture)
        self.session_button = QPushButton('Добавить пользователя')
        self.session_button.clicked.connect(self.addUser)

        vbox = QVBoxLayout(self)
        vbox.addWidget(self.session_button)
        vbox.addWidget(self.start_button)
        vbox.addWidget(self.quit_button)
        self.setLayout(vbox)
        self.setWindowTitle('Панель управления')
        self.setGeometry(100, 100, 200, 200)
        self.show()

    def _make_user_directory(self, user_directory):
        if not os.path.exists(user_directory):
            os.makedirs(user_directory)

    def _save_user_photos(self, frame, user_directory):
        cv2.imshow('saving', frame)
        cv2.imwrite('{}/{}-{}-{} {}:{}:{}:{}.jpg'.format(user_directory,
                                                         datetime.datetime.now().year,
                                                         datetime.datetime.now().month,
                                                         datetime.datetime.now().day,
                                                         datetime.datetime.now().hour,
                                                         datetime.datetime.now().minute,
                                                         datetime.datetime.now().second,
                                                         datetime.datetime.now().microsecond), frame)

    def _make_user_profile(self, person_name):
        photos_list = []
        self.user_directory = WORKING_DIRECTORY + '{}-{}'.format(len(os.listdir(WORKING_DIRECTORY)),
                                                                 person_name)
        self._make_user_directory(self.user_directory)
        cap = cv2.VideoCapture(0)
        while len(photos_list) < 10:
            print(10 - len(photos_list))
            ret, frame = cap.read()
            print('До')
            if ret:
                print('После')
                self._save_user_photos(frame, self.user_directory)
                photos_list.append(frame)
                sleep(1)
        cap.release()
        cap.open(0)

    def addUser(self):
        person_name = raw_input('Введите ФИО в формате:  \'familiya-imya\'\n')
        self._make_user_profile(person_name)
        self.capture.user_classifier = Classifier(users_directory=self.user.user_directory,
                                                  classifier_path=None)
        print('Пользователь успешно добавлен')


    def startCapture(self):
        if not self.capture:
            self.capture = QtCapture
            self.end_button.clicked.connect(self.capture.stop)
            # self.capture.setFPS(1)
            self.capture.setParent(self)
            self.capture.setWindowFlags(QtCore.Qt.Tool)
        self.capture.start()
        self.capture.show()

    def endCapture(self):
        self.capture.deleteLater()
        self.capture = None


if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    window = ControlWindow()
    sys.exit(app.exec_())
