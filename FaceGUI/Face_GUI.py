# -*- coding: utf-8 -*-
import Tkinter as tk
import cv2
from PIL import Image, ImageTk
import os
from ImageProcessing.image_processing import get_face_shape_coordinates, \
    _get_aligned_frame, get_face_detection
from Classifier.classifier import Classifier
from Constants.constants import WORKING_DIRECTORY
from datetime import datetime
from time import sleep, time



"""
class UserGUI - создается в каждой новой сессии
"""


class UserGUI:
    def __init__(self, camera_index):
        # self.width = 640
        # self.height = 480
        self.cap = cv2.VideoCapture(camera_index)
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.root = tk.Tk()
        self.root.bind('<Escape>', lambda e: self.root.quit())
        self.lmain = tk.Label(self.root)
        self.lmain.pack()

    def set_face_shape_contour(self, frame):
        for (x, y) in get_face_shape_coordinates(frame):
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    def set_person_name(self, frame, person_name, confidence):
        x = get_face_shape_coordinates(frame)[0][0]
        y = get_face_shape_coordinates(frame)[0][1]
        cv2.putText(frame, '{}-{}%'.format(person_name, confidence * 100), (x, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    def make_user_directory(self, person_name):
        user_directory = WORKING_DIRECTORY + '{}-{}'.format(len(os.listdir(WORKING_DIRECTORY)),
                                                            person_name)
        os.makedirs(user_directory)
        return user_directory

    def save_user_photos(self, frame, user_directory):
        # cv2.imshow('saving', frame)
        cv2.imwrite('{}/{}-{}-{}_{}:{}:{}.jpg'.format(user_directory,
                                                      datetime.now().year,
                                                      datetime.now().month,
                                                      datetime.now().day,
                                                      datetime.now().hour,
                                                      datetime.now().minute,
                                                      datetime.now().second), frame)

    def make_user_profile(self, person_name):
        photos_list = []
        user_directory = self.make_user_directory(person_name)
        while len(photos_list) < 10:
            print(10 - len(photos_list))
            ret, frame = self.cap.read()
            if ret:
                detection = get_face_detection(frame)
                if not detection:
                    print('Ближе к камере')
                    sleep(1)
                else:
                    self.save_user_photos(frame, user_directory)
                    sleep(1)
                    photos_list.append(frame)
        self.cap.release()
        self.cap.open(0)

    def add_user(self):
        person_name = raw_input('Введите ФИО в формате:  \'familiya-imya\'\n')
        self.make_user_profile(person_name)
        add_user = Classifier()
        add_user.train()
        print('Пользователь успешно добавлен')

    def show_frame(self):
        start = time()
        _, frame = self.cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            present_user = Classifier()
            present_user.infer(frame)
            # self.set_person_name(frame, present_user.person_name, present_user.confidence)
            self.set_face_shape_contour(frame)
            self.img = Image.fromarray(frame)
            self.imgtk = ImageTk.PhotoImage(image=self.img)
            self.lmain.imgtk = self.imgtk
            self.lmain.configure(image=self.imgtk)
            self.lmain.after(1, self.show_frame)
            print(time()-start)
            print(present_user.person_name, present_user.confidence, present_user.face_128_chars[:6])
        except:
            print('Камера не видит твое лицо')
            self.img = Image.fromarray(frame)
            self.imgtk = ImageTk.PhotoImage(image=self.img)
            self.lmain.imgtk = self.imgtk
            self.lmain.configure(image=self.imgtk)
            self.lmain.after(1, self.show_frame)
            print(time()-start)



if __name__ == '__main__':
    user = UserGUI(0)
    user.add_user()
    user.show_frame()
    user.root.mainloop()
