# -*- coding: utf-8 -*-
"""
Methods for creating user's profile
"""
from Constants.constants import WORKING_DIRECTORY
import os
import cv2
from time import sleep
from datetime import datetime
from ImageProcessing.image_processing import get_face_detection


def _make_user_directory(person_name):
    """
    Creates directory for user profile into the working directory
    :param person_name: name of the user
    :type person_name: str
    :return: name of the user directory
    """
    user_directory = WORKING_DIRECTORY + '{}-{}'.format(len(os.listdir(WORKING_DIRECTORY)),
                                                        person_name)
    os.makedirs(user_directory)
    return user_directory


def _save_user_photos(frame, user_directory):
    """
    Saves users' photos at the users' directory
    :param frame: frame with the user's face
    :type frame: np.array
    :param user_directory: user directory within the working directory
    :type user_directory: str
    :return: 
    """
    cv2.imwrite('{}/{}-{}-{}_{}:{}:{}.jpg'.format(user_directory,
                                                  datetime.now().year,
                                                  datetime.now().month,
                                                  datetime.now().day,
                                                  datetime.now().hour,
                                                  datetime.now().minute,
                                                  datetime.now().second), frame)


def make_user_profile(person_name):
    """
    Initialize camera session for creating user profile
    :param person_name: user name
    :type: str
    """
    _photos_list = []
    user_directory = _make_user_directory(person_name)
    cap = cv2.VideoCapture(0)
    while len(_photos_list) < 10:
        print(10 - len(_photos_list))
        ret, frame = cap.read()
        if ret:
            detection = get_face_detection(frame)
            if not detection:
                print('Ближе к камере')
                sleep(1)
            else:
                _save_user_photos(frame, user_directory)
                sleep(1)
                _photos_list.append(frame)
    cap.release()
    cap.open(0)
