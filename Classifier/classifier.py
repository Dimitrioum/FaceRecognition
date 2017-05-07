# -*- coding: utf-8 -*-
import cv2
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from ImageProcessing.image_processing import _get_128_face_chars
import os


class GeneratorLength(object):
    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length


class Classifier:
    def __init__(self, classifier_path, users_directory):
        self.classifier_path = classifier_path
        self.users_directory = users_directory

    def _get_classifier(self):
        with open(self.classifier_path, 'r') as clf:
            self.label_encoder, self.classifier = pickle.load(clf)
            return self.label_encoder, self.classifier

    def _photos_list_object(self):
        for path, names, file_names in os.walk(self.users_directory):
            for file_name in file_names:
                yield os.path.join(path, file_name)

    def _get_users_128_chars(self):
        face_128_chars = np.zeros(shape=(len(GeneratorLength(self._photos_list_object())), 128))
        for index, photo in enumerate(self._photos_list_object()):
            frame = cv2.imread(photo)
            face_128_chars[index] = _get_128_face_chars(frame)
        return face_128_chars

    def _train(self):
        self.people_name_list = os.listdir(self.users_directory)
        self.label_encoder = LabelEncoder().fit(self.people_name_list)
        self.number_labels = self.label_encoder.transform(self.people_name_list)
        self.classifier = DecisionTreeClassifier()
        self.faces_128_chars = self._get_users_128_chars(self._photos_list_object())
        self.classifier.fit(self.faces_128_chars, self.number_labels)
        return self.label_encoder, self.classifier

    def _infer(self, frame):
        self.face_128_chars = np.array(_get_128_face_chars(frame))
        self.predictions = self.classifier.predict_proba(self.face_128_chars).ravel()
        self.max_prob = np.argmax(self.predictions)
        self.person_name = self.label_encoder.inverse_transform(self.max_prob)
        self.confidence = self.predictions[self.max_prob]
        return self.person_name, self.confidence


if __name__ == '__main__':
    import timeit
    print(timeit.timeit('_train()', setup='from __main__ import _train'))

