# -*- coding: utf-8 -*-
"""
Classifier module allows training exact classifier, using training set, searching for the best parameters,
inferring person and confidence of the prediction
"""

import os
import re
import csv
import cv2
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from ImageProcessing.image_processing import get_128_face_chars
from Constants.constants import CLASSIFIER_PATH, WORKING_DIRECTORY, TRAINING_DATASET_CSV
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd


def _get_row_string_from_csv(index, train_set):
    """
    :param index: index of the analysing photo
    :type: index: int
    :param train_set: training data set of 128 face chars
    :type train_set: <class 'pandas.core.frame.DataFrame'>
    :return: list of strings of 128 face chars of the analysing photo
    """

    list_of_strings = re.split('\[|,|\]', train_set.values[index][0])
    return list_of_strings


def _get_list_chars_from_string_csv(train_set):
    """
    :param train_set: training data set of 128 face chars
    :type train_set: <class 'pandas.core.frame.DataFrame'>
    :return: list of lists of 128 face chars of all users' photos
    """

    faces_128_chars = []
    for index in range(len(train_set.values)):
        _chars_row_set = []
        for string_char in _get_row_string_from_csv(index, train_set):
            if string_char:
                _chars_row_set.append(float(string_char))
        faces_128_chars.append(_chars_row_set)
        del _chars_row_set

    return faces_128_chars


def choose_classifier(method_name):
    """
    :param method_name: name of the chosen classifier to train
    :return: best parameters for training of the chosen classifier
    """

    training_dataset = pd.read_csv(TRAINING_DATASET_CSV)
    training_dataset.dropna(axis=0, how='any', inplace=True)
    labels = training_dataset['user_name_label']
    training_dataset.drop(['user_name_label'],
                          axis=1,
                          inplace=True)
    faces_128_chars = _get_list_chars_from_string_csv(training_dataset)
    best_classifier = ClassifierSearch(method_name=method_name,
                                       X_dataset=faces_128_chars,
                                       labels=labels)
    best_classifier.dispatch()
    best_classifier.save_classifier()
    # print(best_classifier.grid_search())


class ClassifierSearch(object):
    """
    ChooseClassifier class gets name of the chosen classifier, training and test data sets and finds the best parameters
    for the chosen classifier. 
    """

    def __init__(self, method_name, X_dataset, labels):
        """
        :param method_name: name of the chosen classifier
        :type: method_name: str
        :param X_dataset: dataset of 128 characters of every photo in WORKING_DIRECTORY
        :type X_dataset: <type 'numpy.ndarray'>
        :param labels: name of directories into the WORKING_DIRECTORY
        :type labels: <class 'pandas.core.series.Series'>
        """

        self.people_name_labels = [photo_path.split('/')[4] for photo_path in _photos_list_object()]
        self.method_name = method_name
        self.classifier = None
        self.label_encoder = LabelEncoder().fit(self.people_name_labels)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_dataset, labels,
                                                                                test_size=0.3,
                                                                                random_state=17)

    def _svm_classifier(self):
        """
        Support Vector Machines method
        :return: SVM classifier
        """

        self.classifier = SVC()
        return self.classifier

    def _tree_classifier(self):
        """
        Decision Tree method
        :return: DecisionTree classifier
        """

        self.classifier = DecisionTreeClassifier(max_depth=161,
                                                 max_features=4)
        return self.classifier

    def _neighbors_classifier(self):
        """
        KNearestNeighbours method
        :return: kNN classifier
        """

        self.classifier = KNeighborsClassifier()
        return self.classifier

    def save_classifier(self):
        """
        Saves trained classifier and label encoder at exact directory
                (might be not the same as working directory)
        """

        with open(CLASSIFIER_PATH, 'w') as clf:
            pickle.dump((self.label_encoder, self.classifier), clf)

    def dispatch(self):
        """
        Dispatcher for choosing the exact classifier
        :return: chosen classifier
        """

        classifier_name = '_' + str(self.method_name) + '_' + 'classifier'
        get_exact_classifier = getattr(self, classifier_name)
        self.classifier = get_exact_classifier()
        self.classifier.fit(self.X_train, self.y_train)

    def grid_search(self):
        """
        Searching for the best classifier parameters
        :param classifier: chosen classifier
        :type classifier: <class 'sklearn../name of the class/..'>
        :return: best parameters for the chosen classifier
        """

        def get_classifier_params(classifier):
            """
            :param classifier: chosen classifier
            :type classifier: <class 'sklearn../name of the class/..'>
            :return: parameters dictionary of chosen classifier for Grid Search module
            """

            classifier_type = classifier.__module__.split('.')[1]
            classifier_params_dict = {'tree': {'max_depth': range(1, 200),
                                               'max_features': range(1, 80)},

                                      'neighbors': {'n_neighbors': range(1, 210)},

                                      'svm': {'kernel': ('linear', 'rbf'),
                                              'C': [0.01, 0.1, 1, 10, 100, 1000]}
                                      }
            classifier_params = classifier_params_dict[classifier_type]
            return classifier_params

        classifier_params = get_classifier_params(self.classifier)
        classifier_grid = GridSearchCV(self.classifier, classifier_params,
                                       cv=5, n_jobs=-1, verbose=True)
        print(self.X_train, self.y_train, self.X_test, self.y_test)
        classifier_grid.fit(self.X_train, self.y_train)
        prediction_accuracy = accuracy_score(self.y_test, classifier_grid.predict(self.X_test)) * 100

        return '''Best parameters for {} classifier are: {}.
                  Best score on the training data: {} %.
                  Accuracy score on the test data: {} %.'''.format(self.classifier,
                                                                   classifier_grid.best_params_,
                                                                   classifier_grid.best_score_ * 100,
                                                                   prediction_accuracy)


def _photos_list_object():
    """
    :return: generator-object of users' photos list
    """

    for path, _, file_names in os.walk(WORKING_DIRECTORY):
        for file_name in file_names:
            yield os.path.join(path, file_name)


def _get_users_128_chars():
    """
    :return: array of 128 face characters of previously saved every user's photo
             within working directory
    """

    # photos_number = len(list(_photos_list_object()))
    # faces_128_chars = np.zeros(shape=(photos_number, 128))
    faces_128_chars = []
    for photo in _photos_list_object():
        frame = cv2.imread(photo)
        faces_128_chars.append(get_128_face_chars(frame))
    return faces_128_chars


class Classifier(object):
    """
    :return: trained classifier and label encoder for
             training set of users' photos
    """

    def __init__(self):
        self.working_directory = WORKING_DIRECTORY
        self.person_name = None
        self.confidence = None
        self.classifier = None
        self.people_name_labels = [photo_path.split('/')[4] for photo_path in _photos_list_object()]
        self.label_encoder = LabelEncoder().fit(self.people_name_labels)
        self.faces_128_chars = _get_users_128_chars()
        self.name_number_labels = self.label_encoder.transform(self.people_name_labels)

    def get_classifier(self):
        """
        previously trained label encoder and classifier
        """

        try:
            with open(CLASSIFIER_PATH, 'r') as clf:
                self.label_encoder, self.classifier = pickle.load(clf)
        except IOError:
            print 'Указан неверный путь классификатора,' \
                  ' либо классификатор не обучен'

    def save_training_set_as_csv(self):
        """
        Saves the training data set of 128 characters of every user photo
        """

        users_dict = {'128_face_chars': _get_users_128_chars(),
                      'user_name_label': self.name_number_labels}
        training_dataframe = pd.DataFrame.from_dict(users_dict)
        training_dataframe.to_csv(TRAINING_DATASET_CSV, index_label=None, index=None)

    def train(self):
        """
        Saving trained classifier
        """

        self.classifier = SVC(C=0.1, kernel='linear', probability=True)
        self.classifier.fit(self.faces_128_chars, self.name_number_labels)

    def infer(self, frame):
        """
        :param frame: camera's frame with face, that should be inferred
        :return: inferred person name on the frame and confidence of the prediction
        """

        try:
            face_128_chars = np.array(get_128_face_chars(frame)).reshape(1, -1)
            predictions = self.classifier.predict_proba(face_128_chars).ravel()
            max_prob = np.argmax(predictions)
            person_name = self.label_encoder.inverse_transform(max_prob)
            confidence = predictions[max_prob]
            return person_name, confidence
        except TypeError:
            pass


if __name__ == '__main__':
    # classifier = Classifier()
    # classifier.save_training_set_as_csv()
    # method_name = 'tree'
    # choose_classifier(method_name)
    classifier = Classifier()
    classifier.get_classifier()
    frames = [cv2.imread('/home/malov/anon5.jpg'), cv2.imread('/home/malov/anon8.jpg'),
              cv2.imread('/home/malov/anon10.jpg'), cv2.imread('/home/malov/dima.jpg')]
    for frame in frames:
        print(classifier.infer(frame))
