# -*- coding: utf-8 -*-
import cv2
import dlib
import numpy as np
import pickle
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from re import search
from sklearn.preprocessing import LabelEncoder
import datetime
from os.path import getmtime


labels = pd.read_csv('/home/malov/openface/generated-embeddings/labels.csv', header=None).as_matrix()[:, 1]
file_name = '/home/malov/PycharmProjects/FaceRecognition/Classifier/classifier.pkl'


def _train(labels, reps):
    name_labels = [search('(?<=./aligned-images/)\w+', labels[i]).group(0) for i in range(len(labels))]
    le = LabelEncoder().fit(name_labels)
    num_labels = le.transform(name_labels)

    # zero_user = np.zeros((10, 128))
    # ones_user = np.ones((10,128))
    # X = np.concatenate((reps, zero_user))
    # X = np.concatenate((X, ones_user))
    # y = np.array([0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,3,3,3,3,3,3,3,3,3,3])

    clf = DecisionTreeClassifier()
    clf.fit(reps, num_labels)
    with open(file_name, 'w') as f:
        pickle.dump((le, clf), f)

    # param_grid = [
    #     {'C': [1, 10, 100, 1000],
    #      'kernel': ['linear']},
    #     {'C': [1, 10, 100, 1000],
    #      'gamma': [0.001, 0.0001],
    #      'kernel': ['rbf']}
    # ]

    # X_pca = PCA(n_components=50).fit_transform(X, X)
    # tsne = TSNE(n_components=2, init='random', random_state=0)
    # X_r = tsne.fit_transform(X_pca)

    # svm = GridSearchCV(SVC(C=1), param_grid, cv=5).fit(X, y)

    # labels_new = []
    # for i in range(10):
    #     labels_new.append(name)
    # labels = np.array(labels)
    # le = LabelEncoder().fit(labels)
    # labels_num = le.transform(labels)
    # clf = SVC(C=1, kernel='linear', probability=True)
    # clf.fit(reps, labels_num)
    # return le, clf

def _camera_frame(index):
    text = 'malov'
    font = cv2.FONT_HERSHEY_SIMPLEX
    faceCascade = cv2.CascadeClassifier('/home/malov/PycharmProjects/object_detection/cascade_face_detection.xml')
    cap = cv2.VideoCapture(index)
    while True:
        ret, frame = cap.read()
        cv2.imshow('Terminal1', frame) if ret else 'Camera is not active, choose another terminal'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, text, (int((x + w) / 2), int((y + h) / 2)), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
            aligned_frame = frame[y:(y + w), x:(x + h)]

    cap.release()
    cv2.destroyAllWindows()

def _infer(index):
    # once a day clf is downloaded from db

    if not (datetime.datetime.now().day - datetime.datetime.fromtimestamp(getmtime(file_name)).day) == 0:
        with open('/home/malov/openface/generated-embeddings/classifier_new_version.pkl', 'r') as f:
            (le, clf) = pickle.load(f)
    else:
        with open('/home/malov/openface/generated-embeddings/classifier_old_version.pkl', 'r') as f:
            (le, clf) = pickle.load(f)
    cap = cv2.VideoCapture(index)
    while True:
        ret, frame = cap.read()
        reps = []
        if ret:
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

def _save_clf(le, clf):
    with open('/home/malov/PycharmProjects/object_detection/classifier.pkl', 'w') as raw_file:
        pickle.dump((le, clf), raw_file)

#
# if __name__ == '__main__':
#     name = 'malov '
#     # reps = _get_reps(0)
#     # le, clf = _train(name, reps)
#     # _save_clf(le, clf)
#
#     reps = _get_reps(0)
#     # reps_vert = np.vstack(reps)
#     print(reps.shape)
#     clf = _train(reps)
#     confidence = _infer(0, clf)
#     print(confidence)
