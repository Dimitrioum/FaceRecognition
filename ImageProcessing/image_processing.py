# -*- coding: utf-8 -*-
import numpy as np
import cv2
from Constants.constants import FACE_DETECTOR, FACE_SHAPE_PREDICT,\
    FACE_128_CHARS_RECOGNIZE, INNER_EYES_AND_BOTTOM_LIP, MINMAX_TEMPLATE


def _get_face_detection(frame):
    detection = FACE_DETECTOR(frame, 1)
    return enumerate(detection) if len(detection) else None


def _get_face_shape_coordinates(frame):
    for index, face_rectangle in _get_face_detection(frame):
        face_shape = FACE_SHAPE_PREDICT(frame, face_rectangle)
        return list(map(lambda part: (part.x, part.y), face_shape.parts()))


def _get_aligned_frame(frame):
    face_shape_coordinates = np.float32(_get_face_shape_coordinates(frame))
    face_shape_coordinates_indices = np.array(INNER_EYES_AND_BOTTOM_LIP)
    affine_transform = cv2.getAffineTransform(face_shape_coordinates[face_shape_coordinates_indices],
                                              96 * MINMAX_TEMPLATE[face_shape_coordinates_indices])
    return cv2.warpAffine(frame, affine_transform, (96, 96))  # type: np.array, like a frame


def _get_128_face_chars(frame):
    aligned_frame = _get_aligned_frame(frame)
    detections = _get_face_detection(frame)
    for index, face_box in enumerate(detections):
        face_shape = FACE_SHAPE_PREDICT(frame, face_box)
        face_128_chars = FACE_128_CHARS_RECOGNIZE.compute_face_descriptor(aligned_frame,
                                                                         face_shape)
        return np.array(face_128_chars)  # 128 characters


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        print(_get_face_detection(frame))
