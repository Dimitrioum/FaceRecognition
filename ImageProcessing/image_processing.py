# -*- coding: utf-8 -*-
"""
Methods for image processing
"""
import numpy as np
import cv2
from Constants.constants import FACE_DETECTOR, FACE_SHAPE_PREDICT, \
    FACE_128_CHARS_RECOGNIZE, INNER_EYES_AND_BOTTOM_LIP, MINMAX_TEMPLATE


def get_face_detection(frame):
    """
    Detects faces on the image
    :param frame: input image
    :type frame: <type 'numpy.ndarray'>
    :return: enumerate list of rectangles of detected faces
    """

    detection = FACE_DETECTOR(frame, 1)
    if len(detection):
        return enumerate(detection)
    else:
        pass


def get_face_shape_coordinates(frame):
    """
    Gets coordinates of 68 points of face shape
    :param frame: input image
    :type frame: <type 'numpy.ndarray'>
    :return: list with 68-points  coordinates (x,y)
    """

    face_detection = get_face_detection(frame)
    for index, face_rectangle in face_detection:
        face_shape = FACE_SHAPE_PREDICT(frame, face_rectangle)
        return list(map(lambda part: (part.x, part.y), face_shape.parts()))


def _get_aligned_frame(frame):
    """
    Gets aligned frame with centered shape of face
    :param frame: input image
    :type frame: <type 'numpy.ndarray'>
    :return: aligned frame
    """

    face_shape_coordinates = np.float32(get_face_shape_coordinates(frame))
    face_shape_coordinates_indices = np.array(INNER_EYES_AND_BOTTOM_LIP)
    affine_transform = cv2.getAffineTransform(face_shape_coordinates[face_shape_coordinates_indices],
                                              96 * MINMAX_TEMPLATE[face_shape_coordinates_indices])
    return cv2.warpAffine(frame, affine_transform, (96, 96))  # type: np.array, like a frame


def get_128_face_chars(frame):
    """
    Gets 128 unique characters of human face
    :param frame: input image
    :type frame: <type 'numpy.ndarray'>
    :return: list of 128 unique characters of human face
    """

    try:
        for index, face_rectangle in get_face_detection(frame):
            face_shape = FACE_SHAPE_PREDICT(_get_aligned_frame(frame), face_rectangle)
            face_128_chars = FACE_128_CHARS_RECOGNIZE.compute_face_descriptor(_get_aligned_frame(frame),
                                                                              face_shape)
            return list(face_128_chars)  # 128 characters
    except TypeError:
        pass


if __name__ == '__main__':
    # cap = cv2.VideoCapture(0)
    # ret, frame = cap.read()
    # if ret:
    #     print(get_face_detection(frame))
    print(globals())
