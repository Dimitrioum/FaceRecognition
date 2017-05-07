import dlib



FACE_SHAPE_PREDICT_MODEL = '/home/malov/PycharmProjects/object_detection/shape_predictor_68_face_landmarks.dat'
FACE_128_CHARS_MODEL     = '/home/malov/PycharmProjects/object_detection/dlib_face_recognition_resnet_model_v1.dat'

FACE_DETECTOR               = dlib.get_frontal_face_detector()
FACE_SHAPE_PREDICT          = dlib.shape_predictor(FACE_SHAPE_PREDICT_MODEL)
FACE_128_CHARS_RECOGNIZE    = dlib.face_recognition_model_v1(FACE_128_CHARS_MODEL)