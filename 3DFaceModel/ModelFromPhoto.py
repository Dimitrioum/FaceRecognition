import eos
from ImageProcessing.image_processing import get_face_shape_coordinates


def create_face_model(frame, directory=):
    """
    Generates 3D-model of face from the only photo
    :param frame: input image
    saves .obj file with 3D face model
    """

    landmarks = get_face_shape_coordinates(frame)
    landmark_ids = list(map(str, range(1, 69)))  # generates the numbers 1 to 68, as strings
    image_width = frame.shape[0]  # Make sure to adjust these when using your own images!
    image_height = frame.shape[1]

    model = eos.morphablemodel.load_model("../share/sfm_shape_3448.bin")
    blendshapes = eos.morphablemodel.load_blendshapes("../share/expression_blendshapes_3448.bin")
    landmark_mapper = eos.core.LandmarkMapper('../share/ibug_to_sfm.txt')
    edge_topology = eos.morphablemodel.load_edge_topology('../share/sfm_3448_edge_topology.json')
    contour_landmarks = eos.fitting.ContourLandmarks.load('../share/ibug_to_sfm.txt')
    model_contour = eos.fitting.ModelContour.load('../share/model_contours.json')

    (mesh, pose, shape_coeffs, blendshape_coeffs) = eos.fitting.fit_shape_and_pose(model, blendshapes,
                                                                                   landmarks, landmark_ids,
                                                                                   landmark_mapper,
                                                                                   image_width, image_height,
                                                                                   edge_topology, contour_landmarks,
                                                                                   model_contour)
    eos.core.write_obj(mesh, 'dima_face_model.obj')
