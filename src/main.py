import configparser

import dearpygui.dearpygui as dpg
import cv2
import numpy as np
import math
import os
from objloader_simple import *

# ------------------ Config ------------------

config = configparser.ConfigParser()
config.read('./src/config.conf')

pair = config.get('Default', 'pair')
reference_path = config.get('path', 'reference_path')
reference_name = config.get('Default', 'reference_name')
reference = reference_path + reference_name
model_path = config.get('path', 'model_path')
model_name = config.get('Default', 'model_name')
input_model_name = model_name

Draw_rectangle = config.getboolean('Default', 'draw_rectangle')
Colordetection = config.getboolean('Default', 'colordetection')
# Minimum number of matches that have to be found
# to consider the recognition valid
MIN_MATCHES = config.getint('Default', 'min_matches')
Draw_Matches = config.getint('Default', 'draw_matches')

# Max Queue Length
Q_LEN = config.getint('Default', 'q_len')

frame_size_width = config.getint('window', 'frame_size_width')
frame_size_height = config.getint('window', 'frame_size_height')
frame_matches_size_width = config.getint('window', 'frame_matches_size_width')
frame_matches_size_height = config.getint('window', 'frame_matches_size_height')
window_width = frame_size_width + frame_matches_size_width + 45 + 600
window_height = frame_matches_size_height + 512

dpg.create_context()
dpg.create_viewport(title='Augmented reality application', width=window_width, height=window_height)
dpg.setup_dearpygui()

# ------------------ uitls ------------------

def render(img, obj, projection, model, color=False):
    """
    Render a loaded obj model into the current video frame
    """
    vertices = obj.vertices
    scale_matrix = np.eye(3) * 3
    h, w = model.shape

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        if color is False:
            cv2.fillConvexPoly(img, imgpts, (137, 27, 211))
        else:
            color = hex_to_rgb(face[-1])
            color = color[::-1]  # reverse
            cv2.fillConvexPoly(img, imgpts, color)

    return img

def projection_matrix(camera_parameters, homography):
    """
    From the camera calibration matrix and the estimated homography
    compute the 3D projection matrix
    """
    # Compute rotation along the x and y axis as well as the translation
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]
    # normalise vectors
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l
    # compute the orthonormal basis
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)
    # finally, compute the 3D projection matrix from the model to the current frame
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    return np.dot(camera_parameters, projection)

def hex_to_rgb(hex_color):
    """
    Helper function to convert hex strings to RGB
    """
    hex_color = hex_color.lstrip('#')
    h_len = len(hex_color)
    return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))

def processing_frame_data(frame):
    data = np.flip(frame, 2)
    data = data.ravel()
    data = np.asfarray(data, dtype='f')

    return(np.true_divide(data, 255.0))

# ------------------ GUI Callback ------------------

def if_draw_rec(sender, data):
    global Draw_rectangle
    Draw_rectangle = data

def if_color_dec(sender, data):
    global Colordetection
    Colordetection = data

def input_model(sender, data):
    global input_model_name
    input_model_name = data

def read_model(sender, data):
    global input_model_name, model_name, if_file_exist
    if os.path.isfile(model_path + input_model_name):
        print("file exists")
        model_name = input_model_name
    else:
        print("no such file")

# ------------------ Main ------------------

if __name__ == '__main__':

    HomoQueue = []
    last_homo = None
    # matrix of camera parameters (made up but works quite well for me)
    camera_parameters = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
    # create ORB keypoint detector
#    sift = cv2.xfeatures2d.SIFT_create()
    orb = cv2.ORB_create()
    # create BFMatcher object based on hamming distance
#    if pair == "sift":
#        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    if pair == "orb":
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    # load the reference surface that will be searched in the video stream
    dir_name = os.getcwd()
    model = cv2.imread(os.path.join(dir_name, reference), 0)
    # Compute model keypoints and its descriptors
#    if pair == "sift":
#        kp_model, des_model = sift.detectAndCompute(model, None)
    if pair == "orb":
        kp_model, des_model = orb.detectAndCompute(model, None)


    # init video capture
    cap = cv2.VideoCapture(1)
    ret, frame = cap.read()
    
    frame_matches = cv2.drawMatches(model, kp_model, frame, None, None, 0, flags=2)

    frame = cv2.resize(frame, dsize=(frame_size_width, frame_size_height))
    frame_matches = cv2.resize(frame_matches, dsize=(frame_matches_size_width, frame_matches_size_height))

    texture_data = processing_frame_data(frame)
    matches_data = processing_frame_data(frame_matches)

    with dpg.texture_registry(show=True):
        dpg.add_raw_texture(frame.shape[1], frame.shape[0], texture_data,
            tag="texture_tag", format=dpg.mvFormat_Float_rgb)
        dpg.add_raw_texture(frame_matches.shape[1], frame_matches.shape[0], matches_data,
            tag="matches_tag", format=dpg.mvFormat_Float_rgb)

    with dpg.window(label="AR", pos=(0, 0)):
        dpg.add_text("Hello, world")
        dpg.add_image("texture_tag")

    with dpg.window(label="Show Matches", pos=(frame_size_width + 13, 0)):
        dpg.add_text("Hello, world")
        dpg.add_image("matches_tag")

    with dpg.window(label="Configs", pos=(frame_size_width + 13, frame_matches_size_height + 58), width=frame_matches_size_width + 16, 
        height=frame_matches_size_height):
        dpg.add_text("Project-CV_996  by  Yuqi Zhou & Yiheng Jiang")

        dpg.add_checkbox(label="draw rectangle delimiting target surface on frame", callback=if_draw_rec)
        dpg.add_checkbox(label="if colordetection", callback=if_color_dec)

        dpg.add_input_text(label="read_objmodel", default_value=model_name, callback=input_model)
        dpg.add_button(label="confirm", callback=read_model)

    dpg.show_metrics()
    dpg.show_viewport()

    while dpg.is_dearpygui_running():

        # Load 3D model from OBJ file
        model_out = model_path + model_name
        obj = OBJ(os.path.join(dir_name, model_out), swapyz=True)

        # Normalization
        v = np.array(obj.vertices)
        N = v.size
        center = v.mean(0)
        scale = np.max(np.abs(v - center))
        v = v - center
        v = v / scale * 100
        obj.vertices = v.tolist()

        # read the current frame
        ret, frame = cap.read()
   
        matches = []
        if not ret:
            print ("Unable to capture video")

        # find and draw the keypoints of the frame
        if Colordetection:
            blur = cv2.GaussianBlur(frame, (7, 7), 0)
            hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
            # standard: (35, 43, 46) -> (77, 255, 255)
            low_green = (35, 43, 46)
            high_green = (90, 255, 255)
            clmask = cv2.inRange(hsv, low_green, high_green)
            frame = cv2.add(np.zeros(np.shape(frame), dtype=np.uint8), frame, mask = clmask)
            frame = 255 - frame

#        if pair == "sift":
#            kp_frame, des_frame = sift.detectAndCompute(frame, None)
        if pair == "orb":
            kp_frame, des_frame = orb.detectAndCompute(frame, None)    
        # match frame descriptors with model descriptors

        if kp_frame:
            matches = bf.match(des_model, des_frame)
            # sort them in the order of their distance
            # the lower the distance, the better the match
            matches = sorted(matches, key=lambda x: x.distance)

        frame_matches = cv2.drawMatches(model, kp_model, frame, kp_frame, matches[:Draw_Matches], 0, flags=2)

        # compute Homography if enough matches are found
        if len(matches) > MIN_MATCHES:
            #print(len(matches))
            # differenciate between source points and destination points
            src_pts = np.float32([kp_model[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            # compute Homography
            homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if Draw_rectangle:
                # Draw a rectangle that marks the found model in the frame
                h, w = model.shape
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                # project corners into frame
                dst = cv2.perspectiveTransform(pts, homography)
                # connect them with lines  
                frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)  
            # if a valid homography matrix was found render cube on model plane

            if homography is not None:
                try:
                    if len(HomoQueue) == Q_LEN:
                        HomoQueue.pop(0)
                    HomoQueue.append(homography)

                    for hm in HomoQueue[0: -1]:
                        homography = homography + hm
                    homography = homography / Q_LEN

                    # obtain 3D projection matrix from homography matrix and camera parameters
                    projection = projection_matrix(camera_parameters, homography)
                    # project cube or model
                    frame = render(frame, obj, projection, model, False)
                    #frame = render(frame, model, projection)

                except:
                    pass

        frame = cv2.resize(frame, dsize=(frame_size_width, frame_size_height))
        frame_matches = cv2.resize(frame_matches, dsize=(frame_matches_size_width, frame_matches_size_height))

        texture_data = processing_frame_data(frame)
        matches_data = processing_frame_data(frame_matches)
        dpg.set_value("texture_tag", texture_data)
        dpg.set_value("matches_tag", matches_data)

        dpg.render_dearpygui_frame()


    cap.release()
    dpg.destroy_context()

