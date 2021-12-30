


import argparse

import cv2
import numpy as np
import math
import os
from objloader_simple import *

pair = "orb"

reference = 'reference/model2.png'

model_out = 'models/fox.obj'

Colordetection = False
# Minimum number of matches that have to be found
# to consider the recognition valid
MIN_MATCHES = 10

# Max Queue Length
Q_LEN = 10


def main():
    """
    This functions loads the target surface image,
    """
    HomoQueue = []
    last_homo = None
    # matrix of camera parameters (made up but works quite well for me)
    camera_parameters = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
    # create ORB keypoint detector
    sift = cv2.xfeatures2d.SIFT_create()
    orb = cv2.ORB_create()
    # create BFMatcher object based on hamming distance
    if pair == "sift":
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    if pair == "orb":
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    # load the reference surface that will be searched in the video stream
    dir_name = os.getcwd()
    model = cv2.imread(os.path.join(dir_name, reference), 0)
    # Compute model keypoints and its descriptors
    if pair == "sift":
        kp_model, des_model = sift.detectAndCompute(model, None)
    if pair == "orb":
        kp_model, des_model = orb.detectAndCompute(model, None)
    # Load 3D model from OBJ file
    obj = OBJ(os.path.join(dir_name, model_out), swapyz=True)
    # init video capture
    cap = cv2.VideoCapture(1)

    while True:
        # read the current frame
        ret, frame = cap.read()
        matches = []
        if not ret:
            print ("Unable to capture video")
            return
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

        if pair == "sift":
            kp_frame, des_frame = sift.detectAndCompute(frame, None)
        if pair == "orb":
            kp_frame, des_frame = orb.detectAndCompute(frame, None)    
        # match frame descriptors with model descriptors
        
        if kp_frame:

            matches = bf.match(des_model, des_frame)
            # sort them in the order of their distance
            # the lower the distance, the better the match
            matches = sorted(matches, key=lambda x: x.distance)
        
        # compute Homography if enough matches are found
        if len(matches) > MIN_MATCHES:
            #print(len(matches))
            # differenciate between source points and destination points
            src_pts = np.float32([kp_model[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            # compute Homography
            homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if args.rectangle:
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

                    print(homography)



                except:
                    pass

            # draw first 10 matches.
            if args.matches:
                frame = cv2.drawMatches(model, kp_model, frame, kp_frame, matches[:10], 0, flags=2)
            # show result
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            print ("Insufficient matches - %d/%d" % (len(matches), MIN_MATCHES))

    cap.release()
    cv2.destroyAllWindows()
    return 0

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


# Command line argument parsing

parser = argparse.ArgumentParser(description='Augmented reality application')

parser.add_argument('-r','--rectangle', help = 'draw rectangle delimiting target surface on frame', action = 'store_true')
parser.add_argument('-mk','--model_keypoints', help = 'draw model keypoints', action = 'store_true')
parser.add_argument('-fk','--frame_keypoints', help = 'draw frame keypoints', action = 'store_true')
parser.add_argument('-ma','--matches', help = 'draw matches between keypoints', action = 'store_true')

args = parser.parse_args()

if __name__ == '__main__':
    main()
