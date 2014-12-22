# import the necessary packages
import numpy as np
import numpy.linalg as la
import argparse
import cv2
from pymouse import PyMouse

# define the upper_threshold and lower_threshold boundaries of the HSV pixel
# intensities to be considered 'skin'
lower_threshold = np.array([0, 40, 50], dtype="uint8")
upper_threshold = np.array([22, 255, 255], dtype="uint8")

# keep track of previous 15 frames worth of finger counts
# so it appears smooth over time
num_defects = np.zeros(15, dtype="uint8")
num_defects_counter = 0
detect_gestures = False
moving_time = 0

# keep track of previous 5 frames worth of bounding box coordinates
# so there are no random jumps over time
prev_bounding_boxes = [None] * 5
prev_bounding_boxes_counter = 0
check_bounding_box = False

# keep track of the mouse object and position
mouse = PyMouse()
mouse_position = mouse.position()
prev_average = (-1, -1)

# keep track of previous number of fingers for gestures
# that require you to detect finger count changes
prev_num_fingers = -1

# do initial hue detection and setup the hue thresholds


def hue_detect(camera):
    # setup region that is going to detect hue
    sample_points = []
    for i in range(30, 45):
        for j in range(20, 40):
            sample_points.append((i * 10, j * 10))

    sample_colors = np.zeros((len(sample_points), 3))

    # show rectangles until user has placed hand over region
    while True:
        _, frame = camera.read()

        for top_left in sample_points:
            # calculate bottom right of bounding box
            bottom_right = tuple([(x + 10) for x in top_left])
            # mark region to user
            cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 2)

        cv2.imshow('Place fingers in region', frame)

        # if the 'q' key is pressed, stop the loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # take 25 frames to get average HSV values
    for i in range(0, 25):
        _, frame = camera.read()

        # convert to HSV
        converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        sample_count = 0
        for point in sample_points:
            # add add current frame's color values for each sample
            sample_colors[sample_count] = np.sum(
                [sample_colors[sample_count], converted[point]],
                axis=0)
            # continue marking region to user (different color)
            cv2.rectangle(
                frame, point, tuple([(x + 10) for x in point]), (0, 255, 0), 2)
            sample_count = sample_count + 1

        cv2.imshow('Calculating color', frame)

        # if the 'q' key is pressed, stop the loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()

    # zip the sample colors before doing percentile tests
    sample_colors = [[(y / 25) for y in x] for x in sample_colors]
    hand_zipped = zip(*sample_colors)

    # modify final threshold
    global lower_threshold
    global upper_threshold

    lower_threshold[0] = min(0, np.percentile(hand_zipped[0], 5) * 0.5)
    upper_threshold[0] = min(255, np.percentile(hand_zipped[0], 90) * 2.0)

# preprocess a frame: segment out skin colored objects


def preprocess_frame(frame):
    # get frame shape
    height, width, _ = frame.shape

    # resize the frame, convert it to the HSV color space,
    # and determine the HSV pixel intensities that fall into
    # the speicifed upper_threshold and lower_threshold boundaries
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skin_mask = cv2.inRange(converted, lower_threshold, upper_threshold)

    # apply a series of erosions and dilations to the mask
    # using an elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skin_mask = cv2.erode(skin_mask, kernel, iterations=2)
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)

    # blur the mask to help remove noise, then apply the
    # mask to the frame
    skin_mask = cv2.GaussianBlur(skin_mask, (1, 1), 0)
    skin = cv2.bitwise_and(frame, frame, mask=skin_mask)

    # convert skin to grayscale
    skin_bgr = cv2.cvtColor(skin, cv2.COLOR_HSV2BGR)
    skin_gray = cv2.cvtColor(skin_bgr, cv2.COLOR_BGR2GRAY)

    # threshold the image
    ret, thresh = cv2.threshold(
        skin_gray, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # return the final skin and thresholded image
    return skin, thresh


def find_max_contour(frame):
    # find the contours
    contours, hierarchy = cv2.findContours(
        frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # find number of pixels in frame
    height, width = frame.shape
    frame_size = height * width

    # iterate over all contours and only accept contours with sufficient area
    contour = None
    contour_index = -1
    max_contour_area = 0

    for i in range(len(contours)):
        contour = contours[i]
        area = cv2.contourArea(contour)

        # ensure contours only wrap the hand or face
        # by comparing against (0.80 * frame_size)
        if(area > max_contour_area and area < 0.80 * frame_size):
            max_contour_area = area
            contour_index = i

    # return contour with the maximum area
    if (contour_index == -1):
        return None
    else:
        return contours[contour_index]

# this function filters the defects of a given contour
# the purpose is to only keep finger defects


def filter_defects(defects, contour, frame):
    # defects are invalid until proven valid
    defects_filtered = []
    if defects is not None:
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]

            # decode defect
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])

            # make two lines: point to start and point to end
            v1 = np.array([start[0] - far[0], start[1] - far[1]])
            v2 = np.array([end[0] - far[0], end[1] - far[1]])

            # get angle between lines
            cos_ang = np.dot(v1, v2)
            sin_ang = la.norm(np.cross(v1, v2))
            ang = np.arctan2(sin_ang, cos_ang)

            # check if the defect violates box continuity
            valid_bounding_box = True
            if check_bounding_box:
                for bb in prev_bounding_boxes:
                    if far[0] < bb[0] or far[0] > bb[0] + \
                            bb[2] or far[1] < bb[1] or far[1] > bb[1] + bb[3]:
                        valid_bounding_box = False

            if valid_bounding_box and ang < 1.22 and la.norm(
                    v1) > 75 and la.norm(v2) > 75:
                cv2.circle(frame, far, 5, [0, 0, 255], -1)
                defects_filtered.append(far)

    return frame, defects_filtered

# finds the convexity defects given a contour


def find_convexity_defects(contour, frame):
    # find hull of the contour and find all hull defects
    hull = cv2.convexHull(contour)
    hull_defects = cv2.convexHull(contour, returnPoints=False)

    # find convexity defects
    defects = cv2.convexityDefects(contour, hull_defects)

    # draw bounding box around contour for feedback
    x, y, l, w = cv2.boundingRect(contour)
    cv2.rectangle(frame, (x, y), (x + w, y + l), (255, 0, 0), 2)

    # filter the defects so only the finger defects remain
    frame, defects_filtered = filter_defects(defects, contour, frame)

    # return the modified frame and the filtered defects and the hull bounding
    # box
    return frame, defects_filtered, (x, y, l, w)

# uses all information to detect a gesture and then calls PyUserInput to
# activate it


def detect_and_activate_gestures(defects_filtered, hull_bounds):
    # declare the globals we are going to modify
    global num_defects_counter
    global detect_gestures
    global prev_bounding_boxes_counter
    global check_bounding_box
    global prev_num_fingers
    global moving_time
    global mouse_position
    global prev_average

    # update the circular buffer which tracks the number of defects
    num_defects[num_defects_counter] = len(defects_filtered)
    num_defects_counter = num_defects_counter + 1
    if num_defects_counter >= 15:
        detect_gestures = True
        num_defects_counter = 0

    # update the circular buffer which tracks the sizes of previous hull bounds
    prev_bounding_boxes[prev_bounding_boxes_counter] = hull_bounds
    prev_bounding_boxes_counter = prev_bounding_boxes_counter + 1
    if prev_bounding_boxes_counter >= 5:
        check_bounding_box = True
        prev_bounding_boxes_counter = 0

    # round the median number of fingers in the last
    # 5 frames
    num_fingers = round(np.median(num_defects))

    # disable gestures while finger count is warming up
    if detect_gestures:
        # four fingers up corresponds to an open palm
        if num_fingers == 4:
            average = (np.average([x[0] for x in defects_filtered]),
                       np.average([x[1] for x in defects_filtered]))

            if prev_average[0] == -1:
                prev_average = average

            # calculate new position of the cursor
            # take into account number of frames mouse has been moving for
            #   to emulate acceleration
            mouse_diff = tuple(
                map(lambda x, y: (x - y) * moving_time / 4,
                    prev_average,
                    average))
            if abs(mouse_diff[0]) < moving_time * \
                    4 and abs(mouse_diff[1]) < moving_time * 4:
                mouse_position = map(sum, zip(mouse_position, mouse_diff))

            # move the cursor
            mouse.move(mouse_position[0], mouse_position[1])

            prev_average = average

# cursor acceleration
            if moving_time < 16:
                moving_time = moving_time + 2
        else:
            prev_average = (-1, -1)
            moving_time = 0

            # get the most recent position
            mouse_position = mouse.position()

            # if the number of fingers goes from 1 to 0,
            #   click the mouse button
            if num_fingers == 0 and prev_num_fingers == 1:
                mouse.click(mouse_position[0], mouse_position[1], 1)

            if num_fingers == 0 and prev_num_fingers == 2:
                mouse.click(mouse_position[0], mouse_position[1], 2)

    # update the number of fingers in the last frame
    prev_num_fingers = num_fingers


# main loop
def main():
    # create video capture
    camera = cv2.VideoCapture(0)

    # do initial hue detection
    hue_detect(camera)

    # keep looping over the frames in the video
    while True:
        # grab the current frame
        _, raw_frame = camera.read()

        # get frame shape
        height, width, _ = raw_frame.shape

        # preprocess the frame
        skin_frame, preprocessed_frame = preprocess_frame(raw_frame)

        # find a contour with maximum idea
        max_contour = find_max_contour(preprocessed_frame)

        # if we found a valid contour, do the finger tracking and mapping
        if max_contour is not None:

            # find all valid defects in the contour
            skin_frame, defects_filtered, hull_bounds = find_convexity_defects(
                max_contour, skin_frame)

            # detect and activate gestures
            detect_and_activate_gestures(defects_filtered, hull_bounds)

        # if the 'q' key is pressed, stop the loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # show the skin in the image
        cv2.imshow("Demo", skin_frame)

    # cleanup the camera and close any open windows
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
