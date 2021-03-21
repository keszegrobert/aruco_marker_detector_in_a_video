# The original code was written by Sunita Nayak at BigVision LLC. It is based on the OpenCV project. It is subject to the license terms in the LICENSE file found in this distribution and at http://opencv.org/license.html

# Usage example:   python3 ar_with_aruco.py --video=ar_in.avi

import cv2 as cv

print(cv.__version__)

import argparse
import sys
import os.path
import numpy as np
import json
winName = "Augmented Reality using Aruco markers in OpenCV"

parser = argparse.ArgumentParser(description='Augmented Reality using Aruco markers in OpenCV')
parser.add_argument('--video', default='ar_in.avi', help='Path to the input video file.')
args = parser.parse_args()

if not os.path.isfile(args.video):
    print("Input video file ", args.video, " doesn't exist")
    parser.print_help()
    sys.exit(1)
cap = cv.VideoCapture(args.video)

# Initialize the detector parameters - picked a working combination from millions of random examples
parameters =  cv.aruco.DetectorParameters_create()
parameters.minDistanceToBorder =  7
parameters.cornerRefinementMaxIterations = 149
parameters.minOtsuStdDev= 4.0
parameters.adaptiveThreshWinSizeMin= 7
parameters.adaptiveThreshWinSizeStep= 49
parameters.minMarkerDistanceRate= 0.014971725679291437
parameters.maxMarkerPerimeterRate= 10.075976700411534 
parameters.minMarkerPerimeterRate= 0.2524866841549599 
parameters.polygonalApproxAccuracyRate= 0.05562707541937206
parameters.cornerRefinementWinSize= 9
parameters.adaptiveThreshConstant= 9.0
parameters.adaptiveThreshWinSizeMax= 369
parameters.minCornerDistanceRate= 0.09167132584946237

#Load the dictionary that was used to generate the markers.
dictionary = cv.aruco.Dictionary_get(cv.aruco.DICT_5X5_50)

detected_markers={}
frame_counter = 0
while cv.waitKey(1) < 0:
    frame_counter += 1
    try:
        # get frame from the video
        hasFrame, frame = cap.read()
        # Stop the program if reached end of video
        if not hasFrame:
            print("Done processing !!!")
            cv.waitKey(3000)
            break
                
        # Detect the markers in the image
        markerCorners, markerIds, rejectedCandidates = cv.aruco.detectMarkers(frame, dictionary, parameters=parameters)
        print('frame: {} ids: {}'.format(frame_counter, markerIds.tolist()))
        im_out = cv.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)

        # Showing the original image with the markers drawn on it
        cv.imshow(winName, im_out.astype(np.uint8))

    except Exception as inst:
        print(inst)

cv.waitKey(3000)
cv.destroyAllWindows()