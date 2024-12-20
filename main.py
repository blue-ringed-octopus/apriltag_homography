# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 13:38:05 2024

@author: hibado
"""

import cv2
import pyrealsense2 as rs
from pupil_apriltags import Detector
import numpy as np
import os
import matplotlib.pyplot as plt
import colorsys 
os.add_dll_directory(r"C:\Users\hibad\anaconda3\lib\site-packages\pupil_apriltags.libs")
cv2.destroyAllWindows()
WINDOW_SCALE=1
screenWidth=640; #pixel
screenHeight=480; #pixel


def get_rs_param(cfg):
    profile = cfg.get_stream(rs.stream.color)
    intr = profile.as_video_stream_profile().get_intrinsics()
    return [intr.fx, intr.fy, intr.ppx, intr.ppy]

tag_im_raw = cv2.imread("tag0.png")


# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, screenWidth, screenHeight, rs.format.bgr8, 30)

# Start streaming
cfg=pipeline.start(config)
cam_param=get_rs_param(cfg)
at_detector = Detector(families='tag36h11',
                       nthreads=1,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)

center = [int(screenWidth/2), int(screenHeight/2)]
size=30
corners = np.array([[-1,1],[1,1],[1,-1],[-1,-1]])*size+center
stop = False

try:
    while not stop:
        tag_im_gray = cv2.cvtColor(tag_im_raw, cv2.COLOR_BGR2GRAY)
        tags = at_detector.detect(tag_im_gray)
        corner_im = tags[0].corners
        
        H = np.eye(3)

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        # Convert images to numpy arrays
        image = np.asanyarray(color_frame.get_data())

        #Tag detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
        for corner in corners:
            image= cv2.circle(image, center = (int(corner[0]), int(corner[1])),radius=5,color = [0,255,0], thickness = -1)   
            
        tags = at_detector.detect(gray)
        if len(tags):
            tag = tags[0]
            H = cv2.findHomography(tag.corners,corner_im)[0]
            for corner in tag.corners:
                image= cv2.circle(image, center = (int(corner[0]), int(corner[1])),radius=5,color = [0,0,255], thickness = -1)
            corner_new = cv2.perspectiveTransform(np.array([corners], dtype='float32'), H)
            
            H = cv2.findHomography(corner_im,corner_new)[0]
 
        
            tag_im_raw = cv2.warpPerspective(tag_im_raw,H, (2000,2000))
        cv2.imshow('feed', image)
        cv2.imshow('tag', tag_im_raw)

        cv2.waitKey(300)
except KeyboardInterrupt:
    cv2.destroyAllWindows()
    pipeline.stop()
                       