
# function to read and save video
def readAndSaveVid(videoFileName, videoFilePath):
    import cv2
    from os.path import join

    videoFile = join(videoFilePath, videoFileName)

    # Read the .mp4 file
    video = cv2.VideoCapture(videoFile)

    if video.isOpened():
        print("Video file opened successfully")
    else:
        print("Error opening video file")

    count = 0
    while video.isOpened():
        ret, frame = video.read()
        # Check if frame is read correctly
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        else: # Save the frame as a .png file
            cv2.imwrite(join(videoFilePath, "frames","frame%d.png" % count), frame)
            count += 1

def processImages(framePath, binaryPath):
    import cv2
    import os
    from os.path import join
    import numpy as np
    import matplotlib.pyplot as plt
    
    # framePath = "data/continuousFlow/frames/"
    # binaryPath = "data/continuousFlow/binary/"
    allFramesName = [f for f in os.listdir(framePath) if os.path.isfile(join(framePath, f))]
    
    for frameName in allFramesName:
        print(frameName)
        # Check if the frameName is a .png file
        if frameName[-4:] == ".png":    
            # read the frame
            frame = cv2.imread(join(framePath, frameName), 0)
            
            # Crop the frame from certre, original size is 400x800
            top = 100
            bottom = 700
            left = 100
            right = 300
            frame = frame[top:bottom,left:right]
            
            # NOTE: The best parameters are i=9 and j=5, gaussian adaptive thresholding
            i = 9
            j = 5
            
            th2 = cv2.adaptiveThreshold(frame,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,i,j)
            cv2.imwrite(join(binaryPath, frameName), th2)
            