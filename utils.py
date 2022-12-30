import os
from os import listdir
from os.path import join, isfile
from object import Object
from frame import Frame
from video import Video
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import skimage.measure as skm
import pathlib as pl
import parse

# function to read and save video
def readAndSaveVid(videoFramesPathObj, videoFormat, frameNameTemplate):
    allVideoFiles = [f for f in videoFramesPathObj.parent.iterdir() if (f.is_file() and f.suffix == videoFormat)]
    if len(allVideoFiles) != 1:
        print("Number of videos in the folder is not 1")
        return
    
    video = cv2.VideoCapture(str(allVideoFiles[0]))      # Read the video file
    if (video.isOpened()== False):                                      # Check if video file is opened successfully
        print("Error opening video file")
        return
    
    numFrames       = int(video.get(cv2.CAP_PROP_FRAME_COUNT))          # Get number of frames in the video
    if DoNumExistingFramesMatch(videoFramesPathObj, numFrames):
        return
    
    print("If frames already exist inside {}, they will be overwritten.".format(str(videoFramesPathObj)))
    count = 0
    while video.isOpened():
        ret, frame = video.read()
        if not ret:         # Check if frame is read correctly
            print("Can't receive frame (stream end?). Exiting ...")
            break
        cv2.imwrite(str(videoFramesPathObj/frameNameTemplate.format(count)) , frame )
        count += 1
        
def cropFrames(origFrameDir_pathObj, croppedFrameDir_pathObj, frameNameTemplate, params):
    allFramesNum    = getFrameNumbers_ordered(origFrameDir_pathObj, frameNameTemplate)
    if DoNumExistingFramesMatch(croppedFrameDir_pathObj, len(allFramesNum)):
        return
     
    print("If frames already exist inside {}, they will be overwritten.".format(str(croppedFrameDir_pathObj)))
    for frameNum in allFramesNum:
        frame = cv2.imread(str(origFrameDir_pathObj/frameNameTemplate.format(frameNum)), 0)
        frame = frame[  params["top"]   :   params["bottom"]    ,   params["left"]  :   params["right"] ]
        cv2.imwrite(str(croppedFrameDir_pathObj/frameNameTemplate.format(frameNum)) , frame )

def processImages(originalFrameDir_pathObj, binaryFrameDir_pathObj, nameTemplate, params):
    allFramesNum    = getFrameNumbers_ordered(originalFrameDir_pathObj, nameTemplate)
    if DoNumExistingFramesMatch(binaryFrameDir_pathObj, len(allFramesNum)):
        return
    
    for frameNum in tqdm(allFramesNum, desc="Processing frames"):
        frame           = cv2.imread(str(originalFrameDir_pathObj/nameTemplate.format(frameNum)), 0)
        th2             = cv2.adaptiveThreshold(frame,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,    params["blockSize"],    params["constantSub"])
        invth2          = 255 - th2    
        labelImg, count = skm.label(invth2, connectivity=params["connectivity"], return_num=True)
        
        # closing
        kernel          = np.ones((params["C_O_KernelSize"],params["C_O_KernelSize"]),np.uint8)
        invth2          = cv2.morphologyEx(invth2, cv2.MORPH_CLOSE, kernel)
        
        # opening
        kernel          = np.ones((params["O_C_KernelSize"],params["O_C_KernelSize"]),np.uint8)
        invth2          = cv2.morphologyEx(invth2, cv2.MORPH_OPEN, kernel)
        
        for i in range(1, count+1):
            numPixels = np.sum(labelImg == i)
            if numPixels <= params["minSize"]:
                invth2[labelImg == i] = 0
                
        th2 = 255 - invth2
        cv2.imwrite(str(binaryFrameDir_pathObj/nameTemplate.format(frameNum)), th2)

def makeConcatVideos(lftFrameDir_pathObj, rhtFrameDir_pathObj, nameTemplate, videoDir_pathObj, fps, params):
    if checkVideoFileExists(videoDir_pathObj, 'combined'):
        print("Combined video already exists in {}. Skipping.".format(str(videoDir_pathObj)))
        return    
    allFramesNum    = getFrameNumbers_ordered(rhtFrameDir_pathObj, nameTemplate)
    tempImg1 = cv2.imread(str(lftFrameDir_pathObj/nameTemplate.format(allFramesNum[0])), 0)
    tempImg2 = cv2.imread(str(rhtFrameDir_pathObj/nameTemplate.format(allFramesNum[0])), 0)
    
    height1, width1 = tempImg1.shape
    height2, width2 = tempImg2.shape
    videoWidth  = width1 + width2       # width of the video
    videoHeight = max(height1, height2) # height of the video
    
    # vidCodec = cv2.VideoWriter_fourcc(*'XVID')
    vidCodec    = cv2.VideoWriter_fourcc(*'mp4v')  # codec for .mp4 file
    videoWriter = cv2.VideoWriter(str(videoDir_pathObj / 'videoCombined.mp4'),vidCodec, fps, (videoWidth, videoHeight))
    
    for frameNum in tqdm(allFramesNum, desc="Making video"):
        lft_img = cv2.imread(str(lftFrameDir_pathObj/nameTemplate.format(frameNum)))
        bin_img = cv2.imread(str(rhtFrameDir_pathObj/nameTemplate.format(frameNum)))
        if lft_img is None or bin_img is None:
            print("Can't receive frame (stream end?). Exiting ...")
            exit()
        # make bin_img and frm_img height the same size as videoHeight by adding black pixels
        bin_img     = cv2.copyMakeBorder(bin_img, 0, videoHeight - height1, 0, 0, cv2.BORDER_CONSTANT, value=[0,0,0])
        lft_img     = cv2.copyMakeBorder(lft_img, 0, videoHeight - height2, 0, 0, cv2.BORDER_CONSTANT, value=[0,0,0])
        concat_img  = np.concatenate((lft_img, bin_img), axis=1)     # Concatenate the two images horizontally (i.e. side-by-side)
        videoWriter.write(concat_img) # Write the frame to video file
    cv2.destroyAllWindows()
    videoWriter.release() # Now the video is saved in the current directory

def makeSingleVideo(framePathObj, nameTemplate, fps):
    if checkVideoFileExists(framePathObj.parent, 'isolated'):
        print("Isolated video already exists in {}. Skipping.".format(str(framePathObj.parent)))
        return
    
    allFramesNum    = getFrameNumbers_ordered(framePathObj, nameTemplate)
    tempImg         = cv2.imread(str(framePathObj/nameTemplate.format(allFramesNum[0])), 0)
    height, width   = tempImg.shape
    
    # Define the codec and create VideoWriter object
    # vidCodec = cv2.VideoWriter_fourcc(*'XVID')
    vidCodec    = cv2.VideoWriter_fourcc(*'mp4v')
    videoWriter = cv2.VideoWriter(str(framePathObj.parent / 'videoIsolated.mp4'),vidCodec, fps, (width, height))
    
    for frameNum in tqdm(allFramesNum, desc="Making video"):
        frm_img = cv2.imread(str(framePathObj/nameTemplate.format(frameNum)))
        if frm_img is None:     # Check if frame is read correctly
            print("Can't receive frame (stream end?). Exiting ...")
            exit()
        videoWriter.write(frm_img)    # feed the concatenated image to the video writer
    cv2.destroyAllWindows()
    videoWriter.release()            # Now the video is saved in the current directory

def checkVideoFileExists(videoDir_pathObj, videType):
    if videType == 'combined':
        videoPath = videoDir_pathObj / 'videoCombined.mp4'
    elif videType == 'isolated':
        videoPath = videoDir_pathObj / 'videoIsolated.mp4'
    else:
        print('Invalid video type')
        return False
    return videoPath.exists()

def dropAnalysis(binaryFrameDir_pathObj, analysisBaseDir_pathObj, frameNameTemplate, params):
    video       = Video()
    allFramesNum= getFrameNumbers_ordered(binaryFrameDir_pathObj, frameNameTemplate)
    condn1      = DoNumExistingFramesMatch(analysisBaseDir_pathObj/ "pixSize" / "frames" , len(allFramesNum))
    condn2      = DoNumExistingFramesMatch(analysisBaseDir_pathObj/ "vertPos" / "frames" , len(allFramesNum))
    condn3      = DoNumExistingFramesMatch(analysisBaseDir_pathObj/ "dynamicMarker" / "frames" , len(allFramesNum))
    condn4      = video.checkAnalysisFileExists(analysisBaseDir_pathObj)
    if condn1 and condn2 and condn3 and condn4:
        video.loadFromTextFile(analysisBaseDir_pathObj)
        return video
    connectivity= params["connectivity"]
    for frameNum in tqdm(allFramesNum, desc="Analyzing drops"):
        labelImg, count, imgShape = imgSegmentation(binaryFrameDir_pathObj, frameNameTemplate, frameNum, connectivity)
        frame = Frame()
        frame.setFrameNumber(frameNum)
        frame.setObjectCount(count)
        for objLabel in range(1,count+1):
            rows, cols = np.where(labelImg == objLabel)
            # Get the position (x, y) of the top left bounding box around the bubble, origin at the bottom left corner
            y   = imgShape[0] - np.min(rows)
            x   = np.min(cols)
            objInd = objLabel - 1
            obj = Object(frameNum, objInd, x, y, len(rows), rows, cols)
            frame.addObject(obj)
        
        video.addFrame(frame)
        plotFrameObjectAnalysis(frame, frameNum, count, imgShape, analysisBaseDir_pathObj, frameNameTemplate)
    
    if not video.isVideoContinuous():
        exit('Video is not continuous. Exiting ...')
        
    video.saveToTextFile(analysisBaseDir_pathObj)
    
    newVideo = Video()
    newVideo.loadFromTextFile(analysisBaseDir_pathObj)
    if newVideo.isSame(video):
        print("Video analysis saved successfully")
    else:
        print("Video analysis not saved successfully")
        exit()
    return video
    
def imgSegmentation(binaryFrameDir_pathObj, nameTemplate, frameNum, connectivity):
    img = cv2.imread(str(binaryFrameDir_pathObj/nameTemplate.format(frameNum)), 0)
    img = cv2.bitwise_not(img)                                      # invert the binary image
    imgShape = img.shape
    labelImg, count = skm.label(img, connectivity=connectivity, return_num=True)
    return labelImg, count, imgShape

def getFrameNumbers_ordered(framePathObj, nameTemplate):
    allFramePathObj_unorder  = [f for f in framePathObj.iterdir() if (f.is_file() and f.suffix == ".png")]    # Read the binary images
    allFramesNum           = np.zeros(len(allFramePathObj_unorder))  # create numpy array to store the frame numbers
    
    for i in range(len(allFramePathObj_unorder)):
        frameName       = allFramePathObj_unorder[i].name
        frameNum        = parse.parse(nameTemplate, frameName).fixed[0]
        allFramesNum[i] = frameNum
    allFramesNum = np.sort(allFramesNum).astype(int)                # sort the frame numbers
    
    # Check if the frame numbers are continuous
    if not np.array_equal(allFramesNum, np.arange(allFramesNum[0], allFramesNum[-1]+1)):
        print("Frame numbers are not continuous")
        exit()
    
    return allFramesNum

def DoNumExistingFramesMatch(frameDir_pathObj, numFramesToCreate):
    numExistingFile = len([f for f in frameDir_pathObj.iterdir() if (f.is_file() and f.suffix == ".png")])    # Get number of frames already saved
    if numExistingFile == numFramesToCreate:
        print("Already all the frames exist in {}.".format(frameDir_pathObj))
        print("Skipping frame creation.")
        return True
    return False

def plotFrameObjectAnalysis(frameObj, frameNum, numBubbles, imgShape, analysisBaseDir_pathObj, frameNameTemplate):
    _, bubble_vertPos   = frameObj.getObjectPositionList()
    bubble_pixSize      = frameObj.getObjectSizeList()

    # Plot the bubble pixel sizes, with x axis label as the bubble number, plot title as frame number, set the x and y axis limits
    plt.plot(bubble_pixSize)
    plt.xlabel("Bubble Number")
    plt.ylabel("Bubble Pixel Size")
    plt.title("bubble sizes in frame number: " + str(frameNum))
    plt.xlim(0, 50)
    plt.ylim(0, 1000)
    plt.savefig(analysisBaseDir_pathObj / "pixSize" / "frames" / frameNameTemplate.format(frameNum), dpi=200)
    # plt.show()
    plt.close()
    
    # scatter plot the bubble vertical position, with x axis label as the bubble number, plot title as frame number, set the x and y axis limits
    plt.scatter(range(1, numBubbles+1), bubble_vertPos)
    plt.xlabel("Bubble Number")
    plt.ylabel("Bubble Vertical Position")
    plt.title("bubble positions in frame number: " + str(frameNum))
    plt.xlim(0, 50)
    plt.ylim(0, imgShape[0])
    plt.savefig(analysisBaseDir_pathObj / "vertPos" / "frames" / frameNameTemplate.format(frameNum), dpi=200)
    # plt.show()
    plt.close()
    
    # scatter plot the bubble vertical position, with x axis label as the bubble number, plot title as frame number, set the x and y axis limits
    plt.scatter(range(1, numBubbles+1), bubble_vertPos, s=bubble_pixSize)
    plt.xlabel("Bubble Number")
    plt.ylabel("Bubble Vertical Position")
    plt.title("bubble positions in frame number: " + str(frameNum))
    plt.xlim(0, 50)
    plt.ylim(0, imgShape[0])
    plt.savefig(analysisBaseDir_pathObj / "dynamicMarker" / "frames" / frameNameTemplate.format(frameNum), dpi=200)
    # plt.show()
    plt.close()
    plt.close('all')