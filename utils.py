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

# function to read and save video
def readAndSaveVid(videoDirPath, saveFramePath, videoFormat):
    allVideosName = [f for f in listdir(videoDirPath) if (isfile(join(videoDirPath, f)) and f.endswith(videoFormat))]   # Get all videos in the folder
    if len(allVideosName) != 1:
        print("Number of videos in the folder is not 1")
        return
    
    video = cv2.VideoCapture(join(videoDirPath, allVideosName[0]))      # Read the video file
    if (video.isOpened()== False):                                      # Check if video file is opened successfully
        print("Error opening video file")
        return
    
    numFrames       = int(video.get(cv2.CAP_PROP_FRAME_COUNT))          # Get number of frames in the video
    numExistingFile = len([f for f in listdir(saveFramePath) if (isfile(join(saveFramePath, f)) and f.endswith(".png"))])    # Get number of frames already saved
    if numExistingFile == numFrames:
        print("Number of frames in the video is same as the number of frames already saved in {}.".format(saveFramePath))
        print("Skipping frame extraction.")
        return
    
    print("If frames already exist inside {}, they will be overwritten.".format(saveFramePath))
    count = 0
    while video.isOpened():
        ret, frame = video.read()
        if not ret:         # Check if frame is read correctly
            print("Can't receive frame (stream end?). Exiting ...")
            break
        print("frame number:", count)
        cv2.imwrite(join(saveFramePath,"frame%d.png" % count), frame)
        count += 1

''' def processImages(framePath, binaryPath, top, bottom, left, right, i, j, min_size, connectivity):
    import cv2
    import os
    from os.path import join
    import numpy as np
    import matplotlib.pyplot as plt
    
    allFramesName = [f for f in os.listdir(framePath) if (os.path.isfile(join(framePath, f)) and f.endswith(".png"))]
    
    # NOTE: The best parameters are i=9 and j=5, gaussian adaptive thresholding
    i = 9
    j = 5
    
    for frameName in allFramesName:
        print(frameName)
        
        frame = cv2.imread(join(framePath, frameName), 0)   # Read the frame as grayscale
        
        frame = frame[top:bottom,left:right]    # Crop the frame from certre, original size is 400x800
        
        th2 = cv2.adaptiveThreshold(frame,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,i,j)
        cv2.imwrite(join(binaryPath, "frames", frameName), th2)
 '''
def processImages(framePath, binaryPath, nameTemplate, params):
    allFramesNum    = getFrameNumbers_ordered(framePath, nameTemplate)
    for frameNum in tqdm(allFramesNum, desc="Processing frames"):
        frame           = cv2.imread(join(framePath, nameTemplate % frameNum), 0)
        frame           = frame[    params["top"]:params["bottom"]  ,   params["left"]:params["right"]  ]        
        th2             = cv2.adaptiveThreshold(frame,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,    params["blockSize"],    params["constSub"])
        invth2          = 255 - th2    
        labelImg, count = skm.label(invth2, connectivity=params["connectivity"], return_num=True)
        
        for i in range(1, count+1):
            numPixels = np.sum(labelImg == i)
            if numPixels <= params["min_size"]:
                invth2[labelImg == i] = 0
                
        th2 = 255 - invth2
        cv2.imwrite(join(binaryPath, nameTemplate % frameNum), th2)

def makeConcatVideos(lftFramesPath, rhtFramesPath, nameTemplate, videoName_avi, fps, params, cropOn = True):
    allFramesNum    = getFrameNumbers_ordered(rhtFramesPath, nameTemplate)
    tempImg2 = cv2.imread(join(lftFramesPath, nameTemplate % allFramesNum[0]), 0)
    tempImg1 = cv2.imread(join(rhtFramesPath, nameTemplate % allFramesNum[0]), 0)
    
    height1, width1 = tempImg1.shape
    height2, width2 = tempImg2.shape
    videoWidth  = width1 + width2       # width of the video
    videoHeight = max(height1, height2) # height of the video
    
    # vidCodec = cv2.VideoWriter_fourcc(*'XVID')
    vidCodec    = cv2.VideoWriter_fourcc(*'mp4v')  # codec for .mp4 file
    video       = cv2.VideoWriter(videoName_avi,vidCodec, fps, (videoWidth, videoHeight))
    
    for frameNum in tqdm(allFramesNum, desc="Making video"):
        lft_img = cv2.imread(join(lftFramesPath, nameTemplate % frameNum))
        bin_img = cv2.imread(join(rhtFramesPath, nameTemplate % frameNum))
        if lft_img is None or bin_img is None:
            print("Can't receive frame (stream end?). Exiting ...")
            exit()
        # make bin_img and frm_img height the same size as videoHeight by adding black pixels
        bin_img     = cv2.copyMakeBorder(bin_img, 0, videoHeight - height1, 0, 0, cv2.BORDER_CONSTANT, value=[0,0,0])
        lft_img     = cv2.copyMakeBorder(lft_img, 0, videoHeight - height2, 0, 0, cv2.BORDER_CONSTANT, value=[0,0,0])
        concat_img  = np.concatenate((lft_img, bin_img), axis=1)     # Concatenate the two images horizontally (i.e. side-by-side)
        video.write(concat_img) # Write the frame to video file
    cv2.destroyAllWindows()
    video.release() # Now the video is saved in the current directory

def makeSingleVideo(framePath, nameTemplate, videoPath, fps):
    allFramesNum    = getFrameNumbers_ordered(framePath, nameTemplate)
    tempImg         = cv2.imread(join(framePath, nameTemplate % allFramesNum[0]), 0)
    height, width   = tempImg.shape
    
    # Define the codec and create VideoWriter object
    # vidCodec = cv2.VideoWriter_fourcc(*'XVID')
    vidCodec    = cv2.VideoWriter_fourcc(*'mp4v')
    video       = cv2.VideoWriter(videoPath, vidCodec, fps, (width, height))
    
    for frameNum in tqdm(allFramesNum, desc="Making video"):
        frm_img = cv2.imread(join(framePath, nameTemplate % frameNum))
        if frm_img is None:     # Check if frame is read correctly
            exit()
        video.write(frm_img)    # feed the concatenated image to the video writer
    cv2.destroyAllWindows()
    video.release()            # Now the video is saved in the current directory

def dropAnalysis(binaryPath, analysisPath, nameTemplate, params):
    allFramesNum            = getFrameNumbers_ordered(binaryPath, nameTemplate)
    connectivity            = params["connectivity"]
    video = Video()
    for frameNum in tqdm(allFramesNum, desc="Analyzing drops"):
        labelImg, count, imgShape = imgSegmentation(binaryPath, nameTemplate, frameNum, connectivity)
        frame = Frame()
        for i in range(1,count+1):
            rows, cols = np.where(labelImg == i)
            # Get the position (x, y) of the top left bounding box around the bubble, origin at the bottom left corner
            y = imgShape[0] - np.min(rows)
            x = np.min(cols)
            obj = Object(x, y, len(rows))
            frame.addObject(obj)
        
        video.addFrame(frame)
        video.addFrameObjCount(count)
        plotFrameObjectAnalysis(frame, frameNum, count, imgShape, analysisPath)
    
    # Plot the bubble count, frame wise
    plt.plot(video.getObjCountList())
    plt.xlabel("Frame Number")
    plt.ylabel("Bubble Count")
    plt.title("bubble count in the video")
    plt.savefig(join(analysisPath, "frame_bubbleCount.png"), dpi=200)
    plt.close()
    
    plt.close('all')
    
def imgSegmentation(binaryPath, nameTemplate, frameNum, connectivity):
    img = cv2.imread(join(binaryPath, nameTemplate % frameNum), 0)    # read the image
    img = cv2.bitwise_not(img)                                      # invert the binary image
    imgShape = img.shape
    labelImg, count = skm.label(img, connectivity=connectivity, return_num=True)
    return labelImg, count, imgShape

def getFrameNumbers_ordered(binaryPath, nameTemplate):
    allFrameNames_unorder   = [f for f in os.listdir(binaryPath) if (os.path.isfile(join(binaryPath, f)) and f.endswith(".png"))]    # Read the binary images
    allFramesNum            = np.zeros(len(allFrameNames_unorder))  # create numpy array to store the frame numbers

    for i in range(len(allFrameNames_unorder)):     # use the name template of type "<string>%d.png" to extract the frame number from each frame name
        allFramesNum[i] = int(allFrameNames_unorder[i][len(nameTemplate)-6:-4])
    allFramesNum = np.sort(allFramesNum).astype(int)        # sort the frame numbers
    return allFramesNum

def plotFrameObjectAnalysis(frameObj, frameNum, numBubbles, imgShape, analysisPath):
    _, bubble_vertPos   = frameObj.getObjectPositionList()
    bubble_pixSize      = frameObj.getObjectSizeList()

    # Plot the bubble pixel sizes, with x axis label as the bubble number, plot title as frame number, set the x and y axis limits
    plt.plot(bubble_pixSize)
    plt.xlabel("Bubble Number")
    plt.ylabel("Bubble Pixel Size")
    plt.title("bubble sizes in frame number: " + str(frameNum))
    plt.xlim(0, 50)
    plt.ylim(0, 1000)
    plt.savefig(join(analysisPath, "pixSize", "frames", "frame" + str(frameNum) + ".png"), dpi=200)
    # plt.show()
    plt.close()
    
    # scatter plot the bubble vertical position, with x axis label as the bubble number, plot title as frame number, set the x and y axis limits
    plt.scatter(range(1, numBubbles+1), bubble_vertPos)
    plt.xlabel("Bubble Number")
    plt.ylabel("Bubble Vertical Position")
    plt.title("bubble positions in frame number: " + str(frameNum))
    plt.xlim(0, 50)
    plt.ylim(0, imgShape[0])
    plt.savefig(join(analysisPath, "vertPos", "frames", "frame" + str(frameNum) + ".png"), dpi=200)
    # plt.show()
    plt.close()
    
    # scatter plot the bubble vertical position, with x axis label as the bubble number, plot title as frame number, set the x and y axis limits
    plt.scatter(range(1, numBubbles+1), bubble_vertPos, s=bubble_pixSize)
    plt.xlabel("Bubble Number")
    plt.ylabel("Bubble Vertical Position")
    plt.title("bubble positions in frame number: " + str(frameNum))
    plt.xlim(0, 50)
    plt.ylim(0, imgShape[0])
    plt.savefig(join(analysisPath, "dynamicMarker", "frames", "frame" + str(frameNum) + ".png"), dpi=200)
    # plt.show()
    plt.close()
    plt.close('all')