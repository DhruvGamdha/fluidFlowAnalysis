from utils import readAndSaveVid, processImages, makeConcatVideos, makeSingleVideo

def saveVidFrames():
    videoFileName = "silOilPFD_20uLh_500fps_continuousExtrude.mp4"
    videoFilePath = "data/continuousFlow/"
    saveFramePath = "data/continuousFlow/frames/"
    readAndSaveVid(videoFileName, videoFilePath, saveFramePath)

def cropParameters():
    # Crop the frame from center corresponding to below values, original size is 400x800
    top = 120
    bottom = 700
    left = 150
    right = 250
    
    return top, bottom, left, right

def processData():
    framePath = "data/continuousFlow/frames/"
    binaryPath = "results/continuousFlow/binary/binary_v3/"
    
    # Crop the frame from center corresponding to below values, original size is 400x800
    top, bottom, left, right = cropParameters()
    processImages(framePath, binaryPath, top, bottom, left, right)

def makeVideo_concat():
    framePath = "data/continuousFlow/frames/"
    binaryPath = "results/continuousFlow/binary/binary_v3/"
    videoName_avi = "results/continuousFlow/videos/binary_v3/continuousFlow.avi"
    
    # Crop the frame from center corresponding to below values, original size is 400x800
    top, bottom, left, right = cropParameters()
    makeConcatVideos(framePath, binaryPath, videoName_avi, top, bottom, left, right)

def makeVideo_single():
    framePath = "results/continuousFlow/binary/binary_v3/drop1/frames/"
    nameTemplate = "frame%d.png"
    videoPath = "results/continuousFlow/binary/binary_v3/drop1/video/allFrames.avi"
    makeSingleVideo(framePath, nameTemplate, videoPath)

def dropAnalysis():
    import cv2
    import os
    from os.path import join
    import numpy as np
    import skimage.measure as skm
    import matplotlib.pyplot as plt
    
    binaryPath  = "results/continuousFlow/binary/binary_v3/drop1/frames/"
    analysisPath = "results/continuousFlow/binary/binary_v3/drop1/analysis/"
    nameTemplate = "frame%d.png"
    # drop_startFrameNum = 200
    connectivity = 2
    
    # Read the binary images
    allFrameNames_unorder = [f for f in os.listdir(binaryPath) if (os.path.isfile(join(binaryPath, f)) and f.endswith(".png"))]
    
    # create numpy array to store the frame numbers
    allFramesNum = np.zeros(len(allFrameNames_unorder))
    
    # use the name template of type "<string>%d.png" to extract the frame number from each frame name
    for i in range(len(allFrameNames_unorder)):
        allFramesNum[i] = int(allFrameNames_unorder[i][len(nameTemplate)-6:-4])
    
    allFramesNum = np.sort(allFramesNum).astype(int)        # sort the frame numbers
    
    frame_bubbleCount = []              # bubble count, frame wise
    frame_bubblePixSizes = []           # pixel sizes of all the bubbles, frame wise
    frame_bubbleVertPos = []            # vertical position of all the bubbles, frame wise
    
    for frameNum in allFramesNum:
        img = cv2.imread(join(binaryPath, nameTemplate % frameNum), 0)    # read the image
        img = cv2.bitwise_not(img)                          # invert the binary image
        
        labelImg, count = skm.label(img, connectivity=connectivity, return_num=True)
        frame_bubbleCount.append(count)
        
        # Print frame number and bubble count
        print("Frame Number: ", frameNum, "Bubble Count: ", count)
        
        # Get the bubble details
        bubble_pixSize = []         # bubble pixel size, bubble wise
        bubble_vertPos = []        # bubble highest pixel vertical position, bubble wise
        
        for i in range(1,count+1):
        # for i in range(5,6):
            rows, cols = np.where(labelImg == i)
            bubble_pixSize.append(len(rows))        # adding ith bubble pixel size
            vertPos = np.min(rows)                  # finding the highest pixel vertical position
            vertPos = img.shape[0] - vertPos        # converting to the original image coordinates
            bubble_vertPos.append(vertPos)        # adding ith bubble highest pixel position
        
        frame_bubblePixSizes.append(bubble_pixSize)     # adding all bubble pixel sizes in the frame
        frame_bubbleVertPos.append(bubble_vertPos)      # adding all bubble highest pixel positions in the frame
        
        # Plot the bubble pixel sizes, with x axis label as the bubble number, plot title as frame number, set the x and y axis limits
        plt.plot(bubble_pixSize)
        plt.xlabel("Bubble Number")
        plt.ylabel("Bubble Pixel Size")
        plt.title("bubble sizes in Frame Number: " + str(frameNum))
        plt.xlim(0, 50)
        plt.ylim(0, 1000)
        plt.savefig(join(analysisPath, "pixSize", "bubblePixSize_" + str(frameNum) + ".png"), dpi=200)
        # plt.show()
        plt.close()
        
        # scatter plot the bubble vertical position, with x axis label as the bubble number, plot title as frame number, set the x and y axis limits
        plt.scatter(range(1, count+1), bubble_vertPos)
        # plt.plot(bubble_vertPos)
        plt.xlabel("Bubble Number")
        plt.ylabel("Bubble Vertical Position")
        plt.title("bubble positions in Frame Number: " + str(frameNum))
        plt.xlim(0, 50)
        plt.ylim(0, img.shape[0])
        plt.savefig(join(analysisPath, "vertPos", "bubbleVertPos_" + str(frameNum) + ".png"), dpi=200)
        # plt.show()
        plt.close()
        
        # scatter plot the bubble vertical position, with x axis label as the bubble number, plot title as frame number, set the x and y axis limits
        plt.scatter(range(1, count+1), bubble_vertPos, s=bubble_pixSize)
        # plt.plot(bubble_vertPos)
        plt.xlabel("Bubble Number")
        plt.ylabel("Bubble Vertical Position")
        plt.title("bubble positions in Frame Number: " + str(frameNum))
        plt.xlim(0, 50)
        plt.ylim(0, img.shape[0])
        plt.savefig(join(analysisPath, "dynamicMarker", "bubbleVertPosMarkerSize_" + str(frameNum) + ".png"), dpi=200)
        # plt.show()
        plt.close()
    
    # Plot the bubble count, frame wise
    plt.plot(frame_bubbleCount)
    plt.xlabel("Frame Number")
    plt.ylabel("Bubble Count")
    plt.title("bubble count in the video")
    plt.savefig(join(analysisPath, "frame_bubbleCount.png"), dpi=200)

# main function
if __name__ == "__main__":
    # saveVidFrames()
    # processData()
    # makeVideo_concat()
    makeVideo_single()
    # dropAnalysis()
    print("Done")
    