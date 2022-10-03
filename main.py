from utils import readAndSaveVid, processImages, makeConcatVideos

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
    binaryPath = "results/continuousFlow/binary_v3/"
    
    # Crop the frame from center corresponding to below values, original size is 400x800
    top, bottom, left, right = cropParameters()
    processImages(framePath, binaryPath, top, bottom, left, right)

def makeVideo():
    framePath = "data/continuousFlow/frames/"
    binaryPath = "results/continuousFlow/binary_v3/"
    videoName_avi = "results/continuousFlow/videos/binary_v3/continuousFlow.avi"
    
    # Crop the frame from center corresponding to below values, original size is 400x800
    top, bottom, left, right = cropParameters()
    makeConcatVideos(framePath, binaryPath, videoName_avi, top, bottom, left, right)

def binarySegmentation():
    import cv2
    import os
    from os.path import join
    import skimage.measure as skm
    
    binaryPath  = "results/continuousFlow/binary/"
    segPath     = "results/continuousFlow/segmentation/"
    connectivity = 1
    
    # Read the binary images
    allFramesName = [f for f in os.listdir(binaryPath) if (os.path.isfile(join(binaryPath, f)) and f.endswith(".png"))]
    
    img = cv2.imread(join(binaryPath, allFramesName[0]), 0)
    
    print("Img Shape:   ",img.shape)
    print("Img Type:    ",img.dtype)
    
    labelImg, count = skm.label(img, connectivity=connectivity, return_num=True)
    
    
    
    
    
    

# main function
if __name__ == "__main__":
    # saveVidFrames()
    # processData()
    # makeVideo()
    print("Done")
    