from utils import readAndSaveVid, processImages, makeConcatVideos

def saveVidFrames():
    videoFileName = "silOilPFD_20uLh_500fps_continuousExtrude.mp4"
    videoFilePath = "data/continuousFlow/"
    saveFramePath = "data/continuousFlow/frames/"
    readAndSaveVid(videoFileName, videoFilePath, saveFramePath)

def processData():
    framePath = "data/continuousFlow/frames/"
    binaryPath = "results/continuousFlow/binary/"
    processImages(framePath, binaryPath)

def makeVideo():
    framePath = "data/continuousFlow/frames/"
    binaryPath = "results/continuousFlow/binary/"
    videoName_avi = "results/continuousFlow/videos/continuousFlow.avi"
    makeConcatVideos(framePath, binaryPath, videoName_avi)

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
    