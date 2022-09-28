from utils import readAndSaveVid, processImages, makeConcatVideos

def saveVidFrames():
    videoFileName = "silOilPFD_20uLh_500fps_continuousExtrude.mp4"
    videoFilePath = "data/continuousFlow/"
    readAndSaveVid(videoFileName, videoFilePath)

def processData():
    framePath = "data/continuousFlow/frames/"
    binaryPath = "data/continuousFlow/binary/"
    processImages(framePath, binaryPath)

def makeVideo():
    framePath = "data/continuousFlow/frames/"
    binaryPath = "data/continuousFlow/binary/"
    videoName_avi = "continuousFlow.avi"
    makeConcatVideos(framePath, binaryPath, videoName_avi)

# main function
if __name__ == "__main__":
    # saveVidFrames()
    # processData()
    # makeVideo()
    