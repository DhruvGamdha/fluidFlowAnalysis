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
    framePath = "results/continuousFlow/binary/binary_v3/drop1/analysis/connectivity_1/vertPos/frames/"
    nameTemplate = "frame%d.png"
    videoPath = "results/continuousFlow/binary/binary_v3/drop1/analysis/connectivity_1/vertPos/allFrames.avi"
    fps = 30.0
    makeSingleVideo(framePath, nameTemplate, videoPath, fps)

def useDropAnalysis():
    # Drop analysis
    '''
    You need the following directories inside the analysisPath directory to store the results:
    1. dynamicMarker
    2. pixSize
    3. vertPos
    '''
    
    from utils import dropAnalysis
    binaryPath  = "results/continuousFlow/binary/binary_v3/drop1/frames/"
    analysisPath = "results/continuousFlow/binary/binary_v3/drop1/analysis/connectivity_2/"
    nameTemplate = "frame%d.png"
    connectivity = 1
    
    dropAnalysis(binaryPath, analysisPath, nameTemplate, connectivity)

# main function
if __name__ == "__main__":
    # saveVidFrames()
    # processData()
    # makeVideo_concat()
    # makeVideo_single()
    useDropAnalysis()
    print("Done")
    