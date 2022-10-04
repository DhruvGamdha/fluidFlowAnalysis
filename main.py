from utils import readAndSaveVid, processImages, makeConcatVideos, makeSingleVideo

def saveVidFrames():
    videoFileName = "silOilPFD_20uLh_500fps_continuousExtrude.mp4"
    videoFilePath = "data/continuousFlow/"
    saveFramePath = "data/continuousFlow/frames/"
    readAndSaveVid(videoFileName, videoFilePath, saveFramePath)

def cropParameters():
    # Crop the frame from center corresponding to below values, original size is 400x800
    
    # binary_v1 parameters
    top = 100
    bottom = 700
    left = 100
    right = 300
    
    '''     
    # binary_v3 parameters
    top = 120
    bottom = 700
    left = 150
    right = 250
    '''
    return top, bottom, left, right

def processData():
    framePath = "data/continuousFlow/frames/"
    binaryPath = "results/continuousFlow/binary/binary_v3/"
    
    # Crop the frame from center corresponding to below values, original size is 400x800
    top, bottom, left, right = cropParameters()
    processImages(framePath, binaryPath, top, bottom, left, right)

def makeVideo_concat():
    # framePath = "data/continuousFlow/frames/"
    lftFramesPath = "results/continuousFlow/binary/binary_v3/drop1/frames/"
    # binaryPath = "results/continuousFlow/binary/binary_v1/all/frames/"
    rhtFramesPath = "results/continuousFlow/binary/binary_v3/drop1/analysis/connectivity_1/dynamicMarker/frames/"
    videoLocAndName_mp4 = "results/continuousFlow/binary/binary_v3/drop1/analysis/connectivity_1/dynamicMarker/continuousFlow_club.mp4"
    nameTemplate = "frame%d.png"
    fps = 60.0
    # Crop the frame from center corresponding to below values, original size is 400x800
    top, bottom, left, right = cropParameters()
    makeConcatVideos(lftFramesPath, rhtFramesPath, nameTemplate, videoLocAndName_mp4, fps, top, bottom, left, right)

def makeVideo_single():
    framePath = "results/continuousFlow/binary/binary_v3/drop1/analysis/connectivity_2/dynamicMarker/frames/"
    nameTemplate = "frame%d.png"
    videoPath = "results/continuousFlow/binary/binary_v3/drop1/analysis/connectivity_2/dynamicMarker/allFrames.avi"
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
    binaryPath  = "results/continuousFlow/binary/binary_v3/drop2/frames/"
    analysisPath = "results/continuousFlow/binary/binary_v3/drop2/analysis/connectivity_1/"
    nameTemplate = "frame%d.png"
    connectivity = 1
    
    dropAnalysis(binaryPath, analysisPath, nameTemplate, connectivity)

# main function
if __name__ == "__main__":
    # saveVidFrames()
    # processData()
    makeVideo_concat()
    # makeVideo_single()
    # useDropAnalysis()
    print("Done")
    