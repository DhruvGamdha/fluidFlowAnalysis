from bubbleAnalysis import bubbleAnalysis
def saveVidFrames():
    from utils import readAndSaveVid
    from os.path import join
    videoFileName = "Extrusions_5000cst_500fps_run2_drop2_cut.avi"
    videoFilePath = "/media/dhruv/data/Dhruv/ISU/PhD/Projects/LEAP_HI/software/dataAnalysis/fluidFlow/data/fluidFlow2/"
    saveFramePath = "/media/dhruv/data/Dhruv/ISU/PhD/Projects/LEAP_HI/software/dataAnalysis/fluidFlow/data/fluidFlow2/frames/"
    videoFile = join(videoFilePath, videoFileName)
    
    readAndSaveVid(videoFile, saveFramePath)

def cropParameters():
    # Crop the frame from center corresponding to below values, original size is 400x800
    ''' 
    # fluidFlow1 crop_v1 parameters
    top = 100
    bottom = 700
    left = 100
    right = 300
     '''
        
    ''' 
    # fluidFlow1 crop_v3 parameters
    top = 120
    bottom = 700
    left = 150
    right = 250 
    '''
    
    # fluidFlow2 crop_v1 parameters
    top     = 100
    bottom  = 750
    left    = 105
    right   = 200
   
    return top, bottom, left, right

def createBinaryRep():
    from utils import processImages
    framePath = "data/fluidFlow2/frames/"
    binaryPath = "results/fluidFlow2/binary/binary_v1/"     # NOTE: Create frames directory inside binaryPath, if not present, to store the binary images
    
    top, bottom, left, right = cropParameters()     # frame cropping parameters from center, original size is 400x800
    processImages(framePath, binaryPath, top, bottom, left, right)

def findOptimumThreshold():
    from utils import processImages
    framePath = "data/fluidFlow2/frames/" # frame1209.png
    binaryPath  = "results/fluidFlow2/binary/crop_v1/threshParaStudy/all/v2/"
    nameTemplate = "frame%d.png"
    top, bottom, left, right = cropParameters()     # frame cropping parameters from center, original size is 400x800
    processImages(framePath, binaryPath, top, bottom, left, right)

def makeVideo_concat():
    from utils import makeConcatVideos
    lftFramesPath       = "data/fluidFlow2/frames/"
    # lftFramesPath     = "results/continuousFlow/binary/binary_v3/drop1/frames/"
    rhtFramesPath       = "results/fluidFlow2/binary/crop_v1/threshParaStudy/all/v2/frames/"
    # rhtFramesPath     = "results/continuousFlow/binary/binary_v3/drop1/analysis/connectivity_1/pixSize/frames/"
    videoLocAndName_mp4 = "results/fluidFlow2/binary/crop_v1/threshParaStudy/all/v2/allFrames_club.mp4"
    nameTemplate        = "frame%d.png"
    fps = 60.0
    cropOn = True
    
    top, bottom, left, right = cropParameters()  # frame cropping parameters from center, Applied on the original video frames
    makeConcatVideos(lftFramesPath, rhtFramesPath, nameTemplate, videoLocAndName_mp4, fps, top, bottom, left, right, cropOn)

def makeVideo_single():
    from utils import makeSingleVideo
    framePath = "results/fluidFlow2/binary/binary_v1/frames/"
    nameTemplate = "frame%d.png"
    videoPath = "results/fluidFlow2/binary/binary_v1/allFrames.mp4"
    fps = 60.0
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
    binaryPath  = "results/fluidFlow2/binary/crop_v1/threshParaStudy/all/v2/frames/"
    analysisPath = "results/fluidFlow2/binary/crop_v1/threshParaStudy/all/v2/analysis/"
    nameTemplate = "frame%d.png"
    connectivity = 2
    
    dropAnalysis(binaryPath, analysisPath, nameTemplate, connectivity)

# main function
if __name__ == "__main__":
    # saveVidFrames()
    # createBinaryRep()
    # makeVideo_single()
    # makeVideo_concat()
    # useDropAnalysis()
    
    # findOptimumThreshold()
    # makeVideo_concat()
    
    videoPath   = 'data/fluidFlow2/'
    videoName   = 'Extrusions_5000cst_500fps_run2_drop2_cut.avi'
    baseResDir  = 'results/fluidFlow2/'
    analysis    = bubbleAnalysis(videoPath, videoName, baseResDir)
    analysis.getBinaryImages()
    # analysis.binaryAnalysis()
    # analysis.createConcatVideo('vidFramesDirPath', 'binary/all/frames', 'binary/all')
    
    print("Done")
    