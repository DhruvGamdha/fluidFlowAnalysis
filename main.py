from utils import readAndSaveVid, processImages, makeConcatVideos, makeSingleVideo

def saveVidFrames():
    videoFileName = "Extrusions_5000cst_500fps_run2_drop2_cut.avi"
    videoFilePath = "/media/dhruv/data/Dhruv/ISU/PhD/Projects/LEAP_HI/software/dataAnalysis/fluidFlow/data/fluidFlow2/"
    saveFramePath = "/media/dhruv/data/Dhruv/ISU/PhD/Projects/LEAP_HI/software/dataAnalysis/fluidFlow/data/fluidFlow2/frames/"
    readAndSaveVid(videoFileName, videoFilePath, saveFramePath)

def cropParameters():
    # Crop the frame from center corresponding to below values, original size is 400x800
    ''' 
    # fluidFlow1 binary_v1 parameters
    top = 100
    bottom = 700
    left = 100
    right = 300
     '''
        
    ''' 
    # fluidFlow1 binary_v3 parameters
    top = 120
    bottom = 700
    left = 150
    right = 250 
    '''
    
    # fluidFlow2 binary_v1 parameters
    top     = 100
    bottom  = 750
    left    = 105
    right   = 200
   
    return top, bottom, left, right

def createBinaryRep():
    framePath = "data/fluidFlow2/frames/"
    binaryPath = "results/fluidFlow2/binary/binary_v1/"     # NOTE: Create frames directory inside binaryPath, if not present, to store the binary images
    
    top, bottom, left, right = cropParameters()     # frame cropping parameters from center, original size is 400x800
    processImages(framePath, binaryPath, top, bottom, left, right)

def makeVideo_concat():
    lftFramesPath = "data/fluidFlow2/frames/"
    # lftFramesPath = "results/continuousFlow/binary/binary_v3/drop1/frames/"
    rhtFramesPath = "results/fluidFlow2/binary/binary_v1/frames/"
    # rhtFramesPath = "results/continuousFlow/binary/binary_v3/drop1/analysis/connectivity_1/pixSize/frames/"
    videoLocAndName_mp4 = "results/fluidFlow2/binary/binary_v1/allFrames_club.mp4"
    nameTemplate = "frame%d.png"
    fps = 60.0
    cropOn = True
    
    top, bottom, left, right = cropParameters()  # frame cropping parameters from center, Applied on the original video frames
    makeConcatVideos(lftFramesPath, rhtFramesPath, nameTemplate, videoLocAndName_mp4, fps, top, bottom, left, right, cropOn)

def makeVideo_single():
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
    binaryPath  = "results/continuousFlow/binary/binary_v3/drop2/frames/"
    analysisPath = "results/continuousFlow/binary/binary_v3/drop2/analysis/connectivity_2/"
    nameTemplate = "frame%d.png"
    connectivity = 2
    
    dropAnalysis(binaryPath, analysisPath, nameTemplate, connectivity)

# main function
if __name__ == "__main__":
    # saveVidFrames()
    # createBinaryRep()
    # makeVideo_single()
    makeVideo_concat()
    # useDropAnalysis()
    print("Done")
    