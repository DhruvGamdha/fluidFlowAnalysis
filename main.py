from bubbleAnalysis import bubbleAnalysis

# main function
if __name__ == "__main__":
    videoPath   = 'data/fluidFlow2/'
    videoName   = 'Extrusions_5000cst_500fps_run2_drop2_cut.avi'
    baseResDir  = 'results/fluidFlow2/'
    newResultsDir = False
    resultsDir = 'version_1'
    
    analysis    = bubbleAnalysis(newResultsDir, videoPath, videoName, baseResDir, resultsDir)
    # analysis.getBinaryImages()
    analysis.createVideoFromFrames('binary/all/frames', 'binary/all')
    # analysis.binaryAnalysis()
    # analysis.createConcatVideo('vidFramesDirPath', 'binary/all/frames', 'binary/all')
    
    print("Done")
    