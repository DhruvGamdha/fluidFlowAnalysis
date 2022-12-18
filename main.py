from bubbleAnalysis import bubbleAnalysis

# main function
if __name__ == "__main__":
    baseVideoPath= 'data/fluidFlow2/'
    baseResPath  = 'results/fluidFlow2/'
    # newResultsDir = False
    resultsDirVerNum    = 1     # Either -1 (for new results directory) or the version number
    dataDirVerNum       = 1     # Either -1 (for new data directory) or the version number
    
    analysis    = bubbleAnalysis(baseVideoPath, baseResPath, resultsDirVerNum, dataDirVerNum)
    # analysis.getFramesFromVideo()
    # analysis.getBinaryImages()
    # analysis.createVideoFromFrames('binary/all/frames', 'binary/all')
    # analysis.binaryAnalysis()
    # analysis.createVideoFromFrames('analysis/pixSize/frames', 'analysis/pixSize')
    # analysis.createVideoFromFrames('analysis/vertPos/frames', 'analysis/vertPos')
    # analysis.createVideoFromFrames('analysis/dynamicMarker/frames', 'analysis/dynamicMarker')
    analysis.createConcatVideo('binary/all/frames', 'analysis/dynamicMarker/frames', 'analysis/dynamicMarker')
    # analysis.createConcatVideo('binary/all/frames', 'analysis/pixSize/frames', 'analysis/pixSize')
    # analysis.createConcatVideo('binary/all/frames', 'analysis/vertPos/frames', 'analysis/vertPos')
    
    print("Done")
    