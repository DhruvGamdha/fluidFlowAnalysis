from bubbleAnalysis import bubbleAnalysis
# import pathlib as pl
# main function
def oldMain():
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
    

if __name__ == "__main__":
    # temp main function
    from directories import directories
    inpPth_base   = 'data/fluidFlow2/'
    outpPth_base  = 'results/fluidFlow2/'
    inpTemplate   = 'version_{:02d}'
    outpTemplate  = 'version_{:02d}'
    inpTempIndex  = 1
    outpTempIndex = 1
    inpDirsToCreate_wrtTemplate = [ 'all/frames',
                                    'set1/frames']
    outpDirsToCreate_wrtTemplate= [ 'binary/all/frames', 
                                    'analysis/pixSize/frames', 
                                    'analysis/vertPos/frames', 
                                    'analysis/dynamicMarker/frames',
                                    'analysis/bubbleTracking/']  
    
    inpDirObj = directories(inpPth_base, inpTemplate, inpTempIndex, inpDirsToCreate_wrtTemplate) # Create input directories
    inpDirObj.addDir_usingKey('__base__', 'original/frames')                                    # create original/frames directory wrt base directory
    outpDirObj = directories(outpPth_base, outpTemplate, outpTempIndex, outpDirsToCreate_wrtTemplate)  # Create output directories
    
    videoFPS            = 30
    frameNameTemplate   = 'frame_{:04d}.png'
    flowType            = 2
    inpVideoFormat      = '.avi'
    distanceThreshold   = 10
    sizeThreshold       = 10
    bubbleListIndex     = range(3, 20)
    analysis            = bubbleAnalysis(videoFPS, frameNameTemplate, flowType, inpVideoFormat)
    
    analysis.getFramesFromVideo(inpDirObj.getDirPathObj('original/frames'))
    analysis.getCroppedFrames(inpDirObj.getDirPathObj('original/frames'), inpDirObj.getDirPathObj('all/frames'))
    analysis.getBinaryImages(inpDirObj.getDirPathObj('all/frames'), outpDirObj.getDirPathObj('binary/all/frames'))
    analysis.createVideoFromFrames(outpDirObj.getDirPathObj('binary/all/frames'))
    analysis.extractFrameObjects(outpDirObj.getDirPathObj('binary/all/frames'), outpDirObj.getDirPathObj('analysis/pixSize/frames').parents[1])
    analysis.createVideoFromFrames(outpDirObj.getDirPathObj('analysis/pixSize/frames'))
    analysis.createVideoFromFrames(outpDirObj.getDirPathObj('analysis/vertPos/frames'))
    analysis.createVideoFromFrames(outpDirObj.getDirPathObj('analysis/dynamicMarker/frames'))
    analysis.evaluateBubbleTrajectory(distanceThreshold, sizeThreshold)
    analysis.plotBubbleTrajectory(bubbleListIndex, outpDirObj.getDirPathObj('binary/all/frames'), outpDirObj.getDirPathObj('analysis/bubbleTracking/'))