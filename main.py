from bubbleAnalysis import bubbleAnalysis
# import pathlib as pl
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
    

def run():
    # temp main function
    from directories import directories
    baseInpPth    = 'data/fluidFlow2/'
    inpTemplate   = 'version_{:02d}'
    inpTempIndex  = 1
    inpDirsToCreate_wrtTemplate = [ 'all/frames',
                                    'set1/frames']
    inpDirObj = directories(baseInpPth, inpTemplate, inpTempIndex, inpDirsToCreate_wrtTemplate) # Create input directories
    inpDirObj.addDir_usingKey('__base__', 'original/frames')                                    # create original/frames directory wrt base directory
    
    baseOutpPth   = 'results/fluidFlow2/'
    outpTemplate  = 'version_{:02d}'
    outpTempIndex = 1
    outpDirsToCreate_wrtTemplate= [ 'binary/all/frames', 
                                    'analysis/pixSize/frames', 
                                    'analysis/vertPos/frames', 
                                    'analysis/dynamicMarker/frames']  
    
    outpDirObj = directories(baseOutpPth, outpTemplate, outpTempIndex, outpDirsToCreate_wrtTemplate)  # Create output directories
    
    videoFPS            = 30
    frameNameTemplate   = 'frame_{:04d}.png'
    flowType            = 2
    inpVideoFormat      = '.avi'
    analysis            = bubbleAnalysis(videoFPS, frameNameTemplate, flowType, inpVideoFormat)
    
    analysis.getFramesFromVideo(inpDirObj.getDir_usingKey('original/frames'))
    analysis.getCroppedFrames(inpDirObj.getDir_usingKey('original/frames'), inpDirObj.getDir_usingKey('all/frames'))
    analysis.getBinaryImages(inpDirObj.getDir_usingKey('all/frames'), outpDirObj.getDir_usingKey('binary/all/frames'))
    analysis.createVideoFromFrames(outpDirObj.getDir_usingKey('binary/all/frames'))    