from bubbleAnalysis import bubbleAnalysis

# main function
if __name__ == "__main__":
    # temp main function
    from directories import directories
    inpPth_base   = 'data/fluidFlow2/'
    outpPth_base  = 'results/fluidFlow2/'
    inpTemplate   = 'version_{:02d}'
    outpTemplate  = 'version_{:02d}'
    inpTemplateIndex  = 1
    outpTemplateIndex = 2
    inpDirsToCreate_wrtTemplate = [ 'all/frames',
                                    'set1/frames']
    outpDirsToCreate_wrtTemplate= [ 'binary/all/frames', 
                                    'analysis/pixSize/frames', 
                                    'analysis/vertPos/frames', 
                                    'analysis/dynamicMarker/frames',
                                    'analysis/bubbleTracking/',
                                    'analysis/bubbleTracking/frames']  
    
    inpDirObj = directories(inpPth_base, inpTemplate, inpTemplateIndex, inpDirsToCreate_wrtTemplate) # Create input directories
    inpDirObj.addDir_usingKey('__base__', 'original/frames')                                    # create original/frames directory wrt base directory
    outpDirObj = directories(outpPth_base, outpTemplate, outpTemplateIndex, outpDirsToCreate_wrtTemplate)  # Create output directories
    
    videoFPS            = 30
    frameNameTemplate   = 'frame_{:04d}.png'
    flowType            = 2
    inpVideoFormat      = '.avi'
    bubbleListIndex     = range(20)
    analysis            = bubbleAnalysis(videoFPS, frameNameTemplate, flowType, inpVideoFormat)
    
    analysis.getFramesFromVideo(inpDirObj.getDirPathObj('original/frames'))
    analysis.getCroppedFrames(inpDirObj.getDirPathObj('original/frames'), inpDirObj.getDirPathObj('all/frames'))
    analysis.getBinaryImages(inpDirObj.getDirPathObj('all/frames'), outpDirObj.getDirPathObj('binary/all/frames'))
    analysis.createVideoFromFrames(outpDirObj.getDirPathObj('binary/all/frames'))
    analysis.extractFrameObjects(outpDirObj.getDirPathObj('binary/all/frames'), outpDirObj.getDirPathObj('analysis/pixSize/frames').parents[1])
    analysis.createVideoFromFrames(outpDirObj.getDirPathObj('analysis/pixSize/frames'))
    analysis.createVideoFromFrames(outpDirObj.getDirPathObj('analysis/vertPos/frames'))
    analysis.createVideoFromFrames(outpDirObj.getDirPathObj('analysis/dynamicMarker/frames'))
    analysis.evaluateBubbleTrajectory()
    # analysis.plotBubbleTrajectory(bubbleListIndex, outpDirObj.getDirPathObj('binary/all/frames'), outpDirObj.getDirPathObj('analysis/bubbleTracking/'))
    analysis.app2plotBubbleTrajectory(bubbleListIndex, outpDirObj.getDirPathObj('binary/all/frames'), outpDirObj.getDirPathObj('analysis/bubbleTracking/frames'))