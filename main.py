from bubbleAnalysis import bubbleAnalysis
from directories import directories, updateTemplateIndex 
import libconf
import git
# main function
if __name__ == "__main__":
    
    para = dict()
    with open('config.cfg', 'r') as f:
        para = libconf.load(f)
    
    inpPth_base         = para['inpPth_base']
    outpPth_base        = para['outpPth_base']
    inpTemplate         = para['inpTemplate']
    outpTemplate        = para['outpTemplate']
    inpTemplateIndex    = para['inpTemplateIndex']
    outpTemplateIndex   = para['outpTemplateIndex']
    inpDirsToCreate_wrtTemplate = para['inpDirsToCreate_wrtTemplate']
    outpDirsToCreate_wrtTemplate= para['outpDirsToCreate_wrtTemplate']
    
    inpDirObj = directories(inpPth_base, inpTemplate, inpTemplateIndex, inpDirsToCreate_wrtTemplate) # Create input directories
    inpDirObj.addDir_usingKey(para['additionalDirs'][0], para['additionalDirs'][1])                 # create original/frames directory wrt base directory
    outpDirObj = directories(outpPth_base, outpTemplate, outpTemplateIndex, outpDirsToCreate_wrtTemplate)  # Create output directories
    
    # Get the current git commit hash
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    para['gitCommitHash'] = sha     # Add the git commit hash to the config file
    
    # Save the config file in the output directory for future reference
    cfgTempIndex = updateTemplateIndex(outpDirObj.getDirPathObj('__template__'), para['cfgFileTemplate'], -1)
    with open(outpDirObj.getDirPathObj('__template__') / para['cfgFileTemplate'].format(cfgTempIndex), 'w') as f:
        libconf.dump(para, f)
    
    # Create the analysis object
    analysis = bubbleAnalysis(para)
    analysis.getFramesFromVideo(inpDirObj.getDirPathObj('original/frames'))
    analysis.getCroppedFrames(inpDirObj.getDirPathObj('original/frames'), inpDirObj.getDirPathObj('all/frames'))
    analysis.getBinaryImages(inpDirObj.getDirPathObj('all/frames'), outpDirObj.getDirPathObj('binary/all/frames'))
    analysis.createVideoFromFrames(outpDirObj.getDirPathObj('binary/all/frames'))
    analysis.createConcatVideo(inpDirObj.getDirPathObj('all/frames'), outpDirObj.getDirPathObj('binary/all/frames'), outpDirObj.getDirPathObj('binary/all/frames').parent)
    analysis.extractFrameObjects(outpDirObj.getDirPathObj('binary/all/frames'), outpDirObj.getDirPathObj('analysis/pixSize/frames').parents[1])
    # analysis.createVideoFromFrames(outpDirObj.getDirPathObj('analysis/pixSize/frames'))
    # analysis.createVideoFromFrames(outpDirObj.getDirPathObj('analysis/vertPos/frames'))
    # analysis.createVideoFromFrames(outpDirObj.getDirPathObj('analysis/dynamicMarker/frames'))
    analysis.evaluateBubbleTrajectory()
    analysis.plotBubbleTrajectory(outpDirObj.getDirPathObj('binary/all/frames'), outpDirObj.getDirPathObj('analysis/bubbleTracking/'))
    analysis.app2plotBubbleTrajectory(outpDirObj.getDirPathObj('binary/all/frames'), outpDirObj.getDirPathObj('analysis/bubbleTracking/frames'))