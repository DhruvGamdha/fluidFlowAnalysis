# Class for bubble analysis
import os
class bubbleAnalysis:
    def __init__(self, para):
        
        self.videoFPS           = para['videoFPS']  
        self.frameNameTemplate  = para['frameNameTemplate']
        self.flowType           = para['flowType']
        self.videoFormat        = para['inpVideoFormat']        
        self.analysedVideo      = None              # Video object for the analysed video
        self.params             = para
        
    def getFramesFromVideo(self, videoFramesDir_pathObj):
        from utils import readAndSaveVid
        readAndSaveVid(videoFramesDir_pathObj, self.videoFormat, self.frameNameTemplate)
    
    def getCroppedFrames(self, origFrameDir_pathObj, croppedFrameDir_pathObj):
        from utils import cropFrames
        cropFrames(origFrameDir_pathObj, croppedFrameDir_pathObj, self.frameNameTemplate, self.params)
    
    def getBinaryImages(self, origFrameDir_pathObj, binaryFrameDir_pathObj):
        from utils import processImages
        processImages(origFrameDir_pathObj, binaryFrameDir_pathObj, self.frameNameTemplate, self.params)
        
    def createVideoFromFrames(self, frameDir_pathObj):
        from utils import makeSingleVideo
        makeSingleVideo(frameDir_pathObj, self.frameNameTemplate, self.videoFPS)
          
    def createConcatVideo(self, lftFrameDir_pathObj, rhtFrameDir_pathObj, videoDir_pathObj):
        from utils import makeConcatVideos
        makeConcatVideos(lftFrameDir_pathObj, rhtFrameDir_pathObj, self.frameNameTemplate, videoDir_pathObj, self.videoFPS, self.params)    
    
    def extractFrameObjects(self, binaryFrameDir_pathObj, analysisBaseDir_pathObj):
        from utils import dropAnalysis
        self.analysedVideo = dropAnalysis(binaryFrameDir_pathObj, analysisBaseDir_pathObj, self.frameNameTemplate, self.params)
    
    def evaluateBubbleTrajectory(self):
        self.analysedVideo.trackObjects(self.params)
    
    def plotBubbleTrajectory(self, binaryFrameDir_pathObj, videoDir_pathObj):
        bubbleListIndex     = self.params['bubbleListIndex']
        if self.params['bubbleListIndex'] != False:
            bubbleListIndex = range(self.params['bubbleListIndex'])
        
        if bubbleListIndex == False:
            bubbleListIndex = range(self.analysedVideo.getNumBubbles())
        
        for ind in bubbleListIndex:
            self.analysedVideo.plotTrajectory(ind, binaryFrameDir_pathObj, videoDir_pathObj, self.videoFPS, self.frameNameTemplate)
    
    def app2plotBubbleTrajectory(self, binaryFrameDir_pathObj, videoFramesDir_pathObj):
        bubbleListIndices     = self.params['bubbleListIndex']
        if self.params['bubbleListIndex'] != False:
            bubbleListIndices = range(self.params['bubbleListIndex'])
             
        self.analysedVideo.app2_plotTrajectory(bubbleListIndices, binaryFrameDir_pathObj, videoFramesDir_pathObj, self.videoFPS, self.frameNameTemplate)
    
    # def flowTypeParams(self, flowType):
    #     para = {}
    #     if flowType == 1:
    #         # fluidFlow1 crop_v1 parameters
    #         para['top']          = 100
    #         para['bottom']       = 700
    #         para['left']         = 100
    #         para['right']        = 300
    #         para['blockSize']    = 9
    #         para['constantSub']  = 5
    #         para['connectivity'] = 2
    #         para['minSize']      = 1
    #         return para
        
    #     elif flowType == 2:
    #         # fluidFlow2 crop_v1 parameters
    #         para['top']          = 100
    #         para['bottom']       = 750
    #         para['left']         = 105
    #         para['right']        = 200
    #         para['blockSize']    = 41
    #         para['constantSub']  = 7
    #         para['connectivity'] = 2
    #         para['minSize']      = 5
    #         para['distanceThreshold'] = 20
    #         para['C_O_KernelSize'] = 4
    #         para['sizeThresholdPercent'] = 0.05
            
    #         # Parameter testing
    #         para['distanceThreshold'] = 50
    #         para['sizeThresholdPercent'] = 0.5
    #         para['frameConsecThreshold'] = 7
    #         para['bubbleTrajectoryLengthThreshold'] = 5
    #         return para
        
    #     else:
    #         print("ERROR: Invalid flowType")
    #         return