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
        ## check if self.params['inputRotate'] is present
        if 'inputRotate' in self.params:
            readAndSaveVid(videoFramesDir_pathObj, 
                           self.videoFormat, 
                           self.frameNameTemplate, 
                           self.params['inputRotate'])
        
        readAndSaveVid(videoFramesDir_pathObj, 
                       self.videoFormat, 
                       self.frameNameTemplate)
    
    def getCroppedFrames(self, origFrameDir_pathObj, croppedFrameDir_pathObj):
        from utils import cropFrames
        cropFrames(origFrameDir_pathObj, 
                   croppedFrameDir_pathObj, 
                   self.frameNameTemplate, 
                   self.params)
    
    def getBinaryImages(self, origFrameDir_pathObj, binaryFrameDir_pathObj):
        from utils import processImages
        processImages(origFrameDir_pathObj, 
                      binaryFrameDir_pathObj, 
                      self.frameNameTemplate, 
                      self.params)
        
    def createVideoFromFrames(self, frameDir_pathObj):
        from utils import makeSingleVideo
        makeSingleVideo(frameDir_pathObj, 
                        self.frameNameTemplate, 
                        self.videoFPS)
          
    def createConcatVideo(self, lftFrameDir_pathObj, rhtFrameDir_pathObj, videoDir_pathObj):
        from utils import makeConcatVideos
        makeConcatVideos(lftFrameDir_pathObj, 
                         rhtFrameDir_pathObj, 
                         self.frameNameTemplate, 
                         videoDir_pathObj, 
                         self.videoFPS, 
                         self.params)    
    
    def extractFrameObjects(self, binaryFrameDir_pathObj, analysisBaseDir_pathObj):
        from video import Video
        from utils import dropAnalysis, getFrameNumbers_ordered, DoNumExistingFramesMatch
        
        self.analysedVideo = Video()
        # allFramesNum = getFrameNumbers_ordered(binaryFrameDir_pathObj, 
        #                                        self.frameNameTemplate)
        # condn1      = DoNumExistingFramesMatch(analysisBaseDir_pathObj/ "pixSize" / "frames" , 
        #                                        len(allFramesNum))
        # condn2      = DoNumExistingFramesMatch(analysisBaseDir_pathObj/ "vertPos" / "frames" , 
        #                                        len(allFramesNum))
        # condn3      = DoNumExistingFramesMatch(analysisBaseDir_pathObj/ "dynamicMarker" / "frames",
        #                                        len(allFramesNum))
        filePathObj = analysisBaseDir_pathObj / 'videoFrames.txt'
        condn4      = filePathObj.exists()
        
        # if condn1 and condn2 and condn3 and condn4:
        if condn4:
            self.analysedVideo.loadFramesFromTextFile(analysisBaseDir_pathObj)
            return 
        
        self.analysedVideo = dropAnalysis(binaryFrameDir_pathObj, 
                                          analysisBaseDir_pathObj, 
                                          self.frameNameTemplate, 
                                          self.params)
        self.analysedVideo.saveFramesToTextFile(analysisBaseDir_pathObj)
        
    
    def checkVideoFilesOnDisk(self, analysisBaseDir_pathObj):
        from video import Video
        newVideo = Video()
        newVideo.loadFramesFromTextFile(analysisBaseDir_pathObj)
        newVideo.loadBubblesFromTextFile(analysisBaseDir_pathObj)
        if newVideo.isSame(self.analysedVideo):
            print("Video analysis saved successfully")
        else:
            print("Video analysis not saved successfully")
            exit()
        
    def evaluateBubbleTrajectory(self, analysisBaseDir_pathObj):
        filePathObj = analysisBaseDir_pathObj / 'videoBubbles.txt'
        if filePathObj.exists():
            self.analysedVideo.loadBubblesFromTextFile(analysisBaseDir_pathObj)
            return
        
        self.analysedVideo.trackObjects(self.params)
        self.analysedVideo.saveBubblesToTextFile(analysisBaseDir_pathObj)
    
    def plotBubbleTrajectory(self, binaryFrameDir_pathObj, videoDir_pathObj):
        ''' 
        Creates separate videos for each bubble
        '''
        bubbleListIndex     = self.params['bubbleListIndex']
        if self.params['bubbleListIndex'] != False:
            bubbleListIndex = range(self.params['bubbleListIndex'])
        
        if bubbleListIndex == False:
            bubbleListIndex = range(self.analysedVideo.getNumBubbles())
        
        for ind in bubbleListIndex:
            self.analysedVideo.plotTrajectory(ind, 
                                              binaryFrameDir_pathObj, 
                                              videoDir_pathObj, 
                                              self.videoFPS, 
                                              self.frameNameTemplate)
    
    def markBubblesOnFrames(self, binaryFrameDir_pathObj, bubbleTrackFramesDir_pathObj):
        ''' 
        Creates a single video with all the bubble marked in it
        '''
        from utils import getFrameNumbers_ordered, DoNumExistingFramesMatch
        
        allFramesNum = getFrameNumbers_ordered(binaryFrameDir_pathObj, self.frameNameTemplate)
        condn1      = DoNumExistingFramesMatch(bubbleTrackFramesDir_pathObj, len(allFramesNum))
        if condn1:
            return
        
        bubbleListIndices     = self.params['bubbleListIndex']
        if self.params['bubbleListIndex'] != False:
            bubbleListIndices = range(self.params['bubbleListIndex'])
             
        self.analysedVideo.app2_plotTrajectory(bubbleListIndices, 
                                               binaryFrameDir_pathObj, 
                                               bubbleTrackFramesDir_pathObj, 
                                               self.videoFPS, 
                                               self.frameNameTemplate)
    
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