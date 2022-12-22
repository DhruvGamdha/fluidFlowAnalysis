# Class for bubble analysis
import os
class bubbleAnalysis:
    def __init__(self, fps, frameNameTemplate, flowType, videoFormat):
        # Create a dictionary to store all the path names
        self.videoFPS                   = fps          # FPS of the video
        self.frameNameTemplate          = frameNameTemplate # Name template of the frames
        self.flowType                   = flowType             # 1: fluidFlow1, 2: fluidFlow2
        self.videoFormat                = videoFormat
        
    def getFramesFromVideo(self, videoFramesDir_pathObj):
        from utils import readAndSaveVid
        readAndSaveVid(videoFramesDir_pathObj, self.videoFormat, self.frameNameTemplate)
    
    def getBinaryImages(self, origFrameDir_pathObj, binaryFrameDir_pathObj):
        from utils import processImages
        params = self.flowTypeParams(self.flowType)
        processImages(origFrameDir_pathObj, binaryFrameDir_pathObj, self.frameNameTemplate, params)
        
    def createVideoFromFrames(self, frameDir_pathObj):
        from utils import makeSingleVideo
        from os.path import join
        makeSingleVideo(frameDir_pathObj, self.frameNameTemplate, self.videoFPS)
          
    def createConcatVideo(self, lftFrameDir_pathObj, rhtFrameDir_pathObj, videoDir_pathObj):
        from utils import makeConcatVideos
        params = self.flowTypeParams(self.flowType)
        makeConcatVideos(lftFrameDir_pathObj, rhtFrameDir_pathObj, self.frameNameTemplate, videoDir_pathObj, self.videoFPS, params)    
    
    def binaryAnalysis(self, binaryFrameDir_pathObj, analysisBaseDir_pathObj):
        from utils import dropAnalysis
        params = self.flowTypeParams(self.flowType)
        dropAnalysis(binaryFrameDir_pathObj, analysisBaseDir_pathObj, self.frameNameTemplate, params)
    
    def flowTypeParams(self, flowType):
        para = {}
        if flowType == 1:
            # fluidFlow1 crop_v1 parameters
            para['top']          = 100
            para['bottom']       = 700
            para['left']         = 100
            para['right']        = 300
            para['blockSize']    = 9
            para['constantSub']  = 5
            para['connectivity'] = 2
            para['minSize']      = 1
            ''' 
            # fluidFlow1 crop_v3 parameters
            para['top']     = 120
            para['bottom']  = 700
            para['left']    = 150
            para['right']   = 250 
            '''   
            return para
        
        elif flowType == 2:
            # fluidFlow2 crop_v1 parameters
            para['top']          = 100
            para['bottom']       = 750
            para['left']         = 105
            para['right']        = 200
            para['blockSize']    = 41
            para['constantSub']  = 7
            para['connectivity'] = 1
            para['minSize']      = 1
            return para
        
        else:
            print("ERROR: Invalid flowType")
            return