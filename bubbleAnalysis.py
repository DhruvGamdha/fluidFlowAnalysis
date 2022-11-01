# Class for bubble analysis
import os
class bubbleAnalysis:
    def __init__(self, videoPath, videoName, baseResultDirPath):
        self.paths                      = {}            # Dictionary of paths to the directories and files
        self.paths['videoDirPath']      = videoPath 
        self.paths['vidFramesDirPath']  = os.path.join(self.paths['videoDirPath'], 'frames')
        self.paths['baseResultDirPath'] = baseResultDirPath
        
        self.videoName          = videoName
        self.resVersionDirName  = 'version'
        self.subResultDirNames  = ['binary', 'binary/all', 'binary/all/frames', 'analysis']
        self.subResDirPaths     = None          # dictionary of sub directories paths
        
        self.videoFPS           = 60.0          # FPS of the video
        self.frameNameTemplate  = 'frame%4d.png' # Name template of the frames
        self.flowType           = 2             # 1: fluidFlow1, 2: fluidFlow2
        self.createResultDirs()
        
    def createResultDirs(self):
        from directories import directories
        dirObj = directories(self.paths['baseResultDirPath'], self.resVersionDirName, self.subResultDirNames)
        
        self.paths['resultDirPath'] = dirObj.getVerDirPath()
        self.paths.update(dirObj.getSubDirPaths())
        
    def getFramesFromVideo(self):
        from utils import readAndSaveVid
        videoPath = os.path.join(self.paths['videoDirPath'], self.videoName)
        readAndSaveVid(videoPath, self.paths['vidFramesDirPath'])
    
    def getBinaryImages(self):
        from utils import processImages
        para = self.cropParameters(self.flowType)
        processImages(self.paths['vidFramesDirPath'], self.subResDirPaths['binary/all/frames'], para['top'], para['bottom'], para['left'], para['right'])
        
    def createVideoFromFrames(self, frameDir, videoDir):
        from utils import makeSingleVideo
        from os.path import join
        videoPath = join(videoDir, 'video.mp4')
        makeSingleVideo(self.subResDirPaths[frameDir], self.frameNameTemplate, videoPath, self.videoFPS)
          
    def createConcatVideo(self, lftFrameDir, rhtFrameDir, videoDir):
        from utils import makeConcatVideos
        para = self.cropParameters(self.flowType)
        videoLocAndName_mp4 = os.path.join(videoDir, 'video.mp4')
        makeConcatVideos(self.paths[lftFrameDir], self.paths[rhtFrameDir], self.frameNameTemplate, videoLocAndName_mp4, self.videoFPS, para['top'], para['bottom'], para['left'], para['right'])    
    
    def binaryAnalysis(self):
        from utils import dropAnalysis
        para = self.cropParameters(self.flowType)
        dropAnalysis(self.paths['binary/all/frames'], self.paths['analysis'], self.frameNameTemplate, para['connectivity'])
    
    def cropParameters(self, flowType):
        # Crop the frame from center corresponding to below values, original size is 400x800
        para = {}
        if flowType == 1:
            # fluidFlow1 crop_v1 parameters
            para['top']     = 100
            para['bottom']  = 700
            para['left']    = 100
            para['right']   = 300
            para['i']       = 9
            para['j']       = 5
            para['connectivity'] = 2
            para['minSize'] = 1
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
            para['top']     = 100
            para['bottom']  = 750
            para['left']    = 105
            para['right']   = 200
            para['i']       = 41
            para['j']       = 7
            para['connectivity'] = 1
            para['minSize'] = 1
            
            return para
        
        else:
            print("ERROR: Invalid flowType")
            return