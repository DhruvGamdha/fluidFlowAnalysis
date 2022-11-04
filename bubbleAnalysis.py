# Class for bubble analysis
import os
class bubbleAnalysis:
    def __init__(self, newResultsDir, videoDirPath, videoName, baseResultDirPath, resultsDir):
        # Create a dictionary to store all the path names
        self.DirNames = {
            1: 'videoDirPath',
            2: 'vidFramesDirPath',
            3: 'baseResultDirPath',
            4: 'resultDirPath'
        } 
        
        self.paths                      = {}            # Dictionary of paths to the directories and files
        self.paths[self.DirNames[1]]    = videoDirPath 
        self.paths[self.DirNames[2]]    = os.path.join(self.paths[self.DirNames[1]], 'frames')
        self.paths[self.DirNames[3]]    = baseResultDirPath
        self.videoName                  = videoName
        self.subResultDirNames          = ['binary', 'binary/all', 'binary/all/frames', 'analysis', 'analysis/pixSize', 'analysis/vertPos', 'analysis/dynamicMarker']
        self.videoFPS                   = 60.0          # FPS of the video
        self.frameNameTemplate          = 'frame%d.png' # Name template of the frames
        self.flowType                   = 2             # 1: fluidFlow1, 2: fluidFlow2
        self.newResultsDir              = newResultsDir
        
        if self.newResultsDir:
            self.resultsDirTemplate  = 'version'
        else:    
            if not os.path.exists(os.path.join(baseResultDirPath, resultsDir)):     # Check if the results directory exists
                exit("ERROR: Results directory does not exist")
            else:
                self.resultsDirTemplate  = resultsDir
        self.createResultDirs()
        
    def createResultDirs(self):
        from directories import directories
        dirObj = directories(self.paths[self.DirNames[3]], self.resultsDirTemplate, self.subResultDirNames, self.newResultsDir)
        
        self.paths[self.DirNames[4]] = dirObj.getVerDirPath()
        self.paths.update(dirObj.getSubDirPaths())
        
    def getFramesFromVideo(self):
        from utils import readAndSaveVid
        videoPath = os.path.join(self.paths[self.DirNames[1]], self.videoName)
        readAndSaveVid(videoPath, self.paths[self.DirNames[2]])
    
    def getBinaryImages(self):
        from utils import processImages
        params = self.flowTypeParams(self.flowType)
        processImages(self.paths[self.DirNames[2]], self.paths['binary/all/frames'], params)
        
    def createVideoFromFrames(self, frameDir, videoDir):
        from utils import makeSingleVideo
        from os.path import join
        videoPath = join(self.paths[videoDir], 'video.mp4')
        makeSingleVideo(self.paths[frameDir], self.frameNameTemplate, videoPath, self.videoFPS)
          
    def createConcatVideo(self, lftFrameDir, rhtFrameDir, videoDir):
        from utils import makeConcatVideos
        params = self.flowTypeParams(self.flowType)
        videoLocAndName_mp4 = os.path.join(self.paths[videoDir], 'video.mp4')
        makeConcatVideos(self.paths[lftFrameDir], self.paths[rhtFrameDir], self.frameNameTemplate, videoLocAndName_mp4, self.videoFPS, params)    
    
    def binaryAnalysis(self):
        from utils import dropAnalysis
        params = self.flowTypeParams(self.flowType)
        dropAnalysis(self.paths['binary/all/frames'], self.paths['analysis'], self.frameNameTemplate, params)
    
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