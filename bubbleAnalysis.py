# Class for bubble analysis
import os
class bubbleAnalysis:
    def __init__(self, baseDataDir, baseResultDir, resultsDirIndex, dataDirIndex):
        # Create a dictionary to store all the path names
        self.DirNames = {
            1: 'baseDataDir',   # Input
            2: 'Inp_VideoDir',  # Input
            4: 'baseResultDir', # Output
            5: 'resultsVerDir', # Output
        } 
        self.inputPaths                 = {}
        self.outputPaths                = {}
        self.inputPaths[self.DirNames[1]]  = baseDataDir
        self.inputPaths[self.DirNames[2]]  = os.path.join(self.inputPaths[self.DirNames[1]], 'video')
        
        self.outputPaths[self.DirNames[4]]    = baseResultDir
        self.dataDirNames               = ['version_', 'frames']
        self.resultsDirNames            = ['version_', 
                                           'binary', 
                                           'binary/all', 
                                           'binary/all/frames', 
                                           'analysis', 
                                           'analysis/pixSize', 
                                           'analysis/pixSize/frames', 
                                           'analysis/vertPos', 
                                           'analysis/vertPos/frames', 
                                           'analysis/dynamicMarker', 
                                           'analysis/dynamicMarker/frames']
        
        self.videoFPS                   = 30.0          # FPS of the video
        self.frameNameTemplate          = 'frame%d.png' # Name template of the frames
        self.flowType                   = 2             # 1: fluidFlow1, 2: fluidFlow2
        self.resultsDirIndex            = resultsDirIndex
        self.dataDirIndex               = dataDirIndex
        self.videoFormat                = '.avi'
        
        self.createDataDir()
        self.createResultDirs()
        
    def createDataDir(self):
        from directories import directories
        dirObj = directories(self.inputPaths[self.DirNames[1]], self.dataDirNames, self.dataDirIndex)
        self.dataDirNames = dirObj.getDirNames()
        self.inputPaths.update(dirObj.getDirPaths())
        
    def createResultDirs(self):
        from directories import directories
        dirObj = directories(self.outputPaths[self.DirNames[4]], self.resultsDirNames, self.resultsDirIndex)
        self.resultsDirNames = dirObj.getDirNames()
        self.outputPaths.update(dirObj.getDirPaths())
        
    def getFramesFromVideo(self):
        from utils import readAndSaveVid
        videoDirPath = self.inputPaths[self.DirNames[2]]
        saveFramePath = self.inputPaths['frames']
        readAndSaveVid(videoDirPath, saveFramePath, self.videoFormat)
    
    def getBinaryImages(self):
        from utils import processImages
        params = self.flowTypeParams(self.flowType)
        processImages(self.paths[self.DirNames[2]], self.paths['binary/all/frames'], self.frameNameTemplate, params)
        
    def createVideoFromFrames(self, frameDir, videoDir):
        from utils import makeSingleVideo
        from os.path import join
        videoPath = join(self.outputPaths[videoDir], 'videoIsolated.mp4')
        makeSingleVideo(self.outputPaths[frameDir], self.frameNameTemplate, videoPath, self.videoFPS)
          
    def createConcatVideo(self, lftFrameDir, rhtFrameDir, videoDir):
        from utils import makeConcatVideos
        params = self.flowTypeParams(self.flowType)
        videoLocAndName_mp4 = os.path.join(self.outputPaths[videoDir], 'videoPaired.mp4')
        makeConcatVideos(self.outputPaths[lftFrameDir], self.outputPaths[rhtFrameDir], self.frameNameTemplate, videoLocAndName_mp4, self.videoFPS, params)    
    
    def binaryAnalysis(self):
        from utils import dropAnalysis
        params = self.flowTypeParams(self.flowType)
        dropAnalysis(self.outputPaths['binary/all/frames'], self.outputPaths['analysis'], self.frameNameTemplate, params)
    
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