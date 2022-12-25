import numpy as np
import pathlib as pl
from object import Object
from frame import Frame
class Video:
    def __init__(self):
        self.frames = []                # All Frames object orderly placed in a list 
        self.bubbles = []               # All Bubble object orderly placed in a list
        
    def addFrame(self, frame):
        self.frames.append(frame)
        
    def getFrame(self, frameIndex):
        return self.frames[frameIndex]
    
    def getFrames(self, frameIndices):
        return [self.frames[i] for i in frameIndices]
    
    def getAllFrames(self):
        return self.frames
    
    def getNumFrames(self):
        return len(self.frames)
    
    def saveToTextFile(self, saveDir_pathObj):
        # Create a text file to save the video data
        saveDir_pathObj.mkdir(parents=True, exist_ok=True)
        savePath = saveDir_pathObj / 'videoAnalysis.txt'
        saveFile = open(savePath, 'w')
        
        # Save the video data
        saveFile.write('Total frames: ' + str(self.getNumFrames()) + '\n')
        
        for frameInd in range(self.getNumFrames()):
            frame = self.getFrame(frameInd)
            saveFile.write('FrameNumber: ' + str(frame.getFrameNumber())) 
            saveFile.write('\t' + 'TotalObjects: ' + str(frame.getObjectCount()))
            saveFile.write('\n')
            for j in range(frame.getObjectCount()):
                obj = frame.getObject(j)
                
                saveFile.write('\t' + 'FrameNumber: ' + str(obj.getFrameNumber()))
                saveFile.write('\t' + 'ObjectIndex: ' + str(obj.getObjectIndex()))
                saveFile.write('\t' + 'Position: [' + str(obj.getX()) + ' ' + str(obj.getY()) + ']')
                saveFile.write('\t' + 'Size: ' + str(obj.getSize()))
                saveFile.write('\n')
                
        saveFile.close()
        assert savePath.exists()
         
    def loadFromTextFile(self, loadDir_pathObj):
         # Load the video analysis data from the text file
        loadPath = loadDir_pathObj / 'videoAnalysis.txt'
        loadFile = open(loadPath, 'r')
        
        # Read the video data
        lines = loadFile.readlines()
        self.frames = []
        
        lineIndex = 0
        totalFrames = int(lines[lineIndex].split()[2])  # Read the total number of frames from the first line
        
        # Read the data of each frame
        for i in range(totalFrames):
            lineIndex += 1
            frame = Frame()
            frameNumber = int(lines[lineIndex].split()[1])
            objectCount = int(lines[lineIndex].split()[3])
            frame.setFrameNumber(frameNumber)
            frame.setObjectCount(objectCount)
            for j in range(objectCount):
                lineIndex += 1
                objFrameNumber = int(lines[lineIndex].split()[1])
                objIndex = int(lines[lineIndex].split()[3])
                position = np.array([int(lines[lineIndex].split()[5][1:]), int(lines[lineIndex].split()[6][:-1])])
                size = int(lines[lineIndex].split()[8])
                 
                # Create a new object and add it to the frame
                newObject = Object(objFrameNumber, objIndex, position[0], position[1], size)
                frame.addObject(newObject)
                
            # Add the frame to the video
            self.frames.append(frame)            
        loadFile.close()
        assert self.checkAnalysisFileExists(loadDir_pathObj)
    
    def checkAnalysisFileExists(self, loadDir_pathObj):
        loadPath = loadDir_pathObj / 'videoAnalysis.txt'
        if loadPath.exists():
            return True
        return False
    
    # def trackObjects():