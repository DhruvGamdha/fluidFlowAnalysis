import numpy as np
import pathlib as pl
from object import Object
from frame import Frame
class Video:
    def __init__(self):
        self.frames = []                # All Frames object orderly placed in a list 
        self.objCount_eachFrame = []    # Number of objects in each frame, ordered by frame index
        
        
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
    
    def addFrameObjCount(self, objCount):
        self.objCount_eachFrame.append(objCount)
        
    def getFrameObjCount(self, frameIndex):
        return self.objCount_eachFrame[frameIndex]
    
    def getFramesObjCounts(self, frameIndices):
        return [self.objCount_eachFrame[i] for i in frameIndices]
    
    def getObjCountList(self):
        return self.objCount_eachFrame
    
    def saveToTextFile(self, saveDir_pathObj):
        # Create a text file to save the video data
        saveDir_pathObj.mkdir(parents=True, exist_ok=True)
        savePath = saveDir_pathObj / 'videoAnalysis.txt'
        saveFile = open(savePath, 'w')
        
        # Save the video data
        saveFile.write('Total frames: ' + str(self.getNumFrames()) + '\n')
        
        for i in range(self.getNumFrames()):
            saveFile.write('Frame: ' + str(i) + '\t' + 'Total objects: ' + str(self.getFrameObjCount(i)) + '\n')
            for j in range(self.getFrameObjCount(i)):
                # Write position using obj.getX() and obj.getY()
                saveFile.write('\t' + 'Object: ' + str(j) + '\t' + 'Position: [' + str(self.getFrame(i).getObject(j).getX()) + ' ' + str(self.getFrame(i).getObject(j).getY()) + ']' + '\t' + 'Size: ' + str(self.getFrame(i).getObject(j).getSize()) + '\n') 
                # saveFile.write('\t' + 'Object: ' + str(j) + '\t' + 'Position: ' + str(self.getFrame(i).getObject(j).getPosition()) + '\t' + 'Size: ' + str(self.getFrame(i).getObject(j).getSize()) + '\n')
                
        saveFile.close()
         
    def loadFromTextFile(self, loadDir_pathObj):
         # Load the video analysis data from the text file
        loadPath = loadDir_pathObj / 'videoAnalysis.txt'
        loadFile = open(loadPath, 'r')
        
        # Read the video data
        lines = loadFile.readlines()
        self.frames = []
        self.objCount_eachFrame = []
        
        lineIndex = 0
        totalFrames = int(lines[lineIndex].split()[2])  # Read the total number of frames from the first line
        
        # Read the data of each frame
        for i in range(totalFrames):
            lineIndex += 1
            # Read the Totals objects in the frame written as: saveFile.write('Frame: ' + str(i) + '\t' + 'Total objects: ' + str(self.getFrameObjCount(i)) + '\n')
            totalObjects = int(lines[lineIndex].split()[4])
            self.objCount_eachFrame.append(totalObjects)
            frame = Frame()
            for j in range(totalObjects):
                lineIndex += 1
                # Read Position and Size value from the line example; Object: 0	Position: [45 28]	Size: 333
                position = np.array([int(lines[lineIndex].split()[3][1:]), int(lines[lineIndex].split()[4][:-1])])
                size = int(lines[lineIndex].split()[6])
                 
                # Create a new object and add it to the frame
                newObject = Object(position[0], position[1], size)
                frame.addObject(newObject)
                
            # Add the frame to the video
            self.frames.append(frame)            
        loadFile.close()