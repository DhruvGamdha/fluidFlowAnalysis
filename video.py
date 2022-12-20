import numpy as np

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