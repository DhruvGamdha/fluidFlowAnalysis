# Object class: To store the detected object information in the frame
import numpy as np

class Object:
    def __init__(self, frameNumber, objectIndex, x, y, size):
        self.frameNumber = frameNumber
        self.objectIndex = objectIndex
        self.position = np.array([x, y])
        self.size = size
    
    def setFrameNumber(self, frameNumber):
        self.frameNumber = frameNumber
    
    def setObjectIndex(self, objectIndex):
        self.objectIndex = objectIndex
    
    def setPosition(self, x, y):
        self.position = np.array([x, y])
    
    def setSize(self, size):
        self.size = size
        
    def getFrameNumber(self):
        return self.frameNumber
    
    def getObjectIndex(self):
        return self.objectIndex
        
    def getPosition(self):
        return self.position
    
    def getX(self):
        return self.position[0]
    
    def getY(self):
        return self.position[1]
    
    def getSize(self):
        return self.size