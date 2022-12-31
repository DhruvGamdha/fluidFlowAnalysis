# Object class: To store the detected object information in the frame
import numpy as np

class Object:
    def __init__(self, frameNumber, objectIndex, x, y, size, rows, cols):
        self.frameNumber = frameNumber
        self.objectIndex = objectIndex
        self.position   = np.array([x, y])
        self.size = size
        self.pixelLocs = np.array([rows, cols])

        # Update position to the center of mass
        x, y = self.getCenterOfMass()
        self.position = np.array([x, y])
        
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
    
    def getCenterOfMass(self):
        # Get the pixel locations
        rows = self.pixelLocs[0]
        cols = self.pixelLocs[1]
        
        # Get the center of mass
        x = int(round(np.mean(cols)))
        y = int(round(np.mean(rows)))
        return x, y
    
    def getPixelLoc(self, index):
        row = self.pixelLocs[0][index]
        col = self.pixelLocs[1][index]
        return row, col
    
    def getAllPixelLocs(self):
        rows = self.pixelLocs[0]
        cols = self.pixelLocs[1]
        return rows, cols
    
    def isSame(self, object):
        cond1 = self.frameNumber == object.getFrameNumber()
        cond2 = self.objectIndex == object.getObjectIndex()
        cond3 = self.position[0] == object.getX() and self.position[1] == object.getY()
        cond4 = self.size == object.getSize()
        
        # Check if the pixel locations are the same
        objrow = object.getAllPixelLocs()[0]
        objcol = object.getAllPixelLocs()[1]
        for i in range(len(self.pixelLocs[0])):
            selfrow_i = self.pixelLocs[0][i]
            selfcol_i = self.pixelLocs[1][i]
            objrow_i = objrow[i]
            objcol_i = objcol[i]
            cond5 = selfrow_i == objrow_i and selfcol_i == objcol_i
            if not cond5:
                break
        return cond1 and cond2 and cond3 and cond4 and cond5