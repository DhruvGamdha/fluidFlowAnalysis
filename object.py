import numpy as np

class Object:
    """
    A class representing a detected object within a frame.

    Attributes
    ----------
    frameNumber : int
        The frame number where this object was detected.
    objectIndex : int
        The index of the object within the frame.
    topLft_position : np.ndarray
        The top-left position of the object's bounding box.
    size : int
        The size (in pixels) of the object.
    pixelLocs : np.ndarray
        2D array of pixel coordinates (rows, cols) for this object.
    position : np.ndarray
        The object's center of mass (x, y).
    """

    def __init__(self, frameNumber, objectIndex, topLft_x, topLft_y, size, rows, cols):
        self.frameNumber = frameNumber
        self.objectIndex = objectIndex
        self.topLft_position = np.array([topLft_x, topLft_y])
        self.size = size
        self.pixelLocs = np.array([rows, cols])
        self.position = self.getCenterOfMass()

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
        rows, cols = self.pixelLocs
        x = int(round(np.mean(cols)))
        y = int(round(np.mean(rows)))
        return np.array([x, y])

    def getPixelLoc(self, index):
        row = self.pixelLocs[0][index]
        col = self.pixelLocs[1][index]
        return row, col

    def getAllPixelLocs(self):
        return self.pixelLocs[0], self.pixelLocs[1]

    def isSame(self, obj: 'Object'):
        cond1 = self.frameNumber == obj.getFrameNumber()
        cond2 = self.objectIndex == obj.getObjectIndex()
        cond3 = (self.position[0] == obj.getX()) and (self.position[1] == obj.getY())
        cond4 = self.size == obj.getSize()

        selfRows, selfCols = self.getAllPixelLocs()
        objRows, objCols = obj.getAllPixelLocs()

        if len(selfRows) != len(objRows):
            return False

        for i in range(len(selfRows)):
            if selfRows[i] != objRows[i] or selfCols[i] != objCols[i]:
                return False

        return cond1 and cond2 and cond3 and cond4
