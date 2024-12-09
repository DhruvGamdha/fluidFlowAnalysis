import numpy as np
import logging

class Frame:
    """
    A class representing a single frame containing multiple objects.

    Attributes
    ----------
    frameNumber : int
        The frame number.
    objCount : int
        Number of objects in the frame.
    objects : list of Object
        List of objects detected in this frame.
    """

    def __init__(self):
        self.frameNumber = None
        self.objCount = None
        self.objects = []

    def setFrameNumber(self, frameNumber):
        self.frameNumber = frameNumber

    def addObject(self, object):
        self.objects.append(object)

    def setObjectCount(self, objCount):
        self.objCount = objCount

    def getFrameNumber(self):
        return self.frameNumber

    def getObjectCount(self):
        return self.objCount

    def updateObjectCount(self):
        self.objCount = len(self.objects)

    def getAllObjects(self):
        return self.objects

    def getObject(self, objectIndex):
        return self.objects[objectIndex]

    def getObjects(self, objectIndices):
        return [self.objects[i] for i in objectIndices]

    def removeObject_index(self, objectIndex):
        try:
            self.objects.pop(objectIndex)
            self.updateObjectCount()
        except IndexError:
            logging.warning("Attempted to remove invalid object index %d.", objectIndex)

    def getNearbyObjectIndices_object(self, obj, distance):
        """
        Returns indices of objects within a certain distance of the given object's position, sorted by distance.
        """
        objectsIndices = {}
        for i, candidate_obj in enumerate(self.objects):
            dist = np.linalg.norm(candidate_obj.getPosition() - obj.getPosition())
            if dist <= distance:
                objectsIndices[i] = dist
        # Sort by distance
        return [k for k, v in sorted(objectsIndices.items(), key=lambda item: item[1])]

    def getComparableSizeObjectIndices_object(self, obj, sizeThresholdPercent):
        """
        Returns indices of objects with sizes within a certain percentage threshold of the given object's size, sorted by size.
        """
        objectsIndices = {}
        sizeThreshold = sizeThresholdPercent * obj.getSize()
        for i, candidate_obj in enumerate(self.objects):
            if abs(candidate_obj.getSize() - obj.getSize()) <= sizeThreshold:
                objectsIndices[i] = candidate_obj.getSize()

        return [k for k, v in sorted(objectsIndices.items(), key=lambda item: item[1])]

    def getNearbyAndComparableSizeObjectIndices_object(self, obj, distance, sizeThresholdPercent):
        nearbyObjectsIndices = self.getNearbyObjectIndices_object(obj, distance)
        comparableSizeObjectsIndices = self.getComparableSizeObjectIndices_object(obj, sizeThresholdPercent)

        # Intersection maintaining order of nearbyObjectsIndices
        indices = [i for i in nearbyObjectsIndices if i in comparableSizeObjectsIndices]
        return indices

    def getObjectPositionList(self):
        x, y = [], []
        for o in self.objects:
            pos = o.getPosition()
            x.append(pos[0])
            y.append(pos[1])
        return x, y

    def getObjectSizeList(self):
        return [o.getSize() for o in self.objects]

    def copy(self):
        """
        Create a shallow copy of the frame. Objects are not cloned (same references).
        """
        newFrame = Frame()
        newFrame.setFrameNumber(self.frameNumber)
        newFrame.setObjectCount(self.objCount)
        for obj in self.objects:
            newFrame.addObject(obj)
        return newFrame

    def isSame(self, frame):
        """
        Check if another Frame is the same as this one.
        """
        if self.frameNumber != frame.getFrameNumber():
            return False
        if self.objCount != frame.getObjectCount():
            return False
        if self.objCount != len(frame.getAllObjects()):
            return False
        for i in range(self.objCount):
            if not self.objects[i].isSame(frame.getObject(i)):
                return False
        return True
