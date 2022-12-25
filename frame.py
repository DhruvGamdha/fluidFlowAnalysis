import numpy as np

class Frame:
    def __init__(self):
        self.frameNumber = None
        self.objCount   = None
        self.objects    = []
        
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
        self.objects.pop(objectIndex)
    
    def removeObject_object(self, object):
        self.objects.remove(object)
    
    def removeObject_indices(self, objectIndices):
        for i in objectIndices:
            self.objects.pop(i)
    
    def getNearbyObjectIndices_object(self, object, distance):
        objectsIndices = []
        for i in range(len(self.objects)):
            if np.linalg.norm(self.objects[i].getPosition() - object.getPosition()) <= distance:
                objectsIndices.append(i)
        return objectsIndices
    
    def getNearbyObjectIndices_position(self, position, distance):
        objectsIndices = []
        for i in range(len(self.objects)):
            if np.linalg.norm(self.objects[i].getPosition() - position) <= distance:
                objectsIndices.append(i)
        return objectsIndices
    
    def getComparableSizeObjectIndices_object(self, object, sizeThreshold):
        objectsIndices = []
        for i in range(len(self.objects)):
            if abs(self.objects[i].getSize() - object.getSize()) <= sizeThreshold:
                objectsIndices.append(i)
        return objectsIndices
    
    def getComparableSizeObjectIndices_size(self, size, sizeThreshold):
        objectsIndices = []
        for i in range(len(self.objects)):
            if abs(self.objects[i].getSize() - size) <= sizeThreshold:
                objectsIndices.append(i)
        return objectsIndices
    
    def getNearbyAndComparableSizeObjectIndices_object(self, object, distance, sizeThreshold):
        nearbyObjectsIndices            = self.getNearbyObjectIndices_object(object, distance)
        comparableSizeObjectsIndices    = self.getComparableSizeObjectIndices_object(object, sizeThreshold)
        return list(set(nearbyObjectsIndices).intersection(comparableSizeObjectsIndices))
    
    def getNearbyAndComparableSizeObjectIndices_positionAndSize(self, position, size, distance, sizeThreshold):
        nearbyObjectsIndices            = self.getNearbyObjectIndices_position(position, distance)
        comparableSizeObjectsIndices    = self.getComparableSizeObjectIndices_size(size, sizeThreshold)
        return list(set(nearbyObjectsIndices).intersection(comparableSizeObjectsIndices))
    
    def getObjectPositionList(self):
        x = []
        y = []
        for object in self.objects:
            pos = object.getPosition()
            x.append(pos[0])
            y.append(pos[1])
        return x, y
    
    def getObjectSizeList(self):
        size = []
        for object in self.objects:
            size.append(object.getSize())
        return size