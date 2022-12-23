# Object class: To store the detected object information in the frame
import numpy as np

class Object:
    def __init__(self, x, y, size):
        self.position = np.array([x, y])
        self.size = size
    
    def setPosition(self, x, y):
        self.position = np.array([x, y])
    
    def setSize(self, size):
        self.size = size
        
    def getPosition(self):
        return self.position
    
    def getX(self):
        return self.position[0]
    
    def getY(self):
        return self.position[1]
    
    def getSize(self):
        return self.size