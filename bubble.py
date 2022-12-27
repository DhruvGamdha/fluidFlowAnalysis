import numpy as np

class Bubble:
    def __init__(self, bubbleIndex):
        self.trajectory     = []                # 2D List of [frameNumber, objectIndex]
        self.bubbleIndex    = bubbleIndex       # bubble index is the index of the bubble in the bubble list
        
    def appendTrajectory(self, frameNumber, objectIndex):
        self.trajectory.append([frameNumber, objectIndex])
    
    def setBubbleIndex(self, bubbleIndex):
        self.bubbleIndex = bubbleIndex
    
    def getFullTrajectory(self):
        return self.trajectory
    
    def getBubbleIndex(self):
        return self.bubbleIndex
    
    def getLatestLocation(self):
        return self.trajectory[-1]
    
    def getLatestLocation_FrameNumber(self):
        return self.trajectory[-1][0]
    
    def getLatestLocation_ObjectIndex(self):
        return self.trajectory[-1][1]
    
    def getTrajectoryLength(self):
        return len(self.trajectory)