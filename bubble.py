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
    
    def getLocation(self, index):
        return self.trajectory[index]
    
    def getLatestLocation_FrameNumber(self):
        return self.trajectory[-1][0]
    
    def getLatestLocation_ObjectIndex(self):
        return self.trajectory[-1][1]
    
    def getTrajectoryLength(self):
        return len(self.trajectory)
    
    def isTrajectoryContinuous(self):
        for i in range(1, len(self.trajectory)):
            if self.trajectory[i][0] - self.trajectory[i-1][0] != 1:
                return False
        return True
    
    def isSame(self, bubble):
        cond1 = self.bubbleIndex == bubble.getBubbleIndex()
        cond2 = self.trajectory == bubble.getFullTrajectory()
        return cond1 and cond2