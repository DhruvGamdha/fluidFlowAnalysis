
class Bubble:
    """
    A class representing a bubble's trajectory across frames.

    Attributes
    ----------
    trajectory : list of [frameNumber, objectIndex]
        The sequence of (frameNumber, objectIndex) pairs representing the bubble's path.
    bubbleIndex : int
        Unique index identifying the bubble.
    """

    def __init__(self, bubbleIndex: int):
        self.trajectory = []
        self.bubbleIndex = bubbleIndex
        self.positions = None      # Will be a Nx2 array (x, y)
        self.velocities = None     # Will be a Nx2 array (vx, vy)
        self.accelerations = None  # Will be a Mx2 array (ax, ay)

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

    def isSame(self, bubble: 'Bubble'):
        cond1 = self.bubbleIndex == bubble.getBubbleIndex()
        cond2 = self.trajectory == bubble.getFullTrajectory()
        return cond1 and cond2
