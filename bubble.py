
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
        self.trajectory = []       # Will be a list of [frameNumber, objectIndex]
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
    
    # Functions to get and set bubble positions, velocities, and accelerations
    def setPositions_fullTrajectory(self, positions):
        # Check if the number of positions is the same as the number of frames
        if len(positions) != len(self.trajectory):
            raise ValueError("Number of positions does not match the number of frames.")
        
        self.positions = positions
        
    def getPositions_fullTrajectory(self):
        return self.positions
    
    def setVelocities_fullTrajectory(self, velocities):
        # Check if the number of velocities is the same as 1 less than the number of frames
        if len(velocities) != len(self.trajectory) - 1:
            raise ValueError("Number of velocities does not match the number of frames.")
        
        self.velocities = velocities
        
    def getVelocities_fullTrajectory(self):
        return self.velocities
    
    def setAccelerations_fullTrajectory(self, accelerations):
        # Check if the number of accelerations is the same as 2 less than the number of frames
        if len(accelerations) != len(self.trajectory) - 2:
            raise ValueError("Number of accelerations does not match the number of frames.")
        
        self.accelerations = accelerations
        
    def getAccelerations_fullTrajectory(self):
        return self.accelerations
    
    def getLatestPosition(self):
        return self.positions[-1]
    
    def getLatestVelocity(self):
        return self.velocities[-1]
    
    def getLatestAcceleration(self):
        return self.accelerations[-1]
    
    # Get the bubble position, velocities and acceleration at a given index
    def getPosition_atIndex(self, index):
        return self.positions[index]
    
    def getVelocity_atIndex(self, index):
        return self.velocities[index]
    
    def getAcceleration_atIndex(self, index):
        return self.accelerations[index]
    
    # Get the bubble position, velocities and acceleration at a given frame number
    def getPosition_atFrameNumber(self, frameNumber):
        for i in range(len(self.trajectory)):
            if self.trajectory[i][0] == frameNumber:
                return self.positions[i]
        return None
