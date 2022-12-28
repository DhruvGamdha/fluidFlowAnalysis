import numpy as np
import pathlib as pl
from object import Object
from frame import Frame
from bubble import Bubble
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
class Video:
    def __init__(self):
        self.frames = []                # All Frames object orderly placed in a list 
        self.bubbles = []               # All Bubble placed in a list
        
    def addFrame(self, frame):
        self.frames.append(frame)
        
    def getFrame(self, frameIndex):
        return self.frames[frameIndex]
    
    def getFrames(self, frameIndices):
        return [self.frames[i] for i in frameIndices]
    
    def getAllFrames(self):
        return self.frames
    
    def getNumFrames(self):
        return len(self.frames)
    
    def saveToTextFile(self, saveDir_pathObj):
        # Create a text file to save the video data
        saveDir_pathObj.mkdir(parents=True, exist_ok=True)
        savePath = saveDir_pathObj / 'videoAnalysis.txt'
        saveFile = open(savePath, 'w')
        
        # Save the video data
        saveFile.write('Total frames: ' + str(self.getNumFrames()) + '\n')
        
        for frameInd in range(self.getNumFrames()):
            frame = self.getFrame(frameInd)
            saveFile.write('FrameNumber: ' + str(frame.getFrameNumber())) 
            saveFile.write('\t' + 'TotalObjects: ' + str(frame.getObjectCount()))
            saveFile.write('\n')
            for j in range(frame.getObjectCount()):
                obj = frame.getObject(j)
                
                saveFile.write('\t' + 'FrameNumber: ' + str(obj.getFrameNumber()))
                saveFile.write('\t' + 'ObjectIndex: ' + str(obj.getObjectIndex()))
                saveFile.write('\t' + 'Position: [' + str(obj.getX()) + ' ' + str(obj.getY()) + ']')
                saveFile.write('\t' + 'Size: ' + str(obj.getSize()))
                saveFile.write('\n')
                # write all the pixelLocs of the object to the text file in a single line
                rows, cols = obj.getAllPixelLocs()
                saveFile.write('\t')
                for k in range(len(rows)):
                    saveFile.write(str(rows[k]) + ' ' + str(cols[k]) + ' ')
                saveFile.write('\n')
                
        saveFile.close()
        assert savePath.exists()
         
    def loadFromTextFile(self, loadDir_pathObj):
         # Load the video analysis data from the text file
        loadPath = loadDir_pathObj / 'videoAnalysis.txt'
        loadFile = open(loadPath, 'r')
        
        # Read the video data
        lines = loadFile.readlines()
        self.frames = []
        
        lineIndex = 0
        totalFrames = int(lines[lineIndex].split()[2])  # Read the total number of frames from the first line
        
        # Read the data of each frame
        for i in range(totalFrames):
            lineIndex += 1
            frame = Frame()
            frameNumber = int(lines[lineIndex].split()[1])
            objectCount = int(lines[lineIndex].split()[3])
            frame.setFrameNumber(frameNumber)
            frame.setObjectCount(objectCount)
            for j in range(objectCount):
                lineIndex += 1
                objFrameNumber = int(lines[lineIndex].split()[1])
                objIndex = int(lines[lineIndex].split()[3])
                position = np.array([int(lines[lineIndex].split()[5][1:]), int(lines[lineIndex].split()[6][:-1])])
                size = int(lines[lineIndex].split()[8])
                lineIndex += 1
                pixelLocs = lines[lineIndex].split()
                pixelLocs = np.array(pixelLocs[1:-1], dtype=np.int32)
                pixelLocs = pixelLocs.reshape(-1, 2)
                rows = pixelLocs[:, 0]
                cols = pixelLocs[:, 1]
                # Create a new object and add it to the frame
                newObject = Object(objFrameNumber, objIndex, position[0], position[1], size, rows, cols)
                frame.addObject(newObject)
                
            # Add the frame to the video
            self.frames.append(frame)            
        loadFile.close()
        assert self.checkAnalysisFileExists(loadDir_pathObj)
    
    def checkAnalysisFileExists(self, loadDir_pathObj):
        loadPath = loadDir_pathObj / 'videoAnalysis.txt'
        if loadPath.exists():
            return True
        return False
    
    def getFrameIndexFromNumber(self, frameNumber):
        startFrameNumber = self.frames[0].getFrameNumber()
        frameIndex = frameNumber - startFrameNumber
        return frameIndex
    
    def getObjectFromFrameAndObjectIndex(self, frameIndex, objectIndex):
        return self.frames[frameIndex].getObject(objectIndex)
    
    def getObjectFromBubbleLoc(self, bubbleLocation):
        frameNumber = bubbleLocation[0]
        objectIndex = bubbleLocation[1]
        frameIndex = self.getFrameIndexFromNumber(frameNumber)
        return self.getObjectFromFrameAndObjectIndex(frameIndex, objectIndex)
    
    def getLatestBubbleObject(self, bubbleListIndex):
        bubble = self.bubbles[bubbleListIndex]
        latestLocation = bubble.getLatestLocation()
        latestObj = self.getObjectFromBubbleLoc(latestLocation)
        return latestObj
    
    def getBubbleSize(self, location):
        obj = self.getObjectFromBubbleLoc(location)
        return obj.getSize()
        
    def trackObjects(self, distanceThreshold, sizeThreshold):
        """ 
        Algorithm:
        - frame0 = getFrame(0)
        - create bubble objects for all objects in frame0 and add them to the bubble list
        - traverse through all frames[1:] and for each frame:
            - create a copy of the frame
            - for all bubbles in the bubble list:
                - get the latest object of the bubble
                - find the object in the frame that is closest to the latest object (CRITERIA)
                - if found an object:
                    - append the object to the bubble
                    - remove the object from the frame copy
            - for all the remaining objects in the frame copy:
                - create a new bubble for the object and add it to the bubble list
        """
        bubbleIndex = 0
        frame0      = self.getFrame(0)
        for objInd in range(frame0.getObjectCount()):
            obj         = frame0.getObject(objInd)
            frameNum    = frame0.getFrameNumber()
            objIndex    = obj.getObjectIndex()
            bubble      = Bubble(bubbleIndex)
            bubbleIndex += 1
            bubble.appendTrajectory(frameNum, objIndex)
            self.bubbles.append(bubble)
        
        for frameInd in tqdm(range(1, self.getNumFrames()), desc='Tracking objects'):
            frame = self.getFrame(frameInd)
            frameCopy = frame.copy()
            for listInd in range(len(self.bubbles)):
                bubble  = self.bubbles[listInd]
                latestObj = self.getLatestBubbleObject(listInd)
                closestObjsInd = frameCopy.getNearbyAndComparableSizeObjectIndices_object(latestObj, distanceThreshold, sizeThreshold)
                if len(closestObjsInd) > 0:
                    closestObj = frameCopy.getObject(closestObjsInd[0])
                    bubble.appendTrajectory(frameCopy.getFrameNumber(), closestObj.getObjectIndex())
                    frameCopy.removeObject_index(closestObjsInd[0])
            for objInd in range(frameCopy.getObjectCount()):
                obj         = frameCopy.getObject(objInd)
                frameNum    = frame.getFrameNumber()
                objIndex    = obj.getObjectIndex()
                bubble      = Bubble(bubbleIndex)
                bubbleIndex += 1
                bubble.appendTrajectory(frameNum, objIndex)
                self.bubbles.append(bubble)
        
        # Sort the bubbles by their trajectory length
        # self.bubbles.sort(key=lambda bubble: bubble.getTrajectoryLength(), reverse=True)
        
        # Sort the bubbles by their size (largest to smallest)
        self.bubbles.sort(key=lambda bubble: self.getBubbleSize(bubble.getLocation(0)), reverse=True)
        
    
    def getPositionAndSizeArrayFromTrajectory(self, trajectory):
        position = np.zeros((len(trajectory), 2))
        size = np.zeros(len(trajectory))
        for i in range(len(trajectory)):
            loc = trajectory[i]
            obj = self.getObjectFromBubbleLoc(loc)
            position[i, :] = obj.getPosition()
            size[i] = obj.getSize()
        return position, size
    
    def plotTrajectory(self, bubbleListIndex, binaryFrameDir_pathObj, videoDir_pathObj, fps, frameNameTemplate):
        """ 
        Plot 
        """
        bubble = self.bubbles[bubbleListIndex]
        trajectory = bubble.getFullTrajectory()
        position, size = self.getPositionAndSizeArrayFromTrajectory(trajectory)
        
        _, videoWidth, videoHeight = self.plotTrajectory_subFunc(position[0], size[0], trajectory[0][0], frameNameTemplate, binaryFrameDir_pathObj)
        
        # vidCodec = cv2.VideoWriter_fourcc(*'XVID')
        vidCodec    = cv2.VideoWriter_fourcc(*'mp4v')  # codec for .mp4 file
        videoWriter = cv2.VideoWriter(str(videoDir_pathObj / 'videoBubbleTrajectory_Size{:05d}_fnstrt{:05d}_fnend{:05d}.mp4'.format(int(size[0]), trajectory[0][0], trajectory[-1][0])),vidCodec, fps, (videoWidth, videoHeight))
               
        for i in tqdm(range(len(trajectory)) , desc='Plotting trajectory for Size = {:04d}'.format(int(size[0]))):
            # Create plot showing the object position (x, y) with marker size = object size
            videoArray, videoWidth, videoHeight = self.plotTrajectory_subFunc(position[i], size[i], trajectory[i][0], frameNameTemplate, binaryFrameDir_pathObj)
            # Write the combined frame to the video
            videoWriter.write(videoArray)
             
        videoWriter.release()
        
    def plotTrajectory_subFunc(self, position, size, frameNumber, frameNameTemplate, binaryFrameDir_pathObj):
        # Get the binary frame as np array
        frameName   = frameNameTemplate.format(frameNumber)
        framePath   = binaryFrameDir_pathObj / frameName
        frameArray  = cv2.imread(str(framePath))
        
        # Set figsize based on the binary frame size
        sizeReductionFactor = 0.01
        subtractAmount = 4
        figsize = (frameArray.shape[0] * sizeReductionFactor - subtractAmount, frameArray.shape[0] * sizeReductionFactor) 
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.scatter(position[0], position[1], s=size)
        ax.set_xlim(0, frameArray.shape[1])
        ax.set_ylim(0, frameArray.shape[0])
        ax.set_aspect('equal')
        ax.text(0.05, 0.95, 'fN = {}'.format(frameNumber), horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, fontsize=6)
        plt.tight_layout()
        
        # Get plot as np array
        fig.canvas.draw()
        plotArray = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plotArray = plotArray.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
        # Combine the plot and the binary frame
        videoWidth  = frameArray.shape[1] + plotArray.shape[1]
        videoHeight = max(frameArray.shape[0], plotArray.shape[0])
        videoArray  = np.zeros((videoHeight, videoWidth, 3), dtype=np.uint8)
        
        if frameArray.shape[0] < videoHeight: # pad the frameArray
            frameArray = np.pad(frameArray, ((0, videoHeight - frameArray.shape[0]), (0, 0), (0, 0)), 'constant')
        if plotArray.shape[0] < videoHeight: # pad the plotArray
            plotArray = np.pad(plotArray, ((0, videoHeight - plotArray.shape[0]), (0, 0), (0, 0)), 'constant')
        
        videoArray[:, :frameArray.shape[1], :] = frameArray
        videoArray[:, frameArray.shape[1]:, :] = plotArray
        
        plt.close(fig)
        return videoArray, videoWidth, videoHeight
    
    def isSame(self, video):
        if len(self.frames) != len(video.frames):
            return False
        
        for i in range(len(self.frames)):
            if not self.frames[i].isSame(video.frames[i]):
                return False
        return True
        
    def getColor(self, n):
        """ 
        #000000
        #00FF00
        #0000FF
        #FF0000
        #01FFFE
        #FFA6FE
        #FFDB66
        #006401
        #010067
        #95003A
        #007DB5
        #FF00F6
        #FFEEE8
        #774D00
        #90FB92
        #0076FF
        #D5FF00
        #FF937E
        #6A826C
        #FF029D
        #FE8900
        #7A4782
        #7E2DD2
        #85A900
        #FF0056
        #A42400
        #00AE7E
        #683D3B
        #BDC6FF
        #263400
        #BDD393
        #00B917
        #9E008E
        #001544
        #C28C9F
        #FF74A3
        #01D0FF
        #004754
        #E56FFE
        #788231
        #0E4CA1
        #91D0CB
        #BE9970
        #968AE8
        #BB8800
        #43002C
        #DEFF74
        #00FFC6
        #FFE502
        #620E00
        #008F9C
        #98FF52
        #7544B1
        #B500FF
        #00FF78
        #FF6E41
        #005F39
        #6B6882
        #5FAD4E
        #A75740
        #A5FFD2
        #FFB167
        #009BFF
        #E85EBE 
        """
        colorsList = ['#00FF00', '#0000FF', '#FF0000', '#01FFFE', '#FFA6FE', '#FFDB66', '#006401', '#010067', '#95003A', '#007DB5', '#FF00F6', '#FFEEE8', '#774D00', '#90FB92', '#0076FF', '#D5FF00', '#FF937E', '#6A826C', '#FF029D', '#FE8900', '#7A4782', '#7E2DD2', '#85A900', '#FF0056', '#A42400', '#00AE7E', '#683D3B', '#BDC6FF', '#263400', '#BDD393', '#00B917', '#9E008E', '#001544', '#C28C9F', '#FF74A3', '#01D0FF', '#004754', '#E56FFE', '#788231', '#0E4CA1', '#91D0CB', '#BE9970', '#968AE8', '#BB8800', '#43002C', '#DEFF74', '#00FFC6', '#FFE502', '#620E00', '#008F9C', '#98FF52', '#7544B1', '#B500FF', '#00FF78', '#FF6E41', '#005F39', '#6B6882', '#5FAD4E', '#A75740', '#A5FFD2', '#FFB167', '#009BFF', '#E85EBE']
        
        if n < len(colorsList):
            return colorsList[n]
        else:
            return '#FFFFFF'
        
        