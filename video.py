import numpy as np
import logging
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import ImageColor
from pathlib import Path

from object import Object
from frame import Frame
from bubble import Bubble

class Video:
    """
    A class representing a video composed of frames and bubbles.

    Attributes
    ----------
    frames : list of Frame
        A list containing frames in order.
    bubbles : list of Bubble
        A list containing bubble objects.
    """

    def __init__(self):
        self.frames = []
        self.bubbles = []

    def addFrame(self, frame):
        """
        Add a Frame object to the video.
        """
        self.frames.append(frame)

    def getFrame(self, frameIndex):
        return self.frames[frameIndex]

    def getFrames(self, frameIndices):
        return [self.frames[i] for i in frameIndices]

    def getAllFrames(self):
        return self.frames

    def getNumFrames(self):
        return len(self.frames)

    def getNumBubbles(self):
        return len(self.bubbles)

    def saveBubblesToTextFile(self, saveDir_pathObj):
        """
        Save bubble trajectory data to a text file.
        """
        saveDir_pathObj.mkdir(parents=True, exist_ok=True)
        savePath = saveDir_pathObj / 'videoBubbles.txt'
        with open(savePath, 'w') as saveFile:
            saveFile.write('Total bubbles: ' + str(self.getNumBubbles()) + '\n')
            for bubbleInd, bubble in enumerate(self.bubbles):
                saveFile.write('BubbleIndex: ' + str(bubble.getBubbleIndex()))
                saveFile.write('\t' + 'TrajectoryLength: ' + str(bubble.getTrajectoryLength()) + '\n\t')
                for j in range(bubble.getTrajectoryLength()):
                    loc = bubble.getLocation(j)
                    saveFile.write(str(loc[0]) + ' ' + str(loc[1]) + ' ')
                saveFile.write('\n')

        logging.info("Bubbles saved to %s", savePath)

    def loadBubblesFromTextFile(self, loadDir_pathObj):
        """
        Load bubble trajectory data from a text file.
        """
        loadPath = loadDir_pathObj / 'videoBubbles.txt'
        if not loadPath.exists():
            logging.error("Bubbles file not found at %s", loadPath)
            return
        with open(loadPath, 'r') as loadFile:
            lines = loadFile.readlines()
        
        self.bubbles = []
        lineIndex = 0
        try:
            numBubbles = int(lines[lineIndex].split()[-1])
        except (IndexError, ValueError):
            logging.error("Invalid bubble file format.")
            return
        
        for _ in range(numBubbles):
            lineIndex += 1
            lineSplit = lines[lineIndex].split()
            bubbleIndex = int(lineSplit[1])
            numTrajectory = int(lineSplit[3])
            bubble = Bubble(bubbleIndex)
            lineIndex += 1
            lineSplit = lines[lineIndex].split()
            for j in range(numTrajectory):
                frameNumber = int(lineSplit[2*j])
                objectIndex = int(lineSplit[2*j+1])
                bubble.appendTrajectory(frameNumber, objectIndex)
            self.bubbles.append(bubble)

        logging.info("Loaded %d bubbles from %s", numBubbles, loadPath)

    def saveFramesToTextFile(self, saveDir_pathObj):
        """
        Save frames data (including objects) to a text file.
        """
        saveDir_pathObj.mkdir(parents=True, exist_ok=True)
        savePath = saveDir_pathObj / 'videoFrames.txt'
        with open(savePath, 'w') as saveFile:
            saveFile.write('Total frames: ' + str(self.getNumFrames()) + '\n')
            for frameInd, frame in enumerate(self.frames):
                saveFile.write('FrameNumber: ' + str(frame.getFrameNumber()))
                saveFile.write('\t' + 'TotalObjects: ' + str(frame.getObjectCount()) + '\n')
                for objInd in range(frame.getObjectCount()):
                    obj = frame.getObject(objInd)
                    saveFile.write('\tFrameNumber: ' + str(obj.getFrameNumber()))
                    saveFile.write('\tObjectIndex: ' + str(obj.getObjectIndex()))
                    saveFile.write('\tPosition: [' + str(obj.getX()) + ' ' + str(obj.getY()) + ']')
                    saveFile.write('\tSize: ' + str(obj.getSize()) + '\n')

                    # Pixel locations
                    rows, cols = obj.getAllPixelLocs()
                    saveFile.write('\t')
                    for k in range(len(rows)):
                        saveFile.write(str(rows[k]) + ' ' + str(cols[k]) + ' ')
                    saveFile.write('\n')

        logging.info("Frames saved to %s", savePath)

    def loadFramesFromTextFile(self, loadDir_pathObj):
        """
        Load frames data (including objects) from a text file.
        """
        loadPath = loadDir_pathObj / 'videoFrames.txt'
        if not loadPath.exists():
            logging.error("Frames file not found at %s", loadPath)
            return
        with open(loadPath, 'r') as loadFile:
            lines = loadFile.readlines()

        self.frames = []
        lineIndex = 0
        try:
            totalFrames = int(lines[lineIndex].split()[2])
        except (IndexError, ValueError):
            logging.error("Invalid frame file format.")
            return
        
        for _ in range(totalFrames):
            lineIndex += 1
            frame = Frame()
            frameNumber = int(lines[lineIndex].split()[1])
            objectCount = int(lines[lineIndex].split()[3])
            frame.setFrameNumber(frameNumber)
            frame.setObjectCount(objectCount)
            for __ in range(objectCount):
                lineIndex += 1
                lineSplit = lines[lineIndex].split()
                objFrameNumber = int(lineSplit[1])
                objIndex = int(lineSplit[3])
                posX = int(lineSplit[5][1:])
                posY = int(lineSplit[6][:-1])
                size = int(lineSplit[8])

                lineIndex += 1
                pixelLocs = lines[lineIndex].split()
                rows = []
                cols = []
                for k in range(size):
                    rows.append(int(pixelLocs[2*k]))
                    cols.append(int(pixelLocs[2*k+1]))

                newObject = Object(objFrameNumber, objIndex, posX, posY, size, rows, cols)
                frame.addObject(newObject)
            
            self.frames.append(frame)
        
        if not self.checkVideoFramesFileExists(loadDir_pathObj):
            logging.warning("videoFrames.txt found but integrity check failed.")
        else:
            logging.info("Loaded %d frames from %s", totalFrames, loadPath)

    def checkVideoFramesFileExists(self, loadDir_pathObj):
        return (loadDir_pathObj / 'videoFrames.txt').exists()
    
    def checkVideoBubblesFileExists(self, loadDir_pathObj):
        return (loadDir_pathObj / 'videoBubbles.txt').exists()

    def getFrameIndexFromNumber(self, frameNumber):
        """
        Convert a frame number to its index based on the first frame number.
        """
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
        return self.getObjectFromBubbleLoc(latestLocation)

    def getBubbleSize(self, location):
        obj = self.getObjectFromBubbleLoc(location)
        return obj.getSize()

    def trackObjects(self, params):
        """
        Track objects (bubbles) across frames based on criteria like distance and size similarity.

        Parameters
        ----------
        params : dict
            Dictionary containing parameters like:
              'distanceThreshold', 'sizeThresholdPercent', 'frameConsecThreshold', 'bubbleTrajectoryLengthThreshold'.
        """
        distanceThreshold = params['distanceThreshold']
        sizeThresholdPercent = params['sizeThresholdPercent']
        bubbleIndex = 0
        frame0 = self.getFrame(0)

        # Initialize bubbles from frame 0
        for objInd in range(frame0.getObjectCount()):
            obj = frame0.getObject(objInd)
            frameNum = frame0.getFrameNumber()
            bubble = Bubble(bubbleIndex)
            bubbleIndex += 1
            bubble.appendTrajectory(frameNum, obj.getObjectIndex())
            self.bubbles.append(bubble)

        # Track bubbles in subsequent frames
        for frameInd in tqdm(range(1, self.getNumFrames()), desc='Tracking objects'):
            frame = self.getFrame(frameInd)
            iter_frameNum = frame.getFrameNumber()
            frameCopy = frame.copy()
            for listInd in range(len(self.bubbles)):
                bubble = self.bubbles[listInd]
                latestObj = self.getLatestBubbleObject(listInd)
                objFrameNum = latestObj.getFrameNumber()

                # Check frame consecutiveness
                if abs(objFrameNum - iter_frameNum) > params['frameConsecThreshold']:
                    continue

                closestObjsInd = frameCopy.getNearbyAndComparableSizeObjectIndices_object(latestObj, distanceThreshold, sizeThresholdPercent)
                if closestObjsInd:
                    closestObj = frameCopy.getObject(closestObjsInd[0])
                    bubble.appendTrajectory(frameCopy.getFrameNumber(), closestObj.getObjectIndex())
                    frameCopy.removeObject_index(closestObjsInd[0])

            # Create new bubbles for remaining objects in frameCopy
            for objInd in range(frameCopy.getObjectCount()):
                obj = frameCopy.getObject(objInd)
                frameNum = frame.getFrameNumber()
                newBubble = Bubble(bubbleIndex)
                bubbleIndex += 1
                newBubble.appendTrajectory(frameNum, obj.getObjectIndex())
                self.bubbles.append(newBubble)

        # Sort bubbles by size and remove short trajectories
        self.bubbles.sort(key=lambda b: self.getBubbleSize(b.getLatestLocation()), reverse=True)
        self.bubbles = [b for b in self.bubbles if b.getTrajectoryLength() >= params['bubbleTrajectoryLengthThreshold']]

        logging.info("Tracking completed. Number of bubbles: %d", len(self.bubbles))

    def getPositionAndSizeArrayFromTrajectory(self, trajectory):
        """
        Given a trajectory, return arrays for position (N x 2) and size (N).
        """
        position = np.zeros((len(trajectory), 2), dtype=int)
        size = np.zeros(len(trajectory), dtype=int)
        for i, loc in enumerate(trajectory):
            obj = self.getObjectFromBubbleLoc(loc)
            position[i, :] = obj.getPosition()
            size[i] = obj.getSize()
        return position, size

    def app2_plotTrajectory(self, bubbleListIndices, binaryFrameDir_pathObj, videoFramesDir_pathObj, fps, frameNameTemplate):
        """
        Mark bubbles on frames and save the resulting frames in videoFramesDir_pathObj.
        If frames are missing, fill them from the binaryFrameDir_pathObj.
        """
        from utils import getFrameNumbers_ordered, DoNumExistingFramesMatch
        
        if bubbleListIndices is False:
            bubbleListIndices = list(range(len(self.bubbles)))

        colorIndex = 0
        for bubbleListIndex in tqdm(bubbleListIndices, desc='Creating frames'):
            bubble = self.bubbles[bubbleListIndex]
            trajectory = bubble.getFullTrajectory()
            color = self.getColor(colorIndex)
            colorIndex += 1
            for loc in trajectory:
                obj = self.getObjectFromBubbleLoc(loc)
                rows, cols = obj.getAllPixelLocs()
                frameNum = loc[0]
                frameName = frameNameTemplate.format(frameNum)
                framePath = videoFramesDir_pathObj / frameName
                if framePath.exists():
                    frame = cv2.imread(str(framePath))
                else:
                    frame = cv2.imread(str(binaryFrameDir_pathObj / frameName))

                if frame is None:
                    logging.warning("Frame %s not found or not readable.", frameName)
                    continue
                frame[rows, cols, :] = color
                cv2.putText(frame, str(frameNum), (frame.shape[1] - 30, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.imwrite(str(framePath), frame)

        if DoNumExistingFramesMatch(videoFramesDir_pathObj, self.getNumFrames()):
            return

        # Fill missing frames
        incompleteFrameNums = getFrameNumbers_ordered(videoFramesDir_pathObj, frameNameTemplate, False)
        missingFrameNums = list(set(range(self.getNumFrames())) - set(incompleteFrameNums))
        for missingFrameNum in tqdm(missingFrameNums, desc='Adding missing frames'):
            frameName = frameNameTemplate.format(missingFrameNum)
            framePath = binaryFrameDir_pathObj / frameName
            frame = cv2.imread(str(framePath))
            if frame is None:
                logging.warning("Missing frame %s not found in binary directory.", frameName)
                continue
            cv2.imwrite(str(videoFramesDir_pathObj / frameName), frame)

    def plotTrajectory(self, bubbleListIndex, binaryFrameDir_pathObj, videoDir_pathObj, fps, frameNameTemplate):
        """
        Plot bubble trajectory for a single bubble as a video.
        """
        if bubbleListIndex >= len(self.bubbles):
            logging.warning("Bubble index %d out of range.", bubbleListIndex)
            return False

        bubble = self.bubbles[bubbleListIndex]
        trajectory = bubble.getFullTrajectory()
        position, size = self.getPositionAndSizeArrayFromTrajectory(trajectory)
        color = self.getColor(bubbleListIndex)

        # Use the first frame to get dimensions
        videoArray, videoWidth, videoHeight = self.plotTrajectory_subFunc(trajectory[0], frameNameTemplate, binaryFrameDir_pathObj, color)

        if videoArray is None:
            logging.error("Could not read initial frame for plotting trajectory.")
            return False
        
        vidCodec = cv2.VideoWriter_fourcc(*'mp4v')
        outName = f'Bub_Sstrt{int(size[0]):05d}_Send{int(size[-1]):05d}_fnstrt{trajectory[0][0]:05d}_fnend{trajectory[-1][0]:05d}.mp4'
        videoWriter = cv2.VideoWriter(str(videoDir_pathObj / outName), vidCodec, fps, (videoWidth, videoHeight))

        for i in tqdm(range(len(trajectory)), desc=f'Plotting trajectory for Size = {int(size[-1]):04d}'):
            videoArray, videoWidth, videoHeight = self.plotTrajectory_subFunc(trajectory[i], frameNameTemplate, binaryFrameDir_pathObj, color)
            if videoArray is None:
                logging.warning("Frame for trajectory step %d not found.", i)
                continue
            videoWriter.write(videoArray)

        videoWriter.release()
        logging.info("Trajectory video saved to %s", videoDir_pathObj / outName)
        return True

    def plotTrajectory_subFunc(self, trajectory, frameNameTemplate, binaryFrameDir_pathObj, color):
        frameNumber = trajectory[0]
        frameName = frameNameTemplate.format(frameNumber)
        framePath = binaryFrameDir_pathObj / frameName
        frameArray = cv2.imread(str(framePath))
        if frameArray is None:
            logging.error("Failed to read frame %s.", frameName)
            return None, 0, 0

        obj = self.getObjectFromBubbleLoc(trajectory)
        rows, cols = obj.getAllPixelLocs()
        frameArray[rows, cols, :] = color

        return frameArray, frameArray.shape[1], frameArray.shape[0]

    def isVideoContinuous(self):
        """
        Check if frames in the video are continuous in numbering.
        """
        for i in range(len(self.frames)-1):
            if self.frames[i+1].getFrameNumber() - self.frames[i].getFrameNumber() != 1:
                return False
        return True

    def isSame(self, video):
        """
        Check if another video object contains the same frames and bubbles.
        """
        if len(self.frames) != len(video.frames):
            return False
        if len(self.bubbles) != len(video.bubbles):
            return False
        
        for i in range(len(self.frames)):
            if not self.frames[i].isSame(video.frames[i]):
                return False

        for i in range(len(self.bubbles)):
            if not self.bubbles[i].isSame(video.bubbles[i]):
                return False
        return True

    def getColor(self, n):
        """
        Get a unique color from a predefined list, cycling through if out of range.
        """
        colorsList = [
            '#00FF00', '#0000FF', '#FF0000', '#01FFFE', '#FFA6FE', '#FFDB66', '#006401', '#010067', '#95003A',
            '#007DB5', '#FF00F6', '#FFEEE8', '#774D00', '#90FB92', '#0076FF', '#D5FF00', '#FF937E', '#6A826C',
            '#FF029D', '#FE8900', '#7A4782', '#7E2DD2', '#85A900', '#FF0056', '#A42400', '#00AE7E', '#683D3B',
            '#BDC6FF', '#263400', '#BDD393', '#00B917', '#9E008E', '#001544', '#C28C9F', '#FF74A3', '#01D0FF',
            '#004754', '#E56FFE', '#788231', '#0E4CA1', '#91D0CB', '#BE9970', '#968AE8', '#BB8800', '#43002C',
            '#DEFF74', '#00FFC6', '#FFE502', '#620E00', '#008F9C', '#98FF52', '#7544B1', '#B500FF', '#00FF78',
            '#FF6E41', '#005F39', '#6B6882', '#5FAD4E', '#A75740', '#A5FFD2', '#FFB167', '#009BFF', '#E85EBE'
        ]
        
        val = n % len(colorsList)
        colorHex = colorsList[val]
        colorRGB = ImageColor.getcolor(colorHex, "RGB")
        return colorRGB
