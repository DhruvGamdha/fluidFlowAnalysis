import numpy as np
import logging
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import ImageColor
from pathlib import Path
import json

from object import Object
from frame import Frame
from bubble import Bubble
from utils import getFrameNumbers_ordered, imgSegmentation, calculatePixelCentroid, DoNumExistingFramesMatch

from typing import List, Tuple

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
        self.frames: List[Frame] = []
        self.bubbles: List[Bubble] = []
        self.frameHeight = 0
        self.frameWidth = 0

    def addFrame(self, frame: Frame):
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
    
    def getFrameHeight(self):
        return self.frameHeight
    
    def setFrameHeight(self, height):
        self.frameHeight = height
    
    def getFrameWidth(self):
        return self.frameWidth
    
    def setFrameWidth(self, width):
        self.frameWidth = width
    
    @classmethod
    def dropAnalysis(cls, binaryFrameDir_pathObj: Path, analysisBaseDir_pathObj: Path, frameNameTemplate: str, params: dict) -> 'Video':
        """
        Perform analysis on the binary frames to identify objects (bubbles) and their properties.

        Parameters
        ----------
        binaryFrameDir_pathObj : pathlib.Path
            Directory containing binary frames.
        analysisBaseDir_pathObj : pathlib.Path
            Base directory for saving analysis results.
        frameNameTemplate : str
            Template for naming frames.
        params : dict
            Configuration parameters, including 'connectivity'.

        Returns
        -------
        Video
            A Video object populated with Frame and Object data.
        """
        # video = Video()
        video = cls()
        allFramesNum = getFrameNumbers_ordered(binaryFrameDir_pathObj, frameNameTemplate)
        connectivity = params["connectivity"]
        
        # Write the frame dimensions to the video object
        frame0 = cv2.imread(str(binaryFrameDir_pathObj / frameNameTemplate.format(allFramesNum[0])))
        if frame0 is not None:
            video.setFrameHeight(frame0.shape[0])
            video.setFrameWidth(frame0.shape[1])

        for frameNum in tqdm(allFramesNum, desc="Analyzing drops"):
            labelImg, count, imgShape = imgSegmentation(binaryFrameDir_pathObj, frameNameTemplate, frameNum, connectivity)
            
            frame = Frame()
            frame.setFrameNumber(frameNum)
            frame.setObjectCount(count)

            for objLabel in range(1, count+1):
                rows, cols = np.where(labelImg == objLabel)
                # Origin at bottom left corner
                topLft_y = imgShape[0] - np.min(rows)
                topLft_x = np.min(cols)
                objInd = objLabel - 1

                obj = Object(frameNum, objInd, topLft_x, topLft_y, len(rows), rows, cols)
                frame.addObject(obj)
            
            video.addFrame(frame)
        
        if not video.isVideoContinuous():
            logging.error("Video frames are not continuous. Analysis aborted.")
            raise SystemExit("Video is not continuous. Exiting...")
            
        return video
    
    def saveBubblesToTextFile(self, saveDir_pathObj: Path):
        """
        Save bubble data to a JSON file.
        """
        saveDir_pathObj.mkdir(parents=True, exist_ok=True)
        savePath = saveDir_pathObj / 'videoBubbles.json'
        
        data = {
            "numBubbles": self.getNumBubbles(),
            "bubbles": []
        }

        for bubble in self.bubbles:
            bubbleData = {
                "bubbleIndex": bubble.getBubbleIndex(),
                "trajectory": bubble.getFullTrajectory(),
                "positions": bubble.getPositions_fullTrajectory().tolist() if bubble.getPositions_fullTrajectory() is not None and bubble.getPositions_fullTrajectory().size > 0 else [],
                "velocities": bubble.getVelocities_fullTrajectory().tolist() if bubble.getVelocities_fullTrajectory() is not None and bubble.getVelocities_fullTrajectory().size > 0 else [],
                "accelerations": bubble.getAccelerations_fullTrajectory().tolist() if bubble.getAccelerations_fullTrajectory() is not None and bubble.getAccelerations_fullTrajectory().size > 0 else []
            }
            data["bubbles"].append(bubbleData)
            
        # Convert all data to native Python types
        data = convert_to_builtin_types(data)

        with open(savePath, 'w') as f:
            json.dump(data, f, indent=4)

        logging.info("Bubbles saved to %s in JSON format.", savePath)
    
    def loadBubblesFromTextFile(self, loadDir_pathObj: Path):
        """
        Load bubble data (including trajectory, velocities, and accelerations) from a JSON file.
        """
        loadPath = loadDir_pathObj / 'videoBubbles.json'
        if not loadPath.exists():
            logging.error("Bubbles file not found at %s", loadPath)
            return

        with open(loadPath, 'r') as f:
            data = json.load(f)

        # Convert data to native Python types
        data = convert_to_builtin_types(data)

        numBubbles = data.get("numBubbles", 0)
        bubblesList = data.get("bubbles", [])

        self.bubbles = []
        for bubbleInfo in bubblesList:
            bubbleIndex = bubbleInfo["bubbleIndex"]
            trajectory = bubbleInfo["trajectory"]
            positions = bubbleInfo.get("positions", [])
            velocities = bubbleInfo.get("velocities", [])
            accelerations = bubbleInfo.get("accelerations", [])

            bubble = Bubble(bubbleIndex)

            # Add trajectory data
            for frameNumber, objectIndex in trajectory:
                bubble.appendTrajectory(frameNumber, objectIndex)

            # Add kinematics data
            bubble.setPositions_fullTrajectory(np.array(positions) if positions else np.array([]))
            bubble.setVelocities_fullTrajectory(np.array(velocities) if velocities else np.array([]))
            bubble.setAccelerations_fullTrajectory(np.array(accelerations) if accelerations else np.array([]))

            self.bubbles.append(bubble)

        logging.info("Loaded %d bubbles from %s (JSON format).", numBubbles, loadPath)
    

    def saveFramesToTextFile(self, saveDir_pathObj: Path):
        """
        Save frames data (including objects) to a JSON file.
        """
        saveDir_pathObj.mkdir(parents=True, exist_ok=True)
        savePath = saveDir_pathObj / 'videoFrames.json'
        
        data = {
            "numFrames": self.getNumFrames(),
            "frameHeight": self.getFrameHeight(),
            "frameWidth": self.getFrameWidth(),
            "frames": []
        }

        # Construct the data structure
        for frame in self.frames:
            frameData = {
                "frameNumber": frame.getFrameNumber(),
                "numObjects": frame.getObjectCount(),
                "objects": []
            }
            for objInd in range(frame.getObjectCount()):
                obj = frame.getObject(objInd)
                rows, cols = obj.getAllPixelLocs()
                objData = {
                    "frameNumber": obj.getFrameNumber(),
                    "objectIndex": obj.getObjectIndex(),
                    # Position: previously saved as "[x y]". Now we store as a list [x, y].
                    "position": [obj.getX(), obj.getY()],
                    "size": obj.getSize(),
                    "rows": rows.tolist() if isinstance(rows, np.ndarray) else list(rows),
                    "cols": cols.tolist() if isinstance(cols, np.ndarray) else list(cols)
                }
                frameData["objects"].append(objData)
            
            data["frames"].append(frameData)

        # Convert to Python built-in types before serialization
        data = convert_to_builtin_types(data)

        with open(savePath, 'w') as f:
            json.dump(data, f, indent=4)

        logging.info("Frames saved to %s in JSON format.", savePath)

    def loadFramesFromTextFile(self, loadDir_pathObj: Path):
        """
        Load frames data (including objects) from a JSON file.
        """
        loadPath = loadDir_pathObj / 'videoFrames.json'
        if not loadPath.exists():
            logging.error("Frames file not found at %s", loadPath)
            return

        with open(loadPath, 'r') as f:
            data = json.load(f)

        # Convert to built-in types if needed (though json.load should already return them)
        data = convert_to_builtin_types(data)

        numFrames = data.get("numFrames", 0)
        self.setFrameHeight(data.get("frameHeight", 0))
        self.setFrameWidth(data.get("frameWidth", 0))
        
        framesList = data.get("frames", [])

        self.frames = []
        for frameInfo in framesList:
            frameNumber = frameInfo["frameNumber"]
            objectCount = frameInfo["numObjects"]
            frame = Frame()
            frame.setFrameNumber(frameNumber)
            frame.setObjectCount(objectCount)

            for objInfo in frameInfo["objects"]:
                objFrameNumber = objInfo["frameNumber"]
                objIndex = objInfo["objectIndex"]
                posX, posY = objInfo["position"]  # previously from Position: [x y]
                size = objInfo["size"]
                rows = objInfo["rows"]
                cols = objInfo["cols"]

                # Create the object
                # Note: The Object constructor takes (frameNumber, objectIndex, topLft_x, topLft_y, size, rows, cols).
                # Here we pass posX, posY as top-left coordinates (as the old code did).
                # The object will recalculate the center of mass internally from the pixel locations.
                newObject = Object(objFrameNumber, objIndex, posX, posY, size, rows, cols)
                frame.addObject(newObject)
            
            self.frames.append(frame)

        # Optional integrity check
        if numFrames != self.getNumFrames():
            logging.warning("Number of frames in JSON does not match parsed frames.")
        else:
            logging.info("Loaded %d frames from %s (JSON format).", numFrames, loadPath)

    def checkVideoFramesFileExists(self, loadDir_pathObj: Path):
        return (loadDir_pathObj / 'videoFrames.txt').exists()
    
    def checkVideoBubblesFileExists(self, loadDir_pathObj: Path):
        return (loadDir_pathObj / 'videoBubbles.txt').exists()

    def getFrameIndexFromNumber(self, frameNumber: int):
        """
        Convert a frame number to its index based on the first frame number.
        """
        startFrameNumber = self.frames[0].getFrameNumber()
        frameIndex = frameNumber - startFrameNumber
        return frameIndex

    def getObjectFromFrameAndObjectIndex(self, frameIndex: int, objectIndex: int):
        return self.frames[frameIndex].getObject(objectIndex)

    def getObjectFromBubbleLoc(self, bubbleLocation: list):
        frameNumber = bubbleLocation[0]
        objectIndex = bubbleLocation[1]
        frameIndex = self.getFrameIndexFromNumber(frameNumber)
        return self.getObjectFromFrameAndObjectIndex(frameIndex, objectIndex)

    def getLatestBubbleObject(self, bubbleListIndex):
        bubble: Bubble = self.bubbles[bubbleListIndex]
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
        
        After tracking objects, also compute velocity and acceleration for each bubble.
        """
        distanceThreshold = params['distanceThreshold']
        sizeThresholdPercent = params['sizeThresholdPercent']
        bubbleIndex = 0
        frame0 = self.getFrame(0)
        fps = params['frameTimeStep']  # Ensure this is defined in params

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
                latestObj: Object = self.getLatestBubbleObject(listInd)
                objFrameNum = latestObj.getFrameNumber()

                # Check frame consecutiveness
                if abs(objFrameNum - iter_frameNum) > params['frameConsecThreshold']:
                    continue

                closestObjsInd = frameCopy.getNearbyAndComparableSizeObjectIndices_object(latestObj, distanceThreshold, sizeThresholdPercent)
                if closestObjsInd:
                    closestObj: Object = frameCopy.getObject(closestObjsInd[0])
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

        # Sort bubbles by size and filter them based on trajectory length threshold
        self.bubbles.sort(key=lambda b: self.getBubbleSize(b.getLatestLocation()), reverse=True)
        self.bubbles = [b for b in self.bubbles if b.getTrajectoryLength() >= params['bubbleTrajectoryLengthThreshold']]

        # Compute velocities and accelerations for each bubble
        for bubble in self.bubbles:
            self.computeVelocitiesAndAccelerations(bubble, fps)

        logging.info("Tracking completed. Number of bubbles: %d", len(self.bubbles))

    def app2_plotTrajectory(self, 
                            bubbleListIndices, 
                            binaryFrameDir_pathObj, 
                            videoFramesDir_pathObj, 
                            fps, 
                            frameNameTemplate,
                            params: dict):
        """
        Mark bubbles on frames and save the resulting frames in videoFramesDir_pathObj.
        If frames are missing, fill them from the binaryFrameDir_pathObj.
        """
        # from utils import getFrameNumbers_ordered, DoNumExistingFramesMatch

        colorIndex = 0
        for bubbleListIndex in tqdm(bubbleListIndices, desc='Processing Bubbles'):
            bubble: Bubble = self.bubbles[bubbleListIndex]
            trajectory = bubble.getFullTrajectory()
            color = self.getColor(colorIndex)
            colorIndex += 1
            
            # Iterate through each point in the bubble's trajectory
            for loc in trajectory:
                frame_num, object_index = loc
                
                # Retrieve the corresponding object
                obj:Object = self.getObjectFromBubbleLoc(loc)
                if obj is None:
                    logging.warning("Object not found for bubble %d at frame %d and object index %d", bubbleListIndex, frame_num, object_index)
                    continue
                
                rows, cols = obj.getAllPixelLocs()
                
                # Construct frame filename and path
                frameName = frameNameTemplate.format(frame_num)
                framePath = videoFramesDir_pathObj / frameName
                
                # Load the existing frame or create a new one if missing
                if framePath.exists():
                    frame = cv2.imread(str(framePath))
                    if frame is None:
                        logging.warning(f"Failed to read existing frame {framePath}. Creating a blank frame.")
                        exit(1)
                        
                else:
                    frame = np.zeros((self.getFrameHeight(), self.getFrameWidth(), 3), dtype=np.uint8)
                    frame.fill(255)
                
                frame[rows, cols, :] = color
                
                cv2.putText(frame, 
                            str(frame_num), 
                            (frame.shape[1] - 30, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.25, 
                            (0, 0, 0), 
                            1, 
                            cv2.LINE_AA)
                
                # Draw the bubble position on the frame
                bubblePos = None
                if params.get('drawBubblePosition', False) or params.get('drawBubbleVelocity', False) or params.get('drawBubbleAcceleration', False):
                    bubblePos = bubble.getPosition_atFrameNumber(frame_num)
                    if bubblePos is not None:
                        bubblePos = (int(bubblePos[0]), int(bubblePos[1]))
                
                if params.get('drawBubblePosition', False) and bubblePos is not None:
                    cv2.circle( frame, 
                                bubblePos, 
                                radius = 2, 
                                color = (0, 0, 255), 
                                thickness = -1)
                
                # Draw bubble velocity vector
                if params.get('drawBubbleVelocity', False) and bubblePos is not None:
                    bubbleVel = bubble.getVelocity_atFrameNumber(frame_num)
                    if bubbleVel is not None:
                        end_point = (int(bubblePos[0] + bubbleVel[0]), 
                                    int(bubblePos[1] + bubbleVel[1]))
                        cv2.arrowedLine(frame, 
                                        bubblePos,
                                        end_point,
                                        color = (0, 255, 0), 
                                        thickness = 1,
                                        tipLength = 0.2) 
                
                # Draw bubble acceleration vector
                if params.get('drawBubbleAcceleration', False) and bubblePos is not None:
                    bubbleAcc = bubble.getAcceleration_atFrameNumber(frame_num)
                    if bubbleAcc is not None:
                        end_point = (int(bubblePos[0] + bubbleAcc[0]),
                                    int(bubblePos[1] + bubbleAcc[1]))
                        cv2.arrowedLine(frame, 
                                        bubblePos,
                                        end_point,
                                        color=(255, 0, 0),
                                        thickness = 1,
                                        tipLength = 0.2)
                
                success = cv2.imwrite(str(framePath), frame)
                if not success:
                    logging.warning("Failed to write frame %s.", frameName)

        if DoNumExistingFramesMatch(videoFramesDir_pathObj, self.getNumFrames()):
            logging.info("All frames are present. No missing frames to add.")
            logging.info("Trajectory plotting completed without missing frames.")
            return

        logging.warning("Some frames are missing. Adding missing frames...")
        # Fill missing frames
        existing_frame_nums = getFrameNumbers_ordered(videoFramesDir_pathObj, frameNameTemplate, False)
        total_frames = self.getNumFrames()
        missing_frame_nums = sorted(list(set(range(total_frames)) - set(existing_frame_nums)))
        
        # Add missing frames by copying from binaryFrameDir or creating blank frame
        for missing_frame_num in tqdm(missing_frame_nums, desc='Adding missing frames'):
            frameName = frameNameTemplate.format(missing_frame_num)
            binaryFramePath = binaryFrameDir_pathObj / frameName
            frame = cv2.imread(str(binaryFramePath))
            if frame is not None:
                # Save the existing binary frame to videoFramesDir
                success = cv2.imwrite(str(videoFramesDir_pathObj / frameName), frame)
                if not success:
                    logging.warning(f"Failed to write missing frame {frameName} from binary directory.")
            else:
                # Create and save a blank white frame if binary frame is missing
                logging.warning(f"Missing frame {frameName} not found in binary directory. Creating a blank frame.")
                blank_frame = np.full((self.getFrameHeight(), self.getFrameWidth(), 3), 255, dtype=np.uint8)
                success = cv2.imwrite(str(videoFramesDir_pathObj / frameName), blank_frame)
                if not success:
                    logging.warning(f"Failed to write blank frame {frameName}.")
                    
        logging.info("Trajectory plotting completed with MISSING FRAMES ADDED.")

    def plotTrajectory(self, bubbleListIndex, binaryFrameDir_pathObj, videoDir_pathObj, fps, frameNameTemplate):
        """
        Plot bubble trajectory for a single bubble as a video.
        """
        if bubbleListIndex >= len(self.bubbles):
            logging.warning("Bubble index %d out of range.", bubbleListIndex)
            return False

        bubble: Bubble = self.bubbles[bubbleListIndex]
        trajectory = bubble.getFullTrajectory()
        position, size = self.getPositionAndSizeArrayFromTrajectory( bubble)
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
    
    def plotBubbleKinematics(self, bubbleIndex, params, outDir_pathObj):
        """
        Plot the position, velocity, and acceleration of a given bubble and save the plot.
        
        Parameters
        ----------
        bubbleIndex : int
            The index of the bubble in self.bubbles to plot.
        outDir_pathObj : pathlib.Path
            Directory to save the plot.
        """
        if bubbleIndex < 0 or bubbleIndex >= self.getNumBubbles():
            logging.error("Invalid bubble index %d", bubbleIndex)
            return
        dt = params['frameTimeStep']
        bubble: Bubble = self.bubbles[bubbleIndex]
        trajectory = bubble.getFullTrajectory()

        if len(trajectory) == 0:
            logging.warning("Bubble %d has no trajectory to plot.", bubbleIndex)
            return

        # Extract frame numbers and positions
        position = bubble.getPositions_fullTrajectory()
        frameTime = np.linspace(0, (len(position) - 1)*dt, len(position))

        # position is Nx2 (x,y)
        x = position[:, 0]
        y = position[:, 1]

        # Velocities: (N-1)x2 if present
        vx, vy = None, None
        velocity = bubble.getVelocities_fullTrajectory()
        if velocity is not None and velocity.size > 0:
            vx = velocity[:, 0]
            vy = velocity[:, 1]
            velTime = np.linspace(0, (len(velocity) - 1)*dt, len(velocity))
        else:
            velFrames = []

        # Accelerations: (N-2)x2 if present
        ax, ay = None, None
        acceleration = bubble.getAccelerations_fullTrajectory()
        if acceleration is not None and acceleration.size > 0:
            ax = acceleration[:, 0]
            ay = acceleration[:, 1]
            accTime = np.linspace(0, (len(acceleration) - 1)*dt, len(acceleration))
        else:
            accFrames = []

        # Create figure and axes
        fig, axs = plt.subplots(3, 1, figsize=(10, 12))
        fig.suptitle(f"Bubble {bubble.getBubbleIndex()} Kinematics", fontsize=16)

        # Plot Position
        axs[0].plot(frameTime, x, label='X position')
        axs[0].plot(frameTime, y, label='Y position')
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('Position (pixels)')
        axs[0].legend()
        axs[0].grid(True)

        # Plot Velocity if available
        if vx is not None and vy is not None:
            axs[1].plot(velTime, vx, label='Vx')
            axs[1].plot(velTime, vy, label='Vy')
            axs[1].set_xlabel('Time (s)')
            axs[1].set_ylabel('Velocity (pixels/s)')
            axs[1].legend()
            axs[1].grid(True)
        else:
            axs[1].text(0.5, 0.5, 'No velocities available', horizontalalignment='center', verticalalignment='center', transform=axs[1].transAxes)
            axs[1].set_axis_off()

        # Plot Acceleration if available
        if ax is not None and ay is not None:
            axs[2].plot(accTime, ax, label='Ax')
            axs[2].plot(accTime, ay, label='Ay')
            axs[2].set_xlabel('Time (s)')
            axs[2].set_ylabel('Acceleration (pixels/s²)')
            axs[2].legend()
            axs[2].grid(True)
        else:
            axs[2].text(0.5, 0.5, 'No accelerations available', horizontalalignment='center', verticalalignment='center', transform=axs[2].transAxes)
            axs[2].set_axis_off()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        outDir_pathObj.mkdir(parents=True, exist_ok=True)
        plotPath = outDir_pathObj / f"bubble_{bubble.getBubbleIndex()}_kinematics.png"
        plt.savefig(str(plotPath), dpi=300)
        plt.close(fig)

    def isVideoContinuous(self):
        """
        Check if frames in the video are continuous in numbering.
        """
        for i in range(len(self.frames)-1):
            if self.frames[i+1].getFrameNumber() - self.frames[i].getFrameNumber() != 1:
                return False
        return True

    def isSame(self, video: 'Video'):
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
    
    def computeVelocitiesAndAccelerations(self, bubble: 'Bubble', fps) -> None:
        """
        Compute velocities and accelerations from the stored trajectory.

        Parameters
        ----------
        bubble : Bubble
            The Bubble object for which to compute velocities and accelerations.
        """
        trajectory = bubble.getFullTrajectory()
        if len(trajectory) < 2:
            # Not enough data points to compute velocity
            bubble.setVelocities_fullTrajectory(np.array([]))
            bubble.setAccelerations_fullTrajectory(np.array([]))
            logging.warning(f"Not enough data points to compute velocity for Bubble {bubble.bubbleIndex}.")
            return
        
        # Get positions (x, y) for each point in the trajectory
        position, _ = self.getPositionAndSizeArrayFromTrajectory(bubble)  # position is Nx2 array
        
        # Compute velocities
        # v[i] = (pos[i+1] - pos[i]) * fps
        vel = (position[1:] - position[:-1]) * fps  # (N-1)x2 array
        bubble.setVelocities_fullTrajectory(vel)

        if len(trajectory) < 3:
            # Not enough points to compute acceleration
            bubble.setAccelerations_fullTrajectory(np.array([]))
            logging.warning(f"Not enough data points to compute acceleration for Bubble {bubble.bubbleIndex}.")
            return

        # Compute accelerations
        # a[i] = (v[i+1] - v[i]) * fps
        acc = (vel[1:] - vel[:-1]) * fps  # (N-2)x2 array
        bubble.setAccelerations_fullTrajectory(acc)

        logging.debug(f"Computed velocities and accelerations for Bubble {bubble.bubbleIndex}.")
        
    def getPositionAndSizeArrayFromTrajectory(self, bubble: 'Bubble') -> Tuple[np.ndarray, np.ndarray]:
        """
        Given a trajectory, return arrays for position (N x 2) and size (N).

        Parameters
        ----------
        bubble : Bubble
            The Bubble object whose trajectory is to be processed.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple containing:
            - position: Nx2 array of (x, y) positions.
            - size: N array of sizes.
        """
        trajectory = bubble.getFullTrajectory()
        position = np.zeros((len(trajectory), 2), dtype=int)
        size = np.zeros(len(trajectory), dtype=int)
        for i, loc in enumerate(trajectory):
            try:
                obj = self.getObjectFromBubbleLoc(loc)
                position[i, :] = obj.getPosition()
                size[i] = obj.getSize()
            except ValueError as e:
                logging.error(f"Error retrieving object for trajectory location {loc}: {e}")
                position[i, :] = [np.nan, np.nan]  # Assign NaN for missing positions
                size[i] = 0  # Assign 0 for missing sizes
        return position, size

    
    def computeBubbleKinematics(self, params: dict):
        """
        Evaluate bubble features like position, velocity, acceleration.
        
        Steps:
        1. Loop through all bubbles.
        2. For each bubble, get the trajectory (frameNumber, objectIndex).
        3. For each trajectory point, get the object and its pixel locations.
        4. Compute features like position (default: center of mass, but can be changed) and fill the position array.
        5. Use the position array to compute velocities and accelerations using the input time step and store in the velocities and accelerations arrays.
        """
        
        dt = params.get('frameTimeStep')
        
        for bubble in self.bubbles:
            trajectory = bubble.getFullTrajectory()
            n_points = len(trajectory)
            positions = np.zeros((n_points, 2), dtype=float)
            
            # Loop through all trajectory points and compute position
            for i, loc in enumerate(trajectory):
                try:
                    obj:Object = self.getObjectFromBubbleLoc(loc)
                    pixelLocs = obj.getAllPixelLocs()
                    
                    # Compute position (center of mass) 
                    positions[i, :] = calculatePixelCentroid(pixelLocs)
                                        
                except ValueError as e:
                    logging.error(f"Error retrieving object for trajectory location {loc}: {e}")
                    positions[i, :] = [np.nan, np.nan]  # Assign NaN for missing positions
            
            bubble.setPositions_fullTrajectory(positions)
            
            if n_points < 2:
                logging.warning(f"Not enough data points to compute velocity for Bubble {bubble.bubbleIndex}.")
                bubble.setVelocities_fullTrajectory(np.array([]))
                bubble.setAccelerations_fullTrajectory(np.array([]))
                continue
            
            # Compute velocities
            vel = (positions[1:] - positions[:-1]) / dt
            bubble.setVelocities_fullTrajectory(vel)
            
            if n_points < 3:
                logging.warning(f"Not enough data points to compute acceleration for Bubble {bubble.bubbleIndex}.")
                bubble.setAccelerations_fullTrajectory(np.array([]))
                continue
            
            # Compute accelerations
            acc = (vel[1:] - vel[:-1]) / dt
            bubble.setAccelerations_fullTrajectory(acc)    


def convert_to_builtin_types(obj):
    """
    Recursively convert NumPy data types within a data structure to native Python types.
    
    Parameters
    ----------
    obj : Any
        The input data structure containing possible NumPy types.
        
    Returns
    -------
    Any
        The converted data structure with native Python types.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_to_builtin_types(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_builtin_types(value) for key, value in obj.items()}
    else:
        return obj