import logging
from pathlib import Path
from video import Video
import tqdm

class BubbleAnalysis:
    """
    A class to handle various steps of bubble analysis in video frames.
    """

    def __init__(self, para):
        self.saveVideoFPS = para['saveVideoFPS']  
        self.frameNameTemplate = para['frameNameTemplate']
        self.flowType = para['flowType']
        self.videoFormat = para['inpVideoFormat']        
        self.analysedVideo: Video = None
        self.params = para

    def getFramesFromVideo(self, videoFramesDir_pathObj: Path):
        """
        Extract frames from the input video, optionally rotating if 'inputRotate' is specified.
        """
        from utils import readAndSaveVid
        logging.info("Extracting frames from video...")
        inputRotate = self.params.get('inputRotate', None)
        readAndSaveVid(videoFramesDir_pathObj, self.videoFormat, self.frameNameTemplate, inputRotate)

    def getCroppedFrames(self, origFrameDir_pathObj: Path, croppedFrameDir_pathObj: Path):
        """
        Crop original frames based on provided parameters.
        """
        from utils import cropFrames
        logging.info("Cropping frames...")
        cropFrames(origFrameDir_pathObj, croppedFrameDir_pathObj, self.frameNameTemplate, self.params)

    def getBinaryImages(self, origFrameDir_pathObj: Path, binaryFrameDir_pathObj: Path):
        """
        Convert frames to binary (processed) images.
        """
        from utils import processImages
        logging.info("Processing frames into binary images...")
        processImages(origFrameDir_pathObj, binaryFrameDir_pathObj, self.frameNameTemplate, self.params)

    def createVideoFromFrames(self, frameDir_pathObj: Path):
        """
        Create a single video from the frames stored in 'frameDir_pathObj'.
        """
        from utils import makeSingleVideo
        logging.info(f"Creating video from frames in {frameDir_pathObj}")
        makeSingleVideo(frameDir_pathObj, self.frameNameTemplate, self.saveVideoFPS)

    def createConcatVideo(self, lftFrameDir_pathObj: Path, rhtFrameDir_pathObj: Path, videoDir_pathObj: Path):
        """
        Create a concatenated video from two directories of frames side-by-side.
        """
        from utils import makeConcatVideos
        logging.info(f"Creating concatenated video from {lftFrameDir_pathObj} and {rhtFrameDir_pathObj}")
        makeConcatVideos(lftFrameDir_pathObj, rhtFrameDir_pathObj, self.frameNameTemplate, videoDir_pathObj, self.saveVideoFPS, self.params)

    def extractFrameObjects(self, binaryFrameDir_pathObj: Path, analysisBaseDir_pathObj: Path):
        """
        Extract frame objects and perform initial analysis steps, saving the results for future steps.
        """

        logging.info("Extracting frame objects for analysis...")
        self.analysedVideo = Video()

        # Check if previously analyzed results exist
        filePathObj = analysisBaseDir_pathObj / 'videoFrames.txt'
        if filePathObj.exists():
            logging.info("Loading existing frame analysis from text file...")
            self.analysedVideo.loadFramesFromTextFile(analysisBaseDir_pathObj, self.params)
        else:
            self.analysedVideo = Video.dropAnalysis(binaryFrameDir_pathObj, analysisBaseDir_pathObj, self.frameNameTemplate, self.params)
            self.analysedVideo.saveFramesToTextFile(analysisBaseDir_pathObj)

    def checkVideoFilesOnDisk(self, analysisBaseDir_pathObj: Path):
        """
        Verify that the analyzed video and bubble data was saved correctly.
        """
        logging.info("Verifying saved analysis results...")
        newVideo = Video()
        newVideo.loadFramesFromTextFile(analysisBaseDir_pathObj, self.params)
        newVideo.loadBubblesFromTextFile(analysisBaseDir_pathObj)
        
        if newVideo.isSame(self.analysedVideo):
            logging.info("Video analysis data verified successfully.")
        else:
            logging.error("Video analysis data not saved correctly.")
            exit(1)

    def analyzeBubbleMotion(self, 
                            analysisBaseDir_pathObj: Path,
                            bubbleTrackEllipseDir_pathObj: Path):
        """
        Evaluate and save bubble trajectories.
        """
        logging.info("Evaluating bubble trajectory...")
        filePathObj = analysisBaseDir_pathObj / 'videoBubbles.json'
        if filePathObj.exists():
            logging.info("Loading existing bubble trajectory data...")
            self.analysedVideo.loadBubblesFromTextFile(analysisBaseDir_pathObj)
        else:
            self.analysedVideo.trackObjects(self.params)
            self.analysedVideo.computeBubbleKinematics(self.params, bubbleTrackEllipseDir_pathObj)
            self.analysedVideo.saveBubblesToTextFile(analysisBaseDir_pathObj)

    def plotBubbleTrajectory(self, binaryFrameDir_pathObj: Path, videoDir_pathObj: Path):
        """
        Creates separate videos for each bubble's trajectory.
        """
        logging.info("Plotting bubble trajectories...")
        bubbleListIndex = self.params.get('bubbleListIndex', False)
        if bubbleListIndex is False:
            bubbleListIndex = range(self.analysedVideo.getNumBubbles())
        else:
            bubbleListIndex = range(bubbleListIndex)

        for ind in bubbleListIndex:
            self.analysedVideo.plotTrajectory(ind, binaryFrameDir_pathObj, videoDir_pathObj, self.saveVideoFPS, self.frameNameTemplate)
            
    def plotBubbleKinematics(self, videoDir_pathObj: Path):
        """
        Plot bubble kinematics for each bubble.
        """
        logging.info("Plotting bubble kinematics...")
        for i in tqdm.tqdm(range(self.analysedVideo.getNumBubbles()), desc="Plotting bubble kinematics"):
            self.analysedVideo.plotBubbleKinematics(i, self.params, outDir_pathObj=videoDir_pathObj)
        logging.info("Bubble kinematics plotting completed.")

    def markBubblesOnFrames(self, binaryFrameDir_pathObj: Path, bubbleTrackFramesDir_pathObj: Path):
        """
        Create a video (set of frames) with all bubbles marked.
        """
        from utils import getFrameNumbers_ordered, DoNumExistingFramesMatch

        logging.info("Marking bubbles on frames...")
        allFramesNum = getFrameNumbers_ordered(binaryFrameDir_pathObj, self.frameNameTemplate)
        if DoNumExistingFramesMatch(bubbleTrackFramesDir_pathObj, len(allFramesNum)):
            logging.info("Bubble tracking frames already exist. Skipping marking step.")
            return
        
        bubbleListIndices = self.params.get('bubbleListIndex', False)
        if bubbleListIndices is False:
            bubbleListIndices = range(self.analysedVideo.getNumBubbles())
        else:
            bubbleListIndices = range(bubbleListIndices)

        self.analysedVideo.app2_plotTrajectory(bubbleListIndices, 
                                               binaryFrameDir_pathObj, 
                                               bubbleTrackFramesDir_pathObj, 
                                               self.saveVideoFPS, 
                                               self.frameNameTemplate,
                                               self.params)
