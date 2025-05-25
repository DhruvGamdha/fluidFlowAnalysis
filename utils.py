import logging
import cv2
import numpy as np
from tqdm import tqdm
import skimage.measure as skm
import pathlib as pl
import parse

def readAndSaveVid(videoFramesPathObj, videoFormat, frameNameTemplate, rotate=False):
    """
    Read a single video file and save its frames as images.

    Parameters
    ----------
    videoFramesPathObj : pathlib.Path
        Path to the directory where frames should be saved.
    videoFormat : str
        The file extension of the input video (e.g., '.mp4').
    frameNameTemplate : str
        Template for naming the extracted frames, should contain one placeholder (e.g., 'frame_{:04d}.png').
    rotate : bool, optional
        If True, rotate each frame by 90 degrees counter-clockwise.
    """
    allVideoFiles = [f for f in videoFramesPathObj.parent.iterdir() if (f.is_file() and f.suffix == videoFormat)]
    
    if len(allVideoFiles) != 1:
        logging.error("Expected exactly one video in the folder, found %d", len(allVideoFiles))
        return
    
    video = cv2.VideoCapture(str(allVideoFiles[0]))
    if not video.isOpened():
        logging.error("Error opening video file: %s", allVideoFiles[0])
        return
    
    numFrames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if DoNumExistingFramesMatch(videoFramesPathObj, numFrames):
        return
    
    logging.info("Extracting frames from video. Existing frames in %s may be overwritten.", str(videoFramesPathObj))

    count = 0
    while True:
        ret, frame = video.read()
        if not ret:
            logging.info("No more frames to read or stream ended.")
            break
        
        if rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        outFramePath = videoFramesPathObj / frameNameTemplate.format(count)
        cv2.imwrite(str(outFramePath), frame)
        count += 1

def cropFrames(origFrameDir_pathObj, croppedFrameDir_pathObj, frameNameTemplate, params):
    """
    Crop frames based on parameters specified in 'params'.

    Parameters
    ----------
    origFrameDir_pathObj : pathlib.Path
        Directory containing original frames.
    croppedFrameDir_pathObj : pathlib.Path
        Directory to save cropped frames.
    frameNameTemplate : str
        Template for naming frames.
    params : dict
        Dictionary containing crop boundaries (top, bottom, left, right).
        
    Returns
    -------
    bool
        True if cropping was performed, False if skipped.
    """
    # Ensure source directory exists
    if not origFrameDir_pathObj.is_dir():
        logging.error("Source directory %s does not exist or is not a directory.", origFrameDir_pathObj)
        return False
    
    # Create output directory if it doesn't exist
    croppedFrameDir_pathObj.mkdir(parents=True, exist_ok=True)
    
    # Get frame numbers
    allFramesNum = getFrameNumbers_ordered(origFrameDir_pathObj, frameNameTemplate)
    if not allFramesNum.size:
        logging.warning("No frames found for cropping.")
        return False
        
    # Check if we already have the right number of frames
    if DoNumExistingFramesMatch(croppedFrameDir_pathObj, len(allFramesNum)):
        return False
     
    logging.info("Cropping frames. Existing frames in %s may be overwritten.", str(croppedFrameDir_pathObj))
    
    # Extract crop parameters
    top = params.get("top", 0)
    bottom = params.get("bottom", 0)
    left = params.get("left", 0)
    right = params.get("right", 0)
    
    # Read first frame to validate crop parameters
    first_frame_path = origFrameDir_pathObj / frameNameTemplate.format(allFramesNum[0])
    frame = cv2.imread(str(first_frame_path), 0)
    if frame is None:
        logging.error("Cannot read first frame %s. Aborting crop operation.", first_frame_path)
        return False
    
    # Validate crop parameters
    height, width = frame.shape
    if _are_crop_params_invalid(top, bottom, left, right, height, width):
        logging.warning("Invalid crop parameters: top=%d, bottom=%d, left=%d, right=%d. Using full frame.", top, bottom, left, right)
        top, bottom, left, right = 0, height, 0, width
    
    # Process frames with progress indicator
    for frameNum in tqdm(allFramesNum, desc="Cropping frames"):
        frame_path = origFrameDir_pathObj / frameNameTemplate.format(frameNum)
        frame = cv2.imread(str(frame_path), 0)
        if frame is None:
            logging.warning("Frame %s could not be read. Skipping.", frame_path)
            continue
        
        if top >= height-bottom or left >= width-right:
            logging.warning("Crop parameters would result in empty image. Using full frame instead.")
            cropped = frame.copy()
        else:
            cropped = frame[top:height-bottom, left:width-right]
        outFramePath = croppedFrameDir_pathObj / frameNameTemplate.format(frameNum)
        cv2.imwrite(str(outFramePath), cropped)
    
    return True

def _are_crop_params_invalid(top, bottom, left, right, height, width):
    """
    Check if crop parameters are invalid.
    
    Parameters
    ----------
    top, bottom, left, right : int
        Amount to crop from each edge (top, bottom, left, right).
    height, width : int
        Dimensions of the image.
        
    Returns
    -------
    bool
        True if parameters are invalid, False otherwise.
    """
    return (top < 0 or 
            bottom < 0 or
            left < 0 or 
            right < 0 or
            top + bottom >= height or
            left + right >= width)

def processImages(originalFrameDir_pathObj, binaryFrameDir_pathObj, nameTemplate, params):
    """
    Process images to create binary frames suitable for analysis.

    Parameters
    ----------
    originalFrameDir_pathObj : pathlib.Path
        Directory containing original grayscale frames.
    binaryFrameDir_pathObj : pathlib.Path
        Directory to save the processed binary frames.
    nameTemplate : str
        Template for naming frames.
    params : dict
        Dictionary containing threshold and morphological parameters.
    """
    # Get frame numbers
    allFramesNum = getFrameNumbers_ordered(originalFrameDir_pathObj, nameTemplate)
    if not allFramesNum.size:
        logging.warning("No frames found for processing.")
        return
    
    # Create output directory if it doesn't exist
    binaryFrameDir_pathObj.mkdir(parents=True, exist_ok=True)
    
    # Check if we already have the right number of frames
    if DoNumExistingFramesMatch(binaryFrameDir_pathObj, len(allFramesNum)):
        return
    
    # Extract parameters
    blockSize = params["blockSize"]
    constantSub = params["constantSub"]
    connectivity = params["connectivity"]
    minSize = params["minSize"]
    c_o_kernel = params.get("C_O_KernelSize", 1)
    fill_holes = params.get("fillHoles", True)
    
    # Create kernel once (efficiency)
    kernel = np.ones((c_o_kernel, c_o_kernel), np.uint8)
    
    num_processed = 0
    for frameNum in tqdm(allFramesNum, desc="Processing frames"):
        try:
            # Read frame
            frame_path = originalFrameDir_pathObj / nameTemplate.format(frameNum)
            frame = cv2.imread(str(frame_path), 0)
            if frame is None:
                logging.error("Frame %s could not be read. Skipping.", frame_path)
                continue
            
            # Step 1: Adaptive thresholding
            th2 = cv2.adaptiveThreshold(
                frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize, constantSub
            )
            invth2 = 255 - th2
            
            # Step 2: Morphological closing to clean up noise
            invth2 = cv2.morphologyEx(invth2, cv2.MORPH_CLOSE, kernel)
            
            # Step 3: First label all connected components for consistent size filtering
            labelImg, count = skm.label(invth2, connectivity=connectivity, return_num=True)
            
            # Step 4: Create a filtered mask (remove small objects)
            filtered = np.zeros_like(invth2)
            for i in range(1, count+1):
                region = labelImg == i
                if np.sum(region) >= minSize:
                    filtered[region] = 255
            
            # Step 5: Optionally fill holes in remaining objects
            if fill_holes:
                contours, _ = cv2.findContours(filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                filled = np.zeros_like(filtered)
                for contour in contours:
                    cv2.drawContours(filled, [contour], 0, 255, -1)  # -1 means fill
                result = filled
            else:
                result = filtered
            
            # Step 6: Convert back to original binary format and save
            th2 = 255 - result
            outFramePath = binaryFrameDir_pathObj / nameTemplate.format(frameNum)
            cv2.imwrite(str(outFramePath), th2)
            num_processed += 1
            
        except Exception as e:
            logging.error(f"Error processing frame {frameNum}: {str(e)}")
            continue
    
    logging.info(f"Successfully processed {num_processed} out of {len(allFramesNum)} frames")

def makeConcatVideos(lftFrameDir_pathObj, rhtFrameDir_pathObj, nameTemplate, videoDir_pathObj, fps, params):
    """
    Create a concatenated (side-by-side) video from two directories of frames.

    Parameters
    ----------
    lftFrameDir_pathObj : pathlib.Path
        Directory containing left frames.
    rhtFrameDir_pathObj : pathlib.Path
        Directory containing right frames.
    nameTemplate : str
        Template for naming frames.
    videoDir_pathObj : pathlib.Path
        Directory to save the concatenated video.
    fps : int
        Frames per second for the output video.
    params : dict
        Additional parameters for video creation.
    """
    if checkVideoFileExists(videoDir_pathObj, 'combined'):
        logging.info("Combined video already exists in %s. Skipping creation.", str(videoDir_pathObj))
        return
    
    allFramesNum = getFrameNumbers_ordered(rhtFrameDir_pathObj, nameTemplate)
    if not allFramesNum.size:
        logging.warning("No frames found for concatenation.")
        return

    # Read first frame to get dimensions
    tempImg1 = cv2.imread(str(lftFrameDir_pathObj / nameTemplate.format(allFramesNum[0])), 0)
    tempImg2 = cv2.imread(str(rhtFrameDir_pathObj / nameTemplate.format(allFramesNum[0])), 0)

    if tempImg1 is None or tempImg2 is None:
        logging.error("Could not read initial frames for concatenation. Aborting.")
        return

    height1, width1 = tempImg1.shape
    height2, width2 = tempImg2.shape
    videoWidth = width1 + width2
    videoHeight = max(height1, height2)

    vidCodec = cv2.VideoWriter_fourcc(*'mp4v')
    videoWriter = cv2.VideoWriter(str(videoDir_pathObj / 'videoCombined.mp4'),
                                  vidCodec, fps, (videoWidth, videoHeight))

    for frameNum in tqdm(allFramesNum, desc="Making concatenated video"):
        lft_img = cv2.imread(str(lftFrameDir_pathObj / nameTemplate.format(frameNum)))
        bin_img = cv2.imread(str(rhtFrameDir_pathObj / nameTemplate.format(frameNum)))

        if lft_img is None or bin_img is None:
            logging.error("Missing frames at %d. Stopping video creation.", frameNum)
            videoWriter.release()
            return

        # Adjust images to have the same height by padding
        lft_img = cv2.copyMakeBorder(lft_img, 0, videoHeight - height1, 0, 0, cv2.BORDER_CONSTANT, value=[0,0,0])
        bin_img = cv2.copyMakeBorder(bin_img, 0, videoHeight - height2, 0, 0, cv2.BORDER_CONSTANT, value=[0,0,0])

        concat_img = np.concatenate((lft_img, bin_img), axis=1)
        videoWriter.write(concat_img)

    videoWriter.release()
    cv2.destroyAllWindows()

def makeSingleVideo(framePathObj, nameTemplate, fps):
    """
    Create a single video from a directory of frames.

    Parameters
    ----------
    framePathObj : pathlib.Path
        Directory containing frames.
    nameTemplate : str
        Template for naming frames.
    fps : int
        Frames per second for the output video.
    """
    if checkVideoFileExists(framePathObj.parent, 'isolated'):
        logging.info("Isolated video already exists in %s. Skipping creation.", str(framePathObj.parent))
        return
    
    allFramesNum = getFrameNumbers_ordered(framePathObj, nameTemplate)
    if not allFramesNum.size:
        logging.warning("No frames found to create single video.")
        return

    tempImg = cv2.imread(str(framePathObj / nameTemplate.format(allFramesNum[0])), 0)
    if tempImg is None:
        logging.error("First frame could not be read. Aborting video creation.")
        return

    height, width = tempImg.shape
    vidCodec = cv2.VideoWriter_fourcc(*'mp4v')
    videoWriter = cv2.VideoWriter(str(framePathObj.parent / 'videoIsolated.mp4'),
                                  vidCodec, fps, (width, height))

    for frameNum in tqdm(allFramesNum, desc="Making isolated video"):
        frm_img = cv2.imread(str(framePathObj / nameTemplate.format(frameNum)))
        if frm_img is None:
            logging.error("Frame %d could not be read. Stopping video creation.", frameNum)
            videoWriter.release()
            return
        videoWriter.write(frm_img)

    videoWriter.release()
    cv2.destroyAllWindows()

def checkVideoFileExists(videoDir_pathObj, videType):
    """
    Check if a specific type of video file (combined or isolated) already exists.

    Parameters
    ----------
    videoDir_pathObj : pathlib.Path
        Directory to check.
    videType : str
        'combined' or 'isolated'.

    Returns
    -------
    bool
        True if the specified video already exists, False otherwise.
    """
    if videType == 'combined':
        videoPath = videoDir_pathObj / 'videoCombined.mp4'
    elif videType == 'isolated':
        videoPath = videoDir_pathObj / 'videoIsolated.mp4'
    else:
        logging.warning('Invalid video type specified: %s', videType)
        return False

    return videoPath.exists()

def imgSegmentation(binaryFrameDir_pathObj, nameTemplate, frameNum, connectivity):
    """
    Segment a binary frame into labeled objects.

    Parameters
    ----------
    binaryFrameDir_pathObj : pathlib.Path
        Directory containing binary frames.
    nameTemplate : str
        Template for naming frames.
    frameNum : int
        Frame number to segment.
    connectivity : int
        Connectivity parameter for labeling (4 or 8).

    Returns
    -------
    labelImg : np.ndarray
        Labeled image.
    count : int
        Number of objects found.
    imgShape : tuple
        Shape of the image (height, width).
    """
    img_path = binaryFrameDir_pathObj / nameTemplate.format(frameNum)
    img = cv2.imread(str(img_path), 0)
    if img is None:
        logging.error("Failed to read frame %s for segmentation.", img_path)
        return None, 0, (0,0)
    
    img = cv2.bitwise_not(img)  # Invert binary image
    imgShape = img.shape
    labelImg, count = skm.label(img, connectivity=connectivity, return_num=True)
    return labelImg, count, imgShape

def getFrameNumbers_ordered(framePathObj: pl.Path, nameTemplate, exitOnFail=True):
    """
    Retrieve and sort frame numbers from files in a directory, checking continuity.

    Parameters
    ----------
    framePathObj : pathlib.Path
        Directory containing frames.
    nameTemplate : str
        Template for naming frames.
    exitOnFail : bool, optional
        If True, raise SystemExit if continuity check fails.

    Returns
    -------
    np.ndarray
        Sorted array of frame numbers.
    """
    allFramePathObj_unorder = [f for f in framePathObj.iterdir() if (f.is_file() and f.suffix == ".png")]
    if not allFramePathObj_unorder:
        logging.warning("No frames found in %s.", framePathObj)
        return np.array([])

    allFramesNum = np.zeros(len(allFramePathObj_unorder), dtype=int)
    for i, frame_file in enumerate(allFramePathObj_unorder):
        parsed = parse.parse(nameTemplate, frame_file.name)
        if not parsed:
            logging.error("Filename %s does not match template %s.", frame_file.name, nameTemplate)
            if exitOnFail:
                raise SystemExit("Frame naming template mismatch. Exiting...")
            return np.array([])
        allFramesNum[i] = parsed.fixed[0]

    allFramesNum = np.sort(allFramesNum)

    # Check continuity
    expected = np.arange(allFramesNum[0], allFramesNum[-1]+1)
    if not np.array_equal(allFramesNum, expected):
        logging.warning("Frame numbers are not continuous in %s.", framePathObj)
        if exitOnFail:
            raise SystemExit("Frame numbers are not continuous. Exiting...")
    
    return allFramesNum

def DoNumExistingFramesMatch(frameDir_pathObj: pl.Path, numFramesToCreate):
    """
    Check if the number of existing frames in a directory matches the expected number.

    Parameters
    ----------
    frameDir_pathObj : pathlib.Path
        Directory containing frames.
    numFramesToCreate : int
        Expected number of frames.

    Returns
    -------
    bool
        True if the number of existing frames matches expected, False otherwise.
    """
    numExistingFile = sum(1 for f in frameDir_pathObj.iterdir() if f.is_file() and f.suffix == ".png")
    if numExistingFile == numFramesToCreate:
        logging.info("All %d frames already exist in %s. Skipping creation.",
                     numExistingFile, frameDir_pathObj)
        return True
    return False

def calculatePixelCentroid(pixelLocs):
    """
    Calculate the center of mass of a set of pixel locations.
    """
    rows, cols = pixelLocs
    x = int(round(np.mean(cols)))
    y = int(round(np.mean(rows)))
    return np.array([x, y])

def calculatePixelTopPoint(pixelLocs):
    """
    Calculate the top most point of a set of pixel locations.
    """
    
    rows, cols = pixelLocs
    x = int(round(np.mean(cols)))
    y = int(round(np.min(rows)))
    return np.array([x, y])

def calculatePosition(pixelLocs, position = 'topPoint'):
    
    if position == 'centroid':
        return calculatePixelCentroid(pixelLocs)
    elif position == 'topPoint':
        return calculatePixelTopPoint(pixelLocs)
    else:
        logging.error("Invalid position type: %s. Defaulting to centroid.", position)
        return calculatePixelCentroid(pixelLocs)