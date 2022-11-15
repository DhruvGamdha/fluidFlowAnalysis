
# function to read and save video
def readAndSaveVid(videoDirPath, saveFramePath, videoFormat):
    import cv2
    from os import listdir
    from os.path import isfile, join
    
    print("videoDirPath:", videoDirPath)
    
    # Get list of all videos in the directory
    allVideosName = [f for f in listdir(videoDirPath) if (isfile(join(videoDirPath, f)) and f.endswith(videoFormat))]
    
    if len(allVideosName) == 0:
        print("No video found in the directory")
        return
    elif len(allVideosName) > 1:
        print("More than one video found in the directory")
        return
    
    video = cv2.VideoCapture(join(videoDirPath, allVideosName[0]))     # Read the video file

    if video.isOpened():
        print("Video file opened successfully")
    else:
        print("Error opening video file")
    
    # get the number of frames in the video
    numFrames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Check if file already exists inside saveFramePath
    numExistingFile = len([f for f in listdir(saveFramePath) if isfile(join(saveFramePath, f))])
    if numExistingFile > 0:
        print("Frames already exist inside {}.".format(saveFramePath))
        
        # Check if the number of frames in the video is same as the number of frames already saved
        if numExistingFile == numFrames:
            print("Number of frames in the video is same as the number of frames already saved.")
            print("Skipping frame extraction.")
            return
        else:
            print("Number of frames in the video is not same as the number of frames already saved.")
            print("Overwriting existing frames.")
    
    count = 0
    while video.isOpened():
        ret, frame = video.read()
        if not ret:         # Check if frame is read correctly
            print("Can't receive frame (stream end?). Exiting ...")
            break
        else: # Save the frame as a .png file
            print("frame number:", count)
            cv2.imwrite(join(saveFramePath,"frame%d.png" % count), frame)
            count += 1

''' def processImages(framePath, binaryPath, top, bottom, left, right, i, j, min_size, connectivity):
    import cv2
    import os
    from os.path import join
    import numpy as np
    import matplotlib.pyplot as plt
    
    allFramesName = [f for f in os.listdir(framePath) if (os.path.isfile(join(framePath, f)) and f.endswith(".png"))]
    
    # NOTE: The best parameters are i=9 and j=5, gaussian adaptive thresholding
    i = 9
    j = 5
    
    for frameName in allFramesName:
        print(frameName)
        
        frame = cv2.imread(join(framePath, frameName), 0)   # Read the frame as grayscale
        
        frame = frame[top:bottom,left:right]    # Crop the frame from certre, original size is 400x800
        
        th2 = cv2.adaptiveThreshold(frame,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,i,j)
        cv2.imwrite(join(binaryPath, "frames", frameName), th2)
 '''
def processImages(framePath, binaryPath, params):
    import cv2
    import os
    from os.path import join
    import numpy as np
    import matplotlib.pyplot as plt
    import skimage.measure as skm
    from tqdm import tqdm
    
    top     = params["top"]
    bottom  = params["bottom"]
    left    = params["left"]
    right   = params["right"]
    blockSize   = params["blockSize"]
    constSub    = params["constSub"]
    min_size    = params["min_size"]
    connectivity    = params["connectivity"]
    
    allFramesName = [f for f in os.listdir(framePath) if (os.path.isfile(join(framePath, f)) and f.endswith(".png"))]
    
    for frameName in tqdm(allFramesName, desc="Processing frames"):
        frame           = cv2.imread(join(framePath, frameName), 0)
        frame           = frame[top:bottom,left:right]        
        th2             = cv2.adaptiveThreshold(frame,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,blockSize,constSub)
        invth2          = 255 - th2    
        labelImg, count = skm.label(invth2, connectivity=connectivity, return_num=True)
        
        for i in range(1, count+1):
            numPixels = np.sum(labelImg == i)
            if numPixels <= min_size:
                invth2[labelImg == i] = 0
                
        th2 = 255 - invth2
        cv2.imwrite(join(binaryPath, frameName), th2)
   
            
def makeConcatVideos(lftFramesPath, rhtFramesPath, nameTemplate, videoName_avi, fps, params, cropOn = True):
    import cv2
    import os
    from os.path import join
    import numpy as np
    from tqdm import tqdm
    
    top     = params["top"]
    bottom  = params["bottom"]
    left    = params["left"]
    right   = params["right"]
    
    allFramesName_unorder   = [f for f in os.listdir(rhtFramesPath) if (os.path.isfile(join(rhtFramesPath, f)) and f.endswith(".png"))]
    allFramesNum            = np.zeros(len(allFramesName_unorder))
    
    for i in range(len(allFramesName_unorder)):
        allFramesNum[i] = int(allFramesName_unorder[i][len(nameTemplate)-6:-4])
        
    allFramesNum = np.sort(allFramesNum).astype(int)    # sort the frame numbers
        
    tempImg2 = cv2.imread(join(lftFramesPath, nameTemplate % allFramesNum[0]), 0)
    tempImg1 = cv2.imread(join(rhtFramesPath, nameTemplate % allFramesNum[0]), 0)
    
    # if cropOn:
    #     tempImg2 = tempImg2[top:bottom,left:right]
    
    height1, width1 = tempImg1.shape
    height2, width2 = tempImg2.shape
    
    videoWidth = width1 + width2            # width of the video
    videoHeight = max(height1, height2)     # height of the video
    
    # vidCodec = cv2.VideoWriter_fourcc(*'XVID')
    vidCodec = cv2.VideoWriter_fourcc(*'mp4v')  # codec for .mp4 file
    video = cv2.VideoWriter(videoName_avi,vidCodec, fps, (videoWidth, videoHeight))
    
    for frameNum in tqdm(allFramesNum, desc="Making video"):
        lft_img = cv2.imread(join(lftFramesPath, nameTemplate % frameNum))
        if lft_img is None:     # Check if frame is read correctly
            exit()
        
        # print('lft frame size:', lft_img.shape)
        
        bin_img = cv2.imread(join(rhtFramesPath, nameTemplate % frameNum))
        if bin_img is None:    # Check if frame is read correctly
            exit()
        # print('rht frame size:', bin_img.shape)
        
        # if cropOn:
        #     lft_img = lft_img[top:bottom,left:right, :]
        
        # print('updated lft frame size:', lft_img.shape)
        # make bin_img and frm_img height the same size as videoHeight by adding black pixels
        bin_img     = cv2.copyMakeBorder(bin_img, 0, videoHeight - height1, 0, 0, cv2.BORDER_CONSTANT, value=[0,0,0])
        lft_img     = cv2.copyMakeBorder(lft_img, 0, videoHeight - height2, 0, 0, cv2.BORDER_CONSTANT, value=[0,0,0])
        concat_img  = np.concatenate((lft_img, bin_img), axis=1)     # Concatenate the two images horizontally (i.e. side-by-side)
        video.write(concat_img) # Write the frame to video file
    cv2.destroyAllWindows()
    video.release() # Now the video is saved in the current directory

def makeSingleVideo(framePath, nameTemplate, videoPath, fps):
    
    import cv2
    import os
    from os.path import join
    import numpy as np
    from tqdm import tqdm
    
    allFramesName_unorder   = [f for f in os.listdir(framePath) if (os.path.isfile(join(framePath, f)) and f.endswith(".png"))]
    allFramesNum            = np.zeros(len(allFramesName_unorder))  # create numpy array to store the frame numbers
    
    # use the name template of type "<string>%d.png" to extract the frame number from each frame name
    for i in range(len(allFramesName_unorder)):
        allFramesNum[i] = int(allFramesName_unorder[i][len(nameTemplate)-6:-4])
        
    allFramesNum    = np.sort(allFramesNum).astype(int)    # sort the frame numbers
    tempImg         = cv2.imread(join(framePath, nameTemplate % allFramesNum[0]), 0)
    height, width   = tempImg.shape
    
    # Define the codec and create VideoWriter object
    # vidCodec = cv2.VideoWriter_fourcc(*'XVID')
    vidCodec    = cv2.VideoWriter_fourcc(*'mp4v')
    video       = cv2.VideoWriter(videoPath, vidCodec, fps, (width, height))
    
    for frameNum in tqdm(allFramesNum, desc="Making video"):
        frm_img = cv2.imread(join(framePath, nameTemplate % frameNum))
        if frm_img is None:     # Check if frame is read correctly
            exit()
        video.write(frm_img)    # feed the concatenated image to the video writer
    cv2.destroyAllWindows()
    video.release()            # Now the video is saved in the current directory

def dropAnalysis(binaryPath, analysisPath, nameTemplate, params):
    import cv2
    import os
    from os.path import join
    import numpy as np
    import skimage.measure as skm
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    
    connectivity = params["connectivity"]
    allFrameNames_unorder = [f for f in os.listdir(binaryPath) if (os.path.isfile(join(binaryPath, f)) and f.endswith(".png"))]    # Read the binary images
    
    allFramesNum = np.zeros(len(allFrameNames_unorder))  # create numpy array to store the frame numbers
    
    # use the name template of type "<string>%d.png" to extract the frame number from each frame name
    for i in range(len(allFrameNames_unorder)):
        allFramesNum[i] = int(allFrameNames_unorder[i][len(nameTemplate)-6:-4])
    
    allFramesNum = np.sort(allFramesNum).astype(int)        # sort the frame numbers
    
    frame_bubbleCount       = []              # bubble count, frame wise
    frame_bubblePixSizes    = []           # pixel sizes of all the bubbles, frame wise
    frame_bubbleVertPos     = []            # vertical position of all the bubbles, frame wise
    
    for frameNum in tqdm(allFramesNum, desc="Analyzing drops"):
        img = cv2.imread(join(binaryPath, nameTemplate % frameNum), 0)    # read the image
        img = cv2.bitwise_not(img)                          # invert the binary image
        
        labelImg, count = skm.label(img, connectivity=connectivity, return_num=True)
        frame_bubbleCount.append(count)
        
        bubble_pixSize = []         # bubble pixel size, bubble wise
        bubble_vertPos = []        # bubble highest pixel vertical position, bubble wise
        
        for i in range(1,count+1):
            rows, cols = np.where(labelImg == i)
            bubble_pixSize.append(len(rows))        # adding ith bubble pixel size
            vertPos = np.min(rows)                  # finding the highest pixel vertical position
            vertPos = img.shape[0] - vertPos        # converting to the original image coordinates
            bubble_vertPos.append(vertPos)          # adding ith bubble highest pixel position
        
        frame_bubblePixSizes.append(bubble_pixSize)     # adding all bubble pixel sizes in the frame
        frame_bubbleVertPos.append(bubble_vertPos)      # adding all bubble highest pixel positions in the frame
        
        # Plot the bubble pixel sizes, with x axis label as the bubble number, plot title as frame number, set the x and y axis limits
        plt.plot(bubble_pixSize)
        plt.xlabel("Bubble Number")
        plt.ylabel("Bubble Pixel Size")
        plt.title("bubble sizes in Frame Number: " + str(frameNum))
        plt.xlim(0, 50)
        plt.ylim(0, 1000)
        plt.savefig(join(analysisPath, "pixSize", "frame" + str(frameNum) + ".png"), dpi=200)
        # plt.show()
        plt.close()
        
        # scatter plot the bubble vertical position, with x axis label as the bubble number, plot title as frame number, set the x and y axis limits
        plt.scatter(range(1, count+1), bubble_vertPos)
        # plt.plot(bubble_vertPos)
        plt.xlabel("Bubble Number")
        plt.ylabel("Bubble Vertical Position")
        plt.title("bubble positions in Frame Number: " + str(frameNum))
        plt.xlim(0, 50)
        plt.ylim(0, img.shape[0])
        plt.savefig(join(analysisPath, "vertPos", "frame" + str(frameNum) + ".png"), dpi=200)
        # plt.show()
        plt.close()
        
        # scatter plot the bubble vertical position, with x axis label as the bubble number, plot title as frame number, set the x and y axis limits
        plt.scatter(range(1, count+1), bubble_vertPos, s=bubble_pixSize)
        plt.xlabel("Bubble Number")
        plt.ylabel("Bubble Vertical Position")
        plt.title("bubble positions in Frame Number: " + str(frameNum))
        plt.xlim(0, 50)
        plt.ylim(0, img.shape[0])
        plt.savefig(join(analysisPath, "dynamicMarker", "frame" + str(frameNum) + ".png"), dpi=200)
        # plt.show()
        plt.close()
    
    # Plot the bubble count, frame wise
    plt.plot(frame_bubbleCount)
    plt.xlabel("Frame Number")
    plt.ylabel("Bubble Count")
    plt.title("bubble count in the video")
    plt.savefig(join(analysisPath, "frame_bubbleCount.png"), dpi=200)
    plt.close()
    
    plt.close('all')