
# function to read and save video
def readAndSaveVid(videoFile, saveFramePath):
    import cv2
    from os.path import join
    
    print("videoFile:", videoFile)
    
    video = cv2.VideoCapture(videoFile)     # Read the video file

    if video.isOpened():
        print("Video file opened successfully")
    else:
        print("Error opening video file")

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
def processImages(framePath, binaryPath, top, bottom, left, right, blockSize, constSub, min_size, connectivity):
    import cv2
    import os
    from os.path import join
    import numpy as np
    import matplotlib.pyplot as plt
    import skimage.measure as skm
    # from skimage.morphology import remove_small_objects # Import the package for remove small objects
    # Import package for progress bar
    from tqdm import tqdm
    
    allFramesName = [f for f in os.listdir(framePath) if (os.path.isfile(join(framePath, f)) and f.endswith(".png"))]
    
    # Change below for loop to tqdm for loop to show progress bar with file name
    for frameName in tqdm(allFramesName, desc="Processing frames"):
        # print(frameName)
         
        frame = cv2.imread(join(framePath, frameName), 0)   # Read the frame as grayscale
        
        frame = frame[top:bottom,left:right]    # Crop the frame from certre, original size is 400x800
        
        ''' 
        # NOTE: The best parameters are i=9 and j=5, gaussian adaptive thresholding
        i = 41
        j = 7
        # NOTE: The best parameters are min_size=1000 and connectivity=1
        min_size = 1
        connectivity = 1
        '''
        
        th2 = cv2.adaptiveThreshold(frame,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,blockSize,constSub)
        invth2 = 255 - th2   # Invert the image    
        labelImg, count = skm.label(invth2, connectivity=connectivity, return_num=True)
        
        for i in range(1, count+1):         # For loop through all the labels
            numPixels = np.sum(labelImg == i)   # Get the number of pixels in each label
            if numPixels <= min_size:        # If the number of pixels is less than min_size, set the label to 0
                invth2[labelImg == i] = 0
                
        th2 = 255 - invth2   # Invert the image back to normal
        # print('file name : ', join(binaryPath, frameName))
        cv2.imwrite(join(binaryPath, frameName), th2)
   
            
def makeConcatVideos(framePath, binaryPath, nameTemplate, videoName_avi, fps, top, bottom, left, right, cropOn = True):
    import cv2
    import os
    from os.path import join
    import numpy as np
    from tqdm import tqdm
    
    allFramesName_unorder = [f for f in os.listdir(binaryPath) if (os.path.isfile(join(binaryPath, f)) and f.endswith(".png"))]
    
    allFramesNum = np.zeros(len(allFramesName_unorder)) # create numpy array to store the frame numbers
    
    # use the name template of type "<string>%d.png" to extract the frame number from each frame name
    for i in range(len(allFramesName_unorder)):
        allFramesNum[i] = int(allFramesName_unorder[i][len(nameTemplate)-6:-4])
        
    allFramesNum = np.sort(allFramesNum).astype(int)    # sort the frame numbers
        
    tempImg1 = cv2.imread(join(binaryPath, nameTemplate % allFramesNum[0]), 0)
    tempImg2 = cv2.imread(join(framePath, nameTemplate % allFramesNum[0]), 0)
    
    if cropOn:
        tempImg2 = tempImg2[top:bottom,left:right]    # Crop the frame from certre, original size is 400x800
    
    height1, width1 = tempImg1.shape
    height2, width2 = tempImg2.shape
    
    print("temp Img shape:",width1, height1)
    print("temp img type:", type(tempImg1))
    
    # find video size
    videoWidth = width1 + width2
    videoHeight = max(height1, height2)
    
    # vidCodec = cv2.VideoWriter_fourcc(*'XVID')
    vidCodec = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(videoName_avi,vidCodec, fps, (videoWidth, videoHeight))
    
    # Change below for loop to tqdm for loop to show progress bar with file name
    for frameNum in tqdm(allFramesNum, desc="Making video"):
        frm_img = cv2.imread(join(framePath, nameTemplate % frameNum))
        
        if frm_img is None:     # Check if frame is read correctly
            exit()
        
        if cropOn:
            frm_img = frm_img[top:bottom,left:right, :]
        
        bin_img = cv2.imread(join(binaryPath, nameTemplate % frameNum))
        
        if bin_img is None:    # Check if frame is read correctly
            exit()
        
        # make bin_img height the same size as videoHeight by adding black pixels
        bin_img = cv2.copyMakeBorder(bin_img, 0, videoHeight - height1, 0, 0, cv2.BORDER_CONSTANT, value=[0,0,0])
        
        # make frm_img height the same size as videoHeight by adding black pixels
        frm_img = cv2.copyMakeBorder(frm_img, 0, videoHeight - height2, 0, 0, cv2.BORDER_CONSTANT, value=[0,0,0])
        
        # Concatenate the two images horizontally (i.e. side-by-side), frm_img on the left and bin_img on the right
        concat_img = np.concatenate((frm_img, bin_img), axis=1)
        
        video.write(concat_img) # Write the frame to video file
         
    cv2.destroyAllWindows()
    video.release() # Now the video is saved in the current directory

def makeSingleVideo(framePath, nameTemplate, videoPath, fps):
    
    import cv2
    import os
    from os.path import join
    import numpy as np
    
    allFramesName_unorder = [f for f in os.listdir(framePath) if (os.path.isfile(join(framePath, f)) and f.endswith(".png"))]
    
    allFramesNum = np.zeros(len(allFramesName_unorder))  # create numpy array to store the frame numbers
    
    # use the name template of type "<string>%d.png" to extract the frame number from each frame name
    for i in range(len(allFramesName_unorder)):
        allFramesNum[i] = int(allFramesName_unorder[i][len(nameTemplate)-6:-4])
        
    allFramesNum = np.sort(allFramesNum).astype(int)    # sort the frame numbers
    
    tempImg = cv2.imread(join(framePath, nameTemplate % allFramesNum[0]), 0)
    
    height, width = tempImg.shape
    print("temp Img shape:",width, height)
    print("temp img type:", type(tempImg))
    
    # Define the codec and create VideoWriter object
    # vidCodec = cv2.VideoWriter_fourcc(*'XVID')
    vidCodec = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(videoPath, vidCodec, fps, (width, height))
    
    for frameNum in allFramesNum:
        print(frameNum)
        frm_img = cv2.imread(join(framePath, nameTemplate % frameNum))
        
        if frm_img is None:     # Check if frame is read correctly
            exit()
        
        video.write(frm_img)    # feed the concatenated image to the video writer
         
    cv2.destroyAllWindows()
    video.release()            # Now the video is saved in the current directory

def dropAnalysis(binaryPath, analysisPath, nameTemplate, connectivity):
    import cv2
    import os
    from os.path import join
    import numpy as np
    import skimage.measure as skm
    import matplotlib.pyplot as plt
    
    allFrameNames_unorder = [f for f in os.listdir(binaryPath) if (os.path.isfile(join(binaryPath, f)) and f.endswith(".png"))]    # Read the binary images
    
    allFramesNum = np.zeros(len(allFrameNames_unorder))  # create numpy array to store the frame numbers
    
    # use the name template of type "<string>%d.png" to extract the frame number from each frame name
    for i in range(len(allFrameNames_unorder)):
        allFramesNum[i] = int(allFrameNames_unorder[i][len(nameTemplate)-6:-4])
    
    allFramesNum = np.sort(allFramesNum).astype(int)        # sort the frame numbers
    
    frame_bubbleCount = []              # bubble count, frame wise
    frame_bubblePixSizes = []           # pixel sizes of all the bubbles, frame wise
    frame_bubbleVertPos = []            # vertical position of all the bubbles, frame wise
    
    for frameNum in allFramesNum:
        img = cv2.imread(join(binaryPath, nameTemplate % frameNum), 0)    # read the image
        img = cv2.bitwise_not(img)                          # invert the binary image
        
        labelImg, count = skm.label(img, connectivity=connectivity, return_num=True)
        frame_bubbleCount.append(count)
        
        print("Frame Number: ", frameNum, "Bubble Count: ", count)   # Print frame number and bubble count
        
        # Get the bubble details
        bubble_pixSize = []         # bubble pixel size, bubble wise
        bubble_vertPos = []        # bubble highest pixel vertical position, bubble wise
        
        for i in range(1,count+1):
        # for i in range(5,6):
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
        # plt.plot(bubble_vertPos)
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


def roughWork():
       
    ''' 
    # Show the frame in grayscale
    cv2.imshow("Original", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    
    ''' 
    # Apply blurring to the frame
    Gaussian = cv2.GaussianBlur(frame, (3, 3), 0)
    
    # Show the frame in grayscale
    cv2.imshow("Gaussian", Gaussian)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # create a histogram of the blurred grayscale image
    histogram, bin_edges = np.histogram(Gaussian, bins=256)

    fig, ax = plt.subplots()
    plt.plot(bin_edges[0:-1], histogram)
    plt.title("Grayscale Histogram")
    plt.xlabel("grayscale value")
    plt.ylabel("pixels")
    # plt.xlim(0, 1.0)
    plt.show()
    '''
    
    ''' 
    # Apply threshold to the frame starting from 100 to 255 at 30 steps
    # for i in range(50,255,10):
        # ret, thresh = cv2.threshold(frame, i, 255, cv2.THRESH_BINARY)
        # cv2.imwrite(join("thresh", "thresh%d.png" % i), thresh)
        
    # ret, OTSUthresh = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY + + cv2.THRESH_OTSU)
    # cv2.imwrite(join("thresh", "OTSUthresh.png"), OTSUthresh)
    
    # # Otsu's thresholding after Gaussian filtering
    # blur = cv2.GaussianBlur(frame,(5,5),0)
    # ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # cv2.imwrite(join("thresh", "otsu_gaussianFiltering.png"), th3)
    
    # # Adaptive mean thresholding
    # th2 = cv2.adaptiveThreshold(frame,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
    # cv2.imwrite(join("thresh", "adaptiveMeanThresh.png"), th2)
    '''
    
    # # Adaptive Gaussian thresholding
    # th3 = cv2.adaptiveThreshold(frame,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    # cv2.imwrite(join("thresh", "adaptiveGaussThreshold.png"), th3)
    
    # Find right parameter for adaptive thresholding
    # for i in range(3, 20, 2):
    #     j = 2
    #     th2 = cv2.adaptiveThreshold(frame,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,i,j)
    #     cv2.imwrite(join("thresh", "adaptiveGaussThreshold", "agt%d_%d.png" % (i,j)), th2)
    # for j in range(1, 20, 1):
    #     i = 9
    #     th2 = cv2.adaptiveThreshold(frame,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,i,j)
    #     cv2.imwrite(join("thresh", "adaptiveGaussThreshold", "agt%d_%d.png" % (i,j)), th2)