
# function to read and save video
def readAndSaveVid(videoFileName, videoFilePath, saveFramePath):
    import cv2
    from os.path import join

    videoFile = join(videoFilePath, videoFileName)

    # Read the .mp4 file
    video = cv2.VideoCapture(videoFile)

    if video.isOpened():
        print("Video file opened successfully")
    else:
        print("Error opening video file")

    count = 0
    while video.isOpened():
        ret, frame = video.read()
        # Check if frame is read correctly
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        else: # Save the frame as a .png file
            cv2.imwrite(join(saveFramePath, "frames","frame%d.png" % count), frame)
            count += 1

def processImages(framePath, binaryPath, top, bottom, left, right):
    import cv2
    import os
    from os.path import join
    import numpy as np
    import matplotlib.pyplot as plt
    
    # framePath = "data/continuousFlow/frames/"
    # binaryPath = "data/continuousFlow/binary/"
    allFramesName = [f for f in os.listdir(framePath) if (os.path.isfile(join(framePath, f)) and f.endswith(".png"))]
    
    for frameName in allFramesName:
        print(frameName)
        # read the frame
        frame = cv2.imread(join(framePath, frameName), 0)
        
        # Crop the frame from certre, original size is 400x800
        frame = frame[top:bottom,left:right]
        
        # NOTE: The best parameters are i=9 and j=5, gaussian adaptive thresholding
        i = 9
        j = 5
        
        th2 = cv2.adaptiveThreshold(frame,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,i,j)
        cv2.imwrite(join(binaryPath, frameName), th2)
            
def makeConcatVideos(framePath, binaryPath, videoName_avi, top, bottom, left, right):
    import cv2
    import os
    from os.path import join
    import numpy as np
    
    # framePath = "data/continuousFlow/frames/"
    # binaryPath = "data/continuousFlow/binary/"
    allBiFramesName = [f for f in os.listdir(binaryPath) if (os.path.isfile(join(binaryPath, f)) and f.endswith(".png"))]
    
    # arrange the frames in order of their names (frame0.png, frame1.png, ...)
    # Create list of size len(allBiFramesName) with all elements as 0
    allBiFrames = [0] * len(allBiFramesName)
    
    # For each frame name, read the frame and save it in the list at the index of the frame number
    for frameName in allBiFramesName:
        frameNum = int(frameName[5:-4])
        allBiFrames[frameNum] = frameName
        
    # Now allBiFrames is a list of all the frames in order of their names
    # Create a video with all the frames in the list
    
    # Define the codec and create VideoWriter object
    # video = cv2.VideoWriter_fourcc(*'XVID')
    tempImg = cv2.imread(join(binaryPath, allBiFrames[0]), 0)
    # width, height = tempImg.shape
    height, width = tempImg.shape
    print("temp Img shape:",width, height)
    print("temp img type:", type(tempImg))
    video = cv2.VideoWriter(videoName_avi,0, 60.0, (2*width, height))
    
    # bin_img = cv2.imread(join(binaryPath, allBiFrames[0]), 0)
    # frm_img = cv2.imread(join(framePath, allBiFrames[0]), 0)
    # frm_img = frm_img[100:700,100:300]
    
    # # Concatenate the two images horizontally (i.e. side-by-side), frm_img on the left and bin_img on the right
    # concat_img = np.concatenate((frm_img, bin_img), axis=1)
    # print("concat img shape:",concat_img.shape)
    # print("concat img type:", type(concat_img))
    
    for frame in allBiFrames:
        print(frame)
        bin_img = cv2.imread(join(binaryPath, frame))
        # Check if frame is read correctly
        if bin_img is None:
            exit()
        frm_img = cv2.imread(join(framePath, frame))
        # Check if frame is read correctly
        if frm_img is None:
            exit()
        frm_img = frm_img[top:bottom,left:right, :]
        
        # Concatenate the two images horizontally (i.e. side-by-side), frm_img on the left and bin_img on the right
        concat_img = np.concatenate((frm_img, bin_img), axis=1)
        
        # feed the concatenated image to the video writer
        video.write(concat_img)
         
    cv2.destroyAllWindows()
    video.release()
    # Now the video is saved in the current directory


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
    