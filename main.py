from utils import readAndSaveVid, processImages

def saveVidFrames():
    videoFileName = "silOilPFD_20uLh_500fps_continuousExtrude.mp4"
    videoFilePath = "data/continuousFlow/"
    readAndSaveVid(videoFileName, videoFilePath)

def processData():
    framePath = "data/continuousFlow/frames/"
    binaryPath = "data/continuousFlow/binary/"
    processImages(framePath, binaryPath)

# main function
if __name__ == "__main__":
    # saveVidFrames()
    processData()
    # import cv2
    # import os
    # from os.path import join
    # import numpy as np
    # import matplotlib.pyplot as plt
    
    # framePath = "data/continuousFlow/frames/"
    # binaryPath = "data/continuousFlow/binary/"
    # allFramesName = [f for f in os.listdir(framePath) if os.path.isfile(join(framePath, f))]
    
    # for frameName in allFramesName:
    #     print(frameName)
    #     # read the frame
    #     frame = cv2.imread(join(framePath, frameName), 0)
        
    #     # Crop the frame from certre, original size is 400x800
    #     top = 100
    #     bottom = 700
    #     left = 100
    #     right = 300
    #     frame = frame[top:bottom,left:right]
        
    #     # NOTE: The best parameters are i=9 and j=5, gaussian adaptive thresholding
    #     i = 9
    #     j = 5
        
    #     th2 = cv2.adaptiveThreshold(frame,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,i,j)
    #     cv2.imwrite(join(binaryPath, frameName), th2)
        
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
    