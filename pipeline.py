
## @file pipeline.py
 # @authir Andre N. Zabegaev <speench@gmail.com>
 # pipeline for lane line finding on video


import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

## Class for calibration adn correction of camera distortion
class CameraCorrector(object):

    def __init__(self):
        self.nxy = (9, 6)
        self.mtx = None
        self.dist = None
        pass

    ## Run calibrate camera procedure
     # @param[in] imList list of calibration images file names
     # @param[in] retPictures if true pictures with drown chessboard corners
    def calibrateCamera(self, imList, retPictures = False):

        retPicturesList = []
        retList = []
        imagePoints = []
        objectPoints = []

        objp = np.zeros((self.nxy[0]*self.nxy[1], 3), np.float32)
        objp[:,:2] = np.mgrid[0:self.nxy[0],0:self.nxy[1]].T.reshape(-1,2)

        im_ex_g = None

        for im in imList:
            img = cv2.imread(im)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if im_ex_g is None:
                im_ex_g = gray
            ret, corners = cv2.findChessboardCorners(gray, self.nxy, None)
            if ret:
                if retPictures:
                    grayCopy = np.copy(gray)
                    cv2.drawChessboardCorners(grayCopy, self.nxy, corners, ret)
                    retPicturesList.append(grayCopy)
                imagePoints.append(corners)
                objectPoints.append(objp)

        if imagePoints:
            ret, mtx, dist, _, _ = cv2.calibrateCamera(objectPoints, imagePoints, im_ex_g.shape[::-1], None, None)
            if ret:
                self.mtx = mtx
                self.dist = dist

        if retPictures:
            return retPicturesList

    ## correct distortion for one image
    def correctDistortion(self, img):
        if self.mtx is not None:
            return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)  # Delete this line
        else:
            return None
## Class with different thresholds be applied to the image, base ob Sobel method
class Tresholds(object):

    def __init__(self, img, kernel=3):
        self.sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=kernel)
        self.sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=kernel)

    def abs_sobel_thresh(self, orient='x', thresh=(0, 255)):
        # Take the derivative in x or y given orient = 'x' or 'y'
        if orient == 'x':
            sobel = self.sobelx
        elif orient == 'y':
            sobel = self.sobely
        # Take the absolute value of the derivative or gradient
        abs_sobel = np.absolute(sobel)
        # Scale to 8-bit (0 - 255) then convert to type = np.uint8
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        # Create a mask of 1's where the scaled gradient magnitude
        # is > thresh_min and < thresh_max
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
        # Return this mask as your binary_output image
        return sxbinary

    def mag_thresh(self, thresh=(0, 255)):
        # Calculate the magnitude
        abs_sobel = np.sqrt(self.sobelx**2 + self.sobely**2)
        # Scale to 8-bit (0 - 255) and convert to type = np.uint8
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        # Create a binary mask where mag thresholds are met
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
        # Return this mask as your binary_output image
        return sxbinary

    def dir_threshold(self, thresh=(0, np.pi / 2)):
        # Take the absolute value of the x and y gradients
        abs_sobelx = np.abs(self.sobelx)
        abs_sobely = np.abs(self.sobely)
        # Calculate the direction of the gradient
        grad_dir = np.arctan2(abs_sobely, abs_sobelx)
        # Create a binary mask where direction thresholds are met
        sxbinary = np.zeros_like(grad_dir, dtype='uint8')
        sxbinary[(grad_dir >= thresh[0]) & (grad_dir <= thresh[1])] = 1
        # Return this mask as binary image
        return sxbinary

    def applyThresholds(img):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        g_channel = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # l_channel = hls[:, :, 1]
        s_channel = hls[:, :, 2]

        kernel = 15

        # We apply thresholds for S chanel of HLS and for Grayscale
        th1 = Tresholds(s_channel, kernel=kernel)
        th2 = Tresholds(g_channel, kernel=kernel)

        im1 = th1.abs_sobel_thresh(orient='x', thresh=(37,255))
        im2 = th1.abs_sobel_thresh(orient='y', thresh=(39, 255))
        im3 = th2.mag_thresh(thresh=(79,255))
        im4 = th2.dir_threshold(thresh=(0.382, 1.177))
        im5 = th1.mag_thresh(thresh=(79, 255))
        im6 = th1.dir_threshold(thresh=(0.382, 1.177))

        im56 = np.zeros_like(im5)
        # Take only part of the image close to the horizon for grayscale filter
        im56[:im5.shape[0]//4*3,:] = im5[:im5.shape[0]//4*3,:] & im6[:im5.shape[0]//4*3,:]

        # Combine all filters
        im = (im1 & im2) | (im3 & im4) | im56
        # im = im56
        # plt.figure()
        # plt.imshow(im, cmap='gray')
        # plt.figure()
        # plt.imshow(im2, cmap='gray')
        # plt.figure()
        # plt.imshow(im3, cmap='gray')
        # plt.figure()
        # plt.imshow(im4, cmap='gray')
        # plt.figure()
        # plt.imshow(im5, cmap='gray')
        # plt.show()

        return im

## Perspective transform of road serface to the bird view and back
class PerspectiveTransform(object):

    def __init__(self, img_size):
        offset = 0  # offset for dst points
        cx = 640
        dx1 = 600
        dx2 = 100
        ly = 676
        uy = 460
        self.img_size = (img_size[1], img_size[0])
        src = np.float32([(cx-dx1, ly), (cx+dx1, ly), (cx+dx2, uy), (cx-dx2, uy)])
        dst = np.float32([[offset, self.img_size[1] - offset], [self.img_size[0] - offset, self.img_size[1] - offset],
                          [self.img_size[0] - offset, offset], [offset, offset]])
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.MInv = cv2.getPerspectiveTransform(dst, src)

    def transform(self, img):

        return cv2.warpPerspective(img, self.M, self.img_size)

    def transformInv(self, img):

        return cv2.warpPerspective(img, self.MInv, self.img_size)

# Class for line detection on the image
class LineDetector(object):

    def __init__(self, windowWidth, windowHeight, margin):
        self.windoww = int(windowWidth)
        self.windowh = int(windowHeight)
        self.margin = int(margin)
        self.imgh = None
        self.window = np.ones(windowWidth)

        self.ym_per_pix = 3 / 107  # meters per pixel in y dimension
        self.xm_per_pix = 3.7 / 825  # meters per pixel in x dimension

        pass

    def traceLineOnImg(self, img, startPoint = None, refPoly = None):

        if self.imgh is None:
            self.imgh = int(img.shape[0])
            self.ploty = np.linspace(0, img.shape[0]-1, num=img.shape[0])  # to cover same y-range as image
        elif (img.shape[0] != self.imgh):
            raise ValueError('Image shape has changed')

        # Run full search
        if startPoint is not None:

            new_center = int(startPoint)

            self.imgh = img.shape[0]

            windowsImg = np.zeros_like(img)

            # iterate through the y-axis of the image and maximize number of pixels present in the window
            for level in range(0, self.imgh//self.windowh):
                highBoundary = self.imgh - (level+1)*self.windowh
                lowBoundary = self.imgh - level*self.windowh
                img_layer = np.sum(img[highBoundary:lowBoundary, :], axis=0)
                conv_signal = np.convolve(self.window, img_layer, 'same')

                border_left = max(new_center - self.margin, 0)
                border_right = min(new_center + self.margin, img.shape[1])
                conv_max = np.argmax(conv_signal[border_left:border_right]) + border_left
                if conv_signal[conv_max] > 1:
                    new_center = conv_max
                # add new window to the windows image
                windowsImg[highBoundary:lowBoundary, border_left:border_right] = 1

            # apply windows mask to the image
            pointsImg = windowsImg & img

            # convert points on image to the points array
            pointsArray = np.transpose(pointsImg.nonzero())
            # weghts array, the farther pixel from car, the less weight
            w = pointsArray[:,0]

            pointsArray = np.array(pointsArray)

        # Search from previous traced line
        else:
            if refPoly is not None:
                self.fitx = refPoly[0] * self.ploty ** 2 + refPoly[1] * self.ploty + refPoly[2]
            windowsImg = np.zeros_like(img)
            for x, y in zip(self.fitx, self.ploty):
                windowsImg[int(y), max(0, int(x) - self.margin): min(img.shape[1] - 1, int(x) + self.margin)] = 1
            pointsImg = windowsImg & img

            pointsArray = np.transpose(pointsImg.nonzero())
            w = pointsArray[:,0]

        if pointsArray.shape[0] < 20:
            return None, None, None

        #define 2-order polyline best fit pixels
        self.polyLine = np.polyfit(pointsArray[:,0], pointsArray[:,1], 2, w=w)
        polyLineReal = np.polyfit(pointsArray[:,0]*self.ym_per_pix, pointsArray[:,1]*self.xm_per_pix, 2, w=w)
        self.fitx = self.polyLine[0] * self.ploty ** 2 + self.polyLine[1] * self.ploty + self.polyLine[2]
        lineImg = np.zeros_like(img)
        for x,y in zip(self.fitx, self.ploty):
            if x>=0 and x<lineImg.shape[1]:
                lineImg[int(y),int(x-2):int(x+2)] = 1

        #make line point pixel red, windows green, and result line blue
        template = np.dstack((pointsImg, windowsImg, lineImg))

        return self.polyLine, polyLineReal, template


    def determineTheLaneCurvature(polyLine, point):
        # print(polyLine[0])
        if abs(polyLine[0]) < 1e-04:
            return float("inf")
        return ((1 + (2 * polyLine[0] * point + polyLine[1]) ** 2) ** 1.5) / np.absolute(2 * polyLine[0])

# Class for Lane line tracer on image and video
class LaneTracer(object):

    def __init__(self, calibrationImgDir, imgShape):
        # create instance of camera corrector
        self.cameraCorrector = CameraCorrector()
        # read list of calibration pictures
        imList = [os.path.join(calibrationImgDir, f) for f in os.listdir(calibrationImgDir)]
        if len(imList) == 0:
            raise ValueError("No valid images in camera calibration directory")
        
        # calibrate camera
        self.cameraCorrector.calibrateCamera(imList)
        
        # setup for perspective transform
        self.pTransform = PerspectiveTransform(imgShape[0:2])

        self.firstRun = True

        # Create detectors for left and right line
        self.ldetector = LineDetector(50, 80, 180)
        self.rdetector = LineDetector(50, 80, 180)

        self.polylArray = []
        self.polyrArray = []
        self.polyl_sm = None
        self.polyr_sm = None
        self.badCounter = 0

        self.curvl = None
        self.curvr = None
        self.fromCenter = None

    def traceImg(self, img):
        self.img = img

        # Correct image distortion
        self.img_corrected = self.cameraCorrector.correctDistortion(img)

        # Apply tresholds
        img_tresholds = Tresholds.applyThresholds(self.img_corrected)

        # Apply perspective transform
        self.img_birdView = self.pTransform.transform(self.img_corrected)
        self.img_birdView_th = self.pTransform.transform(img_tresholds)

        # if necessary, find start points for left and right line
        if (self.firstRun):
            # find start points for left and right line
            histogram = np.sum(self.img_birdView_th[self.img_birdView_th.shape[0] // 2:, :], axis=0)
            midpoint = np.int(histogram.shape[0] // 2)
            leftx_base = np.argmax(histogram[:midpoint])
            rightx_base = np.argmax(histogram[midpoint:]) + midpoint
            self.firstRun = False
        else:
            leftx_base = None
            rightx_base = None

        # Trace lines
        self.polyl, polyl_real, self.img_detected_l = self.ldetector.traceLineOnImg(self.img_birdView_th, leftx_base, self.polyl_sm)
        self.polyr, polyr_real, self.img_detected_r = self.rdetector.traceLineOnImg(self.img_birdView_th, rightx_base, self.polyr_sm)

        # Define lines curvature
        if polyl_real is not None and polyr_real is not None:
            self.curvl = LineDetector.determineTheLaneCurvature(polyl_real, (img0.shape[0]-1)*self.ldetector.ym_per_pix)
            self.curvr = LineDetector.determineTheLaneCurvature(polyr_real, (img0.shape[0]-1)*self.rdetector.ym_per_pix)

            lp = img.shape[0]-1
            lowPointL = self.polyl[0] * lp ** 2 + self.polyl[1] * lp + self.polyl[2]
            lowPointR = self.polyr[0] * lp ** 2 + self.polyr[1] * lp + self.polyr[2]
            highPointL = self.polyl[2]
            highPointR = self.polyr[2]
            self.fromCenter = ((lowPointL + lowPointR)/2. - img.shape[1]/2.)*self.ldetector.xm_per_pix

            lowPointDiff = (lowPointR-lowPointL)/img.shape[1]
            highPointDiff = (highPointR-highPointL)/img.shape[1]

            # print(self.curvl, self.curvr, self.fromCenter)

            #if lines are good enough thay are append to array to be smoothed
            if (self.curvl/self.curvr) < 4 and (self.curvl/self.curvr) > 0.25 and self.curvl > 100 and self.curvr > 100 and \
                    lowPointDiff > 0.4 and lowPointDiff < 0.8 and highPointDiff > 0.4 and highPointDiff < 0.8:
                self.polylArray.append(self.polyl)
                self.polyrArray.append(self.polyr)
            else:
                self.badCounter +=1
        else:
            self.badCounter += 1

        if self.badCounter > 5:
            self.firstRun = True

        if len(self.polylArray)>5:
            self.polylArray.pop(0)
            self.polyrArray.pop(0)

        # filter line estimation if possible
        if len(self.polylArray) == 0:
            self.polyl_sm = self.polyl
        else:
            self.polyl_sm = np.average(self.polylArray, axis=0)
        if len(self.polyrArray) == 0:
            self.polyr_sm = self.polyr
        else:
            self.polyr_sm = np.average(self.polyrArray, axis=0)

        # y coordinates array
        ploty = np.linspace(0, self.img_birdView_th.shape[0] - 1, num=self.img_birdView_th.shape[0])
        # x coordinates for left and right line
        plotxl = self.polyl_sm[0] * ploty ** 2 + self.polyl_sm[1] * ploty + self.polyl_sm[2]
        plotxr = self.polyr_sm[0] * ploty ** 2 + self.polyr_sm[1] * ploty + self.polyr_sm[2]
        # points of left line from top to bottom and right from bottom to top
        pts = np.array(np.dstack((np.append(plotxl, plotxr[::-1]), np.append(ploty, ploty[::-1])))[0], dtype='int32')
        # zero channel
        zero_channel = np.zeros_like(self.img_birdView_th).astype(np.uint8)
        polyImg = np.zeros_like(self.img_birdView_th).astype(np.uint8)
        # cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
        # fill area between lines
        cv2.fillConvexPoly(polyImg, pts, 255)
        # perspective untransform
        polyImgUntr = self.pTransform.transformInv(np.dstack((zero_channel, polyImg, zero_channel)))
        self.finalImg = cv2.addWeighted(self.img_corrected, 1, polyImgUntr, 0.3, 0)
        cv2.putText(self.finalImg,"L Curve: {:4.1f}".format(self.curvl), (10,700), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(self.finalImg,"R Curve: {:4.1f}".format(self.curvr), (1000,700), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(self.finalImg,"Dist: {:1.2f}m".format(self.fromCenter), (600,700), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        return self.finalImg

    def showPicures(self):
        # show original image
        plt.figure()
        plt.imshow(self.img)
        # show corrected image
        plt.figure()
        plt.imshow(self.img_corrected)
        # show bird view image
        plt.figure()
        plt.imshow(self.img_birdView)
        plt.figure()
        plt.imshow(self.img_birdView_th)
        # show image with windows, lane line pixels and estimated line
        plt.figure()
        plt.imshow((self.img_detected_l | self.img_detected_r)*255)

    def showFinalImg(self, show = True):

        if show:
            plt.figure()
            plt.imshow(self.finalImg)

        return self.finalImg

    def getMeasurements(self):
        return self.curvl, self.curvr, self.fromCenter


if __name__ == '__main__':

    # calibration images directory
    calImgDir = './camera_cal'
    cameraCorrector = CameraCorrector()
    # imList = [os.path.join(calImgDir, f) for f in os.listdir(calImgDir)]
    # cameraCorrector.calibrateCamera(imList)
    # exImg = cv2.cvtColor(cv2.imread('./camera_cal/calibration2.jpg'), cv2.COLOR_BGR2RGB)
    # exImg_cor = cameraCorrector.correctDistortion(exImg)
    # f, ax = plt.subplots(1, 2)
    # ax[0].imshow(exImg)
    # ax[1].imshow(exImg_cor)
    # plt.show()

    # Read image
    # imgPath = './test_images/straight_lines2.jpg'
    imgPath = './test_images/test3.jpg'
    img0 = cv2.cvtColor(cv2.imread(imgPath), cv2.COLOR_BGR2RGB)

    # create instance of lane line tracer
    tracer = LaneTracer(calImgDir, img0.shape)

    # tracer.traceImg(img0)
    # tracer.showPicures()
    # tracer.showFinalImg()
    # plt.show()

    video_in_name = './project_video.mp4'
    # video_in_name = './challenge_video.mp4'
    sp = os.path.splitext(video_in_name)
    video_out_name = sp[0] + '_res' + sp[1]

    video_in = VideoFileClip(video_in_name)
    video_out = video_in.fl_image(tracer.traceImg)
    video_out.write_videofile(video_out_name, audio=False)
