# **Advanced Lane Finding**

This project is for advanced technics of finding lane lines on the camera image.


## Rubric Points

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/571/view) individually and describe how I addressed each point in my implementation.

All code references in this document are for `pipeline.py` file.


### Writeup / README

#### 1. The writeup / README should include a statement and supporting figures / images that explain how each rubric item was addressed, and specifically where in the code each step was handled.

If your are reading this documents, that means I haven't forgotten to submit it.

### Camera Calibration

#### 1. OpenCV functions or other methods were used to calculate the correct camera matrix and distortion coefficients using the calibration chessboard images provided in the repository (note these are 9x6 chessboard images, unlike the 8x6 images used in the lesson). The distortion matrix should be used to un-distort one of the calibration images provided as a demonstration that the calibration is correct. Example of undistorted calibration image is Included in the writeup (or saved to a folder).

All code related to camera calibration is located in `CameraCorrector` class (code line 14). Class has two methods: first one `calibrateCamera` is for calibration procedure, second `correctDistortion` is for image distortion correction after calibration was applied.

`calibrateCamera` method use `cv2` functions `findChessboardCorners` and `calibrateCamera` to define distortion parameters. `correctDistortion` method use `cv2` function `undistort` to remove distortion, with parameters defined in `calibrateCamera` method.

Here is an example of correction applied:

![Distortion correction applied][image_distortion_correction_applied]


### Pipeline (test images)

#### 1. Distortion correction that was calculated via camera calibration has been correctly applied to each image. An example of a distortion corrected image should be included in the writeup (or saved to a folder) and submitted with the project.

In the `traceImg` method of `LaneTracer` class I apply camera calibration to all input images (code line 283):

```
self.img_corrected = self.cameraCorrector.correctDistortion(img)
```

Here is an example of corrected image:

![Distortion correction example][image_distortion_correction_example]


#### 2. A method or combination of methods (i.e., color transforms, gradients) has been used to create a binary image containing likely lane pixels. There is no "ground truth" here, just visual verification that the pixels identified as part of the lane lines are, in fact, part of the lines. Example binary images should be included in the writeup (or saved to a folder) and submitted with the project.

I have used combination of SobelX and SobelY on S-channel of HLS color representation of the image; abs sobel and direction threshold on grayscaled image and abs sobel and direction threshold on S-channel on the part of the image close to the horizon. Here is an example of the result:

![Thresholds example][image_thresholds_example]

#### 3. OpenCV function or other method has been used to correctly rectify each image to a "birds-eye view". Transformed images should be included in the writeup (or saved to a folder) and submitted with the project.

To transform images from camera view to bird view I use class `PerspectiveTransform` (line code 154). It use manually chosen values of transformation and `cv2` functions `getPerspectiveTransform` and `warpPerspective`.

Here is an example of such transformation:

![Bird-view transformation][image_bw_example]

#### 4. Methods have been used to identify lane line pixels in the rectified binary image. The left and right line have been identified and fit with a curved functional form (e.g., spine or polynomial). Example images with line pixels identified and a fit overplotted should be included in the writeup (or saved to a folder) and submitted with the project.

Next I use define start points for left and right lines and trace them over the top of the image with use convolutions.

Here is picture of line pixels, searched windows and estimated curve:

![Traced lines][image_line_traced]

#### 5. Here the idea is to take the measurements of where the lane lines are and estimate how much the road is curving and where the vehicle is located with respect to the center of the lane. The radius of curvature may be given in meters assuming the curve of the road follows a circle. For the position of the vehicle, you may assume the camera is mounted at the center of the car and the deviation of the midpoint of the lane from the center of the image is the offset you're looking for. As with the polynomial fitting, convert from pixels to meters.

This job is done in `traceImg` method of `LaneTracer` class (code lines 335-343) and can be output by uncommenting line 348 or with help of method `getMeasurements` of `LaneTracer` class.

Line curvature us calculated with the formula:

```
(1 + (2 * polyLine[0] * point + polyLine[1]) ** 2) ** 1.5) / np.absolute(2 * polyLine[0])
```

And vehicle position:

```
((lowPointL + lowPointR)/2. - img.shape[1]/2.)*self.ldetector.xm_per_pix
```

#### 6. The fit from the rectified image has been warped back onto the original image and plotted to identify the lane boundaries. This should demonstrate that the lane boundaries were correctly identified. An example image with lanes, curvature, and position from center should be included in the writeup (or saved to a folder) and submitted with the project.

Back transformation is done with method `transformInv` of class `PerspectiveTransform`. Here is the result:


![Result][image_result]

---

### Pipeline (video)

#### 1. The image processing pipeline that was established to find the lane lines in images successfully processes the video. The output here should be a new video where the lanes are identified in every frame, and outputs are generated regarding the radius of curvature of the lane and vehicle position within the lane. The pipeline should correctly map out curved lines and not fail when shadows or pavement color changes are present. The output video should be linked to in the writeup and/or saved and submitted with the project.

Here's a [link to my video result](./project_video_res.mp4)

---

### Discussion

#### 1. Discussion includes some consideration of problems/issues faced, what could be improved about their algorithm/pipeline, and what hypothetical cases would cause their pipeline to fail.

Here is the list of possible issues of current algorithm implementation:

1. Possible presence of other lines on the road with high gradient.
1. Wiped off lines
1. Sunny environment


Here are some ideas how this algorithm can be improved:

1. Apply color filters to detect only white and yellow colors
1. Use inertial measurement unit to estimate vehicle position while camera cannot trace line.


[//]: # (Image References)

[image_distortion_correction_applied]: ./writeup_img/calibr.png "Distortion correction applied"
[image_distortion_correction_example]: ./writeup_img/dist_cor.png "Distortion correction example"
[image_thresholds_example]: ./writeup_img/thresholds.png "Thresholds applied"
[image_bw_example]: ./writeup_img/bv.png "Bird-view transformation"
[image_line_traced]: ./writeup_img/ltrace.png "Traced lines"
[image_result]: ./writeup_img/final.png "Result"
