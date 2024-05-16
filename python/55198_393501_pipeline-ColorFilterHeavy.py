import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

from utils.CameraCalibration import CameraCalibration # import Calibaration

# Make a list of calibration images
chkboard_images = glob.glob('./camera_cal/calibration*.jpg')

# 
calibration = CameraCalibration(chkboard_images)

calibration.mtx, calibration.dist

img = cv2.imread('./camera_cal/calibration1.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
undimg = calibration.undistort_img(img)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(21, 6))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=25)
ax2.imshow(undimg)
ax2.set_title('Undistorted Image', fontsize=25)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

from utils.Thresholding import ImageColorThres

# load road image
img = cv2.imread('./hardpart_image/img_073.jpeg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert image from BGR to RGB

undimg = calibration.undistort_img(img)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(21, 6))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=25)
ax2.imshow(undimg)
ax2.set_title('Undistorted Image', fontsize=25)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

imshape = undimg.shape # image size
print("image shape: ", imshape)

roi_vertices = np.array([[(0,imshape[0]), 
                        (imshape[1]*7/15, imshape[0]*3/5), 
                        (imshape[1]*8/15, imshape[0]*3/5), 
                        (imshape[1],imshape[0])]], 
                         dtype=np.int32) # vertices of roi

colorthres = ImageColorThres(img, roi_vertices)

f, axes = plt.subplots(2, 3, figsize=(15, 7))
f.tight_layout()
axes[0,0].imshow(undimg[:,:,0], cmap='gray')
axes[0,0].set_title('R', fontsize=20)

axes[0,1].imshow(undimg[:,:,1], cmap='gray')
axes[0,1].set_title('G', fontsize=20)

axes[0,2].imshow(undimg[:,:,2], cmap='gray')
axes[0,2].set_title('B', fontsize=20)

hls_ = cv2.cvtColor(undimg, cv2.COLOR_RGB2HLS)

axes[1,0].imshow(hls_[:,:,0], cmap='gray')
axes[1,0].set_title('H', fontsize=20)

axes[1,1].imshow(hls_[:,:,1], cmap='gray')
axes[1,1].set_title('L', fontsize=20)

axes[1,2].imshow(hls_[:,:,2], cmap='gray')
axes[1,2].set_title('S', fontsize=20)

R_ = undimg[:,:,0]
G_ = undimg[:,:,1]
B_ = undimg[:,:,2]
S_ = hls_[:,:,2]
L_ = hls_[:,:,1]

whiteline = np.zeros_like(R_)
comw = R_/255 * G_/255 * B_/255
whiteline[comw>0.63] = 1
plt.imshow(whiteline, cmap='gray')
plt.show()

com = R_/255 * S_/255
yellowline = np.zeros_like(R_)
yellowline[com > 0.8] = 1

plt.imshow(yellowline, cmap='gray')
print(np.amin(com), np.amax(com))
plt.show()

binary_color = np.zeros_like(yellowline)
binary_color = np.maximum(yellowline, whiteline) # combine white and yellow using maximum

plt.imshow(binary_color, cmap='gray')

f, axes = plt.subplots(2, 2, figsize=(15, 9))
f.tight_layout()
axes[0,0].imshow(undimg)
axes[0,0].set_title('Undistorted Image', fontsize=20)

axes[1,0].imshow(yellowline, cmap='gray')
axes[1,0].set_title('Yellow', fontsize=20)

axes[0,1].imshow(whiteline, cmap='gray')
axes[0,1].set_title('White', fontsize=20)

axes[1,1].imshow(binary_color, cmap='gray')
axes[1,1].set_title('Color Thresholds', fontsize=20)

for i in range(2):
    for j in range(2):
        axes[i,j].get_xaxis().set_visible(False)
        axes[i,j].get_yaxis().set_visible(False)

plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

import utils.GradientThresholding as gradThres

redimg = undimg[:,:,0]
redimg_blur = gradThres.gaussian_blur(redimg, 3)

# Choose a Sobel kernel size
ksize = 3 # Choose a larger odd number to smooth gradient measurements

# Apply each of the thresholding functions
gradx = gradThres.abs_sobel_thresh(redimg_blur, orient='x', sobel_kernel=ksize, thresh=(50, 200))
grady = gradThres.abs_sobel_thresh(redimg_blur, orient='y', sobel_kernel=ksize, thresh=(50, 200))

# Combine gradient and trim roi
combined = np.zeros_like(gradx)
combined[((gradx == 1) & (grady == 1))] = 1
binary_grad = gradThres.region_of_interest(combined, roi_vertices)

# plot result
f, axes = plt.subplots(2, 2, figsize=(15, 9))
f.tight_layout()
axes[0,0].imshow(undimg)
axes[0,0].set_title('Undistorted Image', fontsize=20)

axes[1,0].imshow(gradx, cmap='gray')
axes[1,0].set_title('Grad x', fontsize=20)

axes[0,1].imshow(grady, cmap='gray')
axes[0,1].set_title('Grad y', fontsize=20)

axes[1,1].imshow(binary_grad, cmap='gray')
axes[1,1].set_title('Gradient thresholds + Trimming', fontsize=20)

for i in range(2):
    for j in range(2):
        axes[i,j].get_xaxis().set_visible(False)
        axes[i,j].get_yaxis().set_visible(False)

plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

finimg = np.zeros_like(redimg)
finimg[(binary_grad == 1) | (binary_color ==1)] = 1

# plot result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(21, 6))
f.tight_layout()
ax1.imshow(undimg)
ax1.set_title('Undistorted Image', fontsize=25)
plt.imshow(finimg, cmap='gray')
ax2.set_title('Combined color and gradient thresholds', fontsize=25)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

from utils.PerspectiveTransform import PerspectiveTransform

perspective = PerspectiveTransform(finimg)

perspective.M, perspective.Minv

binary_warped = perspective.warp_image(finimg)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(21, 6))
f.tight_layout()
ax1.imshow(finimg, cmap='gray')
ax1.set_title('Original combined image', fontsize=25)
ax2.imshow(binary_warped, cmap='gray')
ax2.set_title('Warped image', fontsize=25)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

from utils.FindLaneLine import *

out_img, window_img, result, left_fit, right_fit, left_curverad, right_curverad, cte = window_search(binary_warped, nwindows=10)
print('radius of curvature: %.3f m, %.3fm' % (left_curverad, right_curverad))
print('cte: %.3f m' % (cte))

# Generate x and y values for plotting
ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

plt.imshow(out_img)
plt.plot(left_fitx, ploty, color='yellow')
plt.plot(right_fitx, ploty, color='yellow')
plt.xlim(0, 1280)
plt.ylim(720, 0)

plt.imshow(result)
plt.plot(left_fitx, ploty, color='yellow')
plt.plot(right_fitx, ploty, color='yellow')
plt.xlim(0, 1280)
plt.ylim(720, 0)

# Assume you now have a new warped binary image 
# from the next frame of video (also called "binary_warped")
# It's now much easier to find line pixels!
nonzero = binary_warped.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])
margin = 100
left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

# Again, extract left and right line pixel positions
leftx = nonzerox[left_lane_inds]
lefty = nonzeroy[left_lane_inds] 
rightx = nonzerox[right_lane_inds]
righty = nonzeroy[right_lane_inds]
# Fit a second order polynomial to each
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)
# Generate x and y values for plotting
ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

# Create an image to draw the lines on
warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

# Recast the x and y points into usable format for cv2.fillPoly()
pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
pts = np.hstack((pts_left, pts_right))

# Draw the lane onto the warped blank image
cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

# Warp the blank back to original image space using inverse perspective matrix (Minv)
newwarp = perspective.unwarp_image(color_warp)

# Combine the result with the original image
result = cv2.addWeighted(undimg, 1, newwarp, 0.3, 0)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(result, 'CTE: %.3f(m)' %(cte) ,(50,90), font, 1,(255,255,255),2,cv2.LINE_AA)
cv2.putText(result, 'Radius of curvature: %.1f, %.1f(m)' %(left_curverad, right_curverad) ,(50,140), font, 1,(255,255,255),2,cv2.LINE_AA)
plt.imshow(result)
plt.title('Detected Lane')



