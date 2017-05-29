import numpy as np
import cv2
import matplotlib.pyplot as plt

def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0) if orient == 'x' else cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    abs_sobel = np.absolute(sobel)
    scaled = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary = np.zeros_like(scaled)
    binary[(scaled <= thresh_max) & (scaled >= thresh_min)] = 1
    return binary

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0,255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel = np.sqrt(sobelx**2 + sobely**2)
    scaled = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary = np.zeros_like(scaled)
    binary[(scaled <= mag_thresh[1]) & (scaled >= mag_thresh[0])] = 1
    return binary

'''
Direction of the gradient is computed with arctan(sobely/sobelx)
'''
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    direction = np.arctan2(abs_sobely, abs_sobelx)
    binary = np.zeros_like(direction)
    binary[(direction <= thresh[1]) & (direction >= thresh[0])] = 1
    return binary

def show_gray(images):
    for im in images:
        plt.figure()
        plt.imshow(im, cmap='gray')

def show(images):
    for im in images:
        plt.figure()
        plt.imshow(im)

def s_select(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s = hls[:,:,2]
    binary_output = np.zeros_like(s)
    binary_output[(s <= thresh[1]) & (s > thresh[0])] = 1
    return binary_output

def corners_unwarp(img, nx, ny, mtx, dist):
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)
    gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    if ret == True:
        src = np.float32([corners[0],corners[nx-1],corners[-1],corners[-nx]])
        offset = 100
        img_size = (gray.shape[1], gray.shape[0])
        dst = np.float32([
            [offset, offset], 
            [img_size[0]-offset, offset], 
            [img_size[0]-offset, img_size[1]-offset], 
            [offset, img_size[1]-offset]])
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(undistorted, M, img_size, flags=cv2.INTER_LINEAR)
        return warped, M

