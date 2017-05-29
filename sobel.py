import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimage

def show(images):
    for im in images:
        plt.figure()
        plt.imshow(im, cmap='gray')
    plt.show()

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



image = cv2.imread('signs_vehicles_xygrad.png')

thresh_min = 20
thresh_max = 100

sobelx_binary = abs_sobel_thresh(image, 'x', thresh_min, thresh_max)
sobely_binary = abs_sobel_thresh(image, 'y', thresh_min, thresh_max)
sobelxy_binary = mag_thresh(image, 9, (thresh_min, thresh_max))
direction = dir_threshold(image, 15, (0.7,1.3))

final = np.zeros_like(direction)
final[(sobelx_binary==1)&(sobely_binary==1)&(sobelxy_binary==1)&(direction==1)] = 1

combined = np.zeros_like(direction)
combined[((sobelx_binary == 1) & (sobely_binary == 1)) | ((sobelxy_binary == 1) & (direction == 1))] = 1

show((image, sobelx_binary, sobely_binary, sobelxy_binary, direction, final, combined))

