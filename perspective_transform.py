import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

dist_pickle = pickle.load( open( "calibration.p", "rb" ) )
mtx = dist_pickle[0]
dist = dist_pickle[1]

image = cv2.imread('test_image.jpg')
nx = 8
ny = 6

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

warped, M = corners_unwarp(image, nx, ny, mtx, dist)

print(M)
plt.figure()
plt.imshow(image)
plt.figure()
plt.imshow(warped)
plt.show()