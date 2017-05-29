import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

nx = 10
ny = 6

filename = 'lens_chessboard.png'
image = cv2.imread(filename)

objpoints = []
imgpoints = []

objp = np.zeros((nx*ny, 3), np.float32)
objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)
#print(objp)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

ret, corners = cv2.findChessboardCorners(image, (nx, ny), None)

if ret == True:
    imgpoints.append(corners)
    objpoints.append(objp)

    with_corners = cv2.drawChessboardCorners(gray, (nx, ny), corners, ret)

    #ret -> True/False, mtx -> camera matrix for transformation, dist -> distortion coefficients, rvecs -> rotation vector, tvecs -> translation vectors\
    #rvecs/tvecs -> position of the camera in the world
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image.shape[:2], None, None)

    dst = cv2.undistort(image, mtx, dist, None, mtx)

    plt.figure()
    plt.imshow(image)
    plt.figure()
    plt.imshow(dst)
    plt.show()