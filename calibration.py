import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle

nx = 8
ny = 6

objpoints = []
imgpoints = []

objp = np.zeros((nx*ny, 3), np.float32)
objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

images = glob.glob('calibration_wide/GO*.jpg')

for idx, filename in enumerate(images):
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)
        cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
        #plt.figure()
        #plt.imshow(img)
#plt.show()

image = cv2.imread('test_image.jpg')
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image.shape[:2], None, None)
dst = cv2.undistort(image, mtx, dist, None, mtx)

pickle.dump((mtx, dist), open("calibration.p", "wb"))

plt.figure()
plt.imshow(image)
plt.figure()
plt.imshow(dst)
#plt.show()

