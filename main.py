import cv2
import numpy as np
import utils

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

kNearest = utils.initModel()
img = cv2.imread("results/061.jpg", 0)
imgROIResized = cv2.resize(img, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
print(imgROIResized.shape)
npaROIResized = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
npaROIResized = np.float32(npaROIResized)
_, npaResults, _, _ = kNearest.findNearest(npaROIResized,k=3)
strCurrentChar = str(chr(int(npaResults[0][0])))
print(strCurrentChar)
cv2.waitKey(0)