import cv2
import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import imutils

cv2.namedWindow('Output', cv2.WINDOW_NORMAL)

cap = cv2.imread("Sample4.webp")

cv2.imshow('input', cap)

frame = cap
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

lower_red = np.array([6, 20, 20])
upper_red = np.array([30, 255, 255])

mask = cv2.inRange(hsv, lower_red, upper_red)
res = cv2.bitwise_and(frame, frame, mask=mask)
cv2.imshow('mask', mask)

kernel = np.ones((7, 7), np.uint8)
opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
# erosion = cv2.erode(mask, kernel, iterations=1)

# kernel = np.ones((3, 3), np.uint8)
# # sure background area
# sure_bg = cv2.dilate(opening, kernel, iterations=3)
# cv2.imshow('sure_bg', sure_bg)
# # Finding sure foreground area
# dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
# cv2.imshow('dist_transform', dist_transform)
# ret, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)
# # Finding unknown region
# sure_fg = np.uint8(sure_fg)
# cv2.imshow('sure_fg', sure_fg)
# unknown = cv2.subtract(sure_bg, sure_fg)
# cv2.imshow('unknown', unknown)
#
# # Marker labelling
# ret, markers = cv2.connectedComponents(sure_fg)
# # Add one to all labels so that sure background is not 0, but 1
# markers = markers + 1
# # Now, mark the region of unknown with zero
# markers[unknown == 255] = 0
#
# markers = cv2.watershed(frame, markers)
# frame[markers == -1] = [255, 0, 0]

# compute the exact Euclidean distance from every binary
# pixel to the nearest zero pixel, then find peaks in this
# distance map
D = ndimage.distance_transform_edt(opening)
localMax = peak_local_max(D, indices=False, min_distance=40,
	labels=opening)
# perform a connected component analysis on the local peaks,
# using 8-connectivity, then appy the Watershed algorithm
markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
labels = watershed(-D, markers, mask=opening)
print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))

# loop over the unique labels returned by the Watershed
# algorithm
for label in np.unique(labels):
	# if the label is zero, we are examining the 'background'
	# so simply ignore it
	if label == 0:
		continue
	# otherwise, allocate memory for the label region and draw
	# it on the mask
	mask = np.zeros(frame.shape[:2], dtype="uint8")
	mask[labels == label] = 255

	# detect contours in the mask and grab the largest one
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
							cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	c = max(cnts, key=cv2.contourArea)
	# draw a circle enclosing the object
	((x, y), r) = cv2.minEnclosingCircle(c)
	# cv2.circle(cap, (int(x), int(y)), int(r), (0, 255, 0), 2)
	cv2.putText(cap, "#{}".format(label), (int(x) - 10, int(y)),
				cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 0, 50), 2)
# show the output image
cv2.imshow("Output", cap)

cv2.imshow('frame', frame)
cv2.imshow('masked', res)
# cv2.imshow('mask', opening)
# cv2.imshow('res', res)

k = cv2.waitKey()

cv2.destroyAllWindows()