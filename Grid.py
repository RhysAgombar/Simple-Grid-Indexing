import cv2
import numpy as np

img = cv2.imread('grid2.png')

edges = cv2.Canny(img,50,200,apertureSize = 3)

lineimg = np.zeros([img.shape[0],img.shape[1],img.shape[2]])

lines = cv2.HoughLines(edges,2,np.pi/180,200)

mb = np.zeros([lines.shape[0],4])

for i in range (0, lines.shape[0] - 1):
    for rho,theta in lines[i]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(lineimg,(x1,y1),(x2,y2),(0,0,255),2)

cv2.imshow('Lines',lineimg)
cv2.imshow('img',img)