import cv2
import numpy as np

def perp( a ) :
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

def intersection(a1,a2, b1,b2) :
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = np.dot( dap, db)
    if (denom == 0):
        return [np.inf, np.inf]
    num = np.dot( dap, dp )
    return (num / denom)*db + b1


img = cv2.imread('grido.png')

edges = cv2.Canny(img,50,300,apertureSize = 3)

lineimg = np.zeros([img.shape[0],img.shape[1],img.shape[2]])

lines = cv2.HoughLines(edges,2,np.pi/180,250)

rtolerance = 30
#ttolerance = 250

shape = lines.shape[0]
i = 0

while (i < shape):
    j = 0
    while (j < shape):
        rho = lines[j][0][0]
        theta = lines[j][0][1]

        diffr = abs(lines[i][0][0] - rho)

        if (lines[i][0][0] == rho and lines[i][0][1] == theta):
            j = j
        elif (diffr < rtolerance):
            lines = np.delete(lines, j, 0)
            shape-=1
            j-=1
            
        j+=1
    i+=1


for i in range (0, lines.shape[0]):
    print lines[i]



SEpoints = np.zeros([lines.shape[0],lines.shape[2],2])

for i in range (0, 7): #lines.shape[0] - 1):
    for rho,theta in lines[i]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 2000*(-b))
        y1 = int(y0 + 2000*(a))
        x2 = int(x0 - 2000*(-b))
        y2 = int(y0 - 2000*(a))


        SEpoints[i][0] = ([x1,y1])
        SEpoints[i][1] = ([x2,y2])

        cv2.line(lineimg,(x1,y1),(x2,y2),(0,0,255),2)

#tolerance = 200
#
#for s1, e1 in SEpoints:
#    index = 0
#    for s2, e2 in SEpoints:
#        diffs = abs(s2 - s1)
#        diffe = abs(e2 - e1)
#
#        if (np.array_equal(diffs,[0,0]) == False and np.array_equal(diffe,[0,0]) == False):
#            if (diffs[0] < tolerance):
#                SEpoints = np.delete(SEpoints,index, 0)
#                break
#            if (diffs[1] < tolerance):
#                SEpoints = np.delete(SEpoints,index, 0)
#                break
#                
#            if (diffe[0] < tolerance):
#                SEpoints = np.delete(SEpoints,index, 0)
#                break
#            if (diffe[1] < tolerance):
#                SEpoints = np.delete(SEpoints,index, 0)
#                break
#
#        index += 1
            

#for x1,y1,x2,y2 in SEpoints:
#    for index, (x3,y3,x4,y4) in enumerate(SEpoints):
#        if y1==y2 and y3==y4: # Horizontal Lines
#            diff = abs(y1-y3)
#        elif x1==x2 and x3==x4: # Vertical Lines
#            diff = abs(x1-x3)
#        else:
#            diff = 0
#
#        if diff < 10 and diff is not 0:
#            del SEpoints[index]

for i in range (0, SEpoints.shape[0] - 1):
    for j in range (0, SEpoints.shape[0] - 1):
        if (np.array_equal(SEpoints[i], SEpoints[j]) == False):
            ntr = intersection(SEpoints[i][0],SEpoints[i][1],SEpoints[j][0],SEpoints[j][1])

            if (ntr[0] != np.inf and ntr[1] != np.inf):
                cv2.circle(lineimg,(int(ntr[0]),int(ntr[1])), 5, (0,255,0), -1)


cv2.imshow('Lines',lineimg)
cv2.imshow('img',img)