import cv2
import numpy as np
import time

class gridSquare:
    topLeft = np.zeros([2])
    topRight = np.zeros([2])
    bottomLeft = np.zeros([2])
    topRight = np.zeros([2])


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

def isOutlier(image, x,y):
    
    if (abs(y) > image[0] + image[0]/10):
        return True
    elif (abs(x) > image[1] + image[1]/10):
        return True
    else:
        return False



#np.set_printoptions(threshold=np.inf)

start_time = time.time()
img = cv2.imread('images\GridClose2.jpg')

edges = cv2.Canny(img,50,100,apertureSize = 3)

cv2.imshow('edges',edges)


lineimg = np.zeros([img.shape[0],img.shape[1],img.shape[2]])

lines = cv2.HoughLines(edges,2,np.pi/180,500)

rtolerance = 20
ttolerance = 0.25

shape = lines.shape[0]
i = 0

#------ Similar Line Deletion ------
while (i < shape):
    j = 0
    while (j < shape):
        rho = lines[j][0][0]
        theta = lines[j][0][1]

        diffr = abs(lines[i][0][0] - rho)
        difft = abs(lines[i][0][1] - theta)

        # If lines are the same, do nothing
        if (lines[i][0][0] == rho and lines[i][0][1] == theta):
            j = j
            
        # If lines are similar, delete
        elif (diffr < rtolerance):
            if (difft < ttolerance):
                lines = np.delete(lines, j, 0)
                shape-=1
                j-=1
        j+=1
    i+=1

#------ Sorting lines into horizontal or vertical sections ------
vert = []
horiz = []

for i in range (0, lines.shape[0] - 1):
    difft = abs((np.pi/2) - lines[i][0][1])
    difftN = abs((np.pi*1.5) - lines[i][0][1])
    if (difft < np.pi/4 or difftN < np.pi/4):
        horiz.append(lines[i][0])
    else:
        vert.append(lines[i][0])   


vert = np.array(vert)
vert.dtype = [('x', np.float32), ('m', np.float32)]
#print "Unsorted V"
#print vert
vert = np.sort(vert, axis=0, kind='mergesort', order=['x'])
#print "Sorted V"
#print vert
horiz = np.array(horiz)
horiz.dtype = [('x', np.float32), ('m', np.float32)]
#print "Unsorted H"
#print horiz
horiz = np.sort(horiz, axis=0, kind='mergesort', order=['x'])
#print "Sorted H"
#print horiz

vsize = vert.shape[0]
hsize = horiz.shape[0]
lsize = vsize + hsize









lines = np.zeros([vert.shape[0]+horiz.shape[0],2])

for i in range (0, vert.shape[0]):
    lines[i][0] = vert[i]['x']
    lines[i][1] = vert[i]['m']

for i in range (0, horiz.shape[0]):
    lines[i + vert.shape[0]][0] = horiz[i]['x']
    lines[i + vert.shape[0]][1] = horiz[i]['m']

#ntr = np.zeros([SEpoints.shape[0]*SEpoints.shape[0],2])


## Carry the fix through. Switch to 2d array


#------ Calculating Lines ------
SEpoints = np.zeros([lines.shape[0],lines.shape[1],2])

for i in range (0, lines.shape[0] - 1):
    rho = lines[i][0]
    theta = lines[i][1]
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

intgrid = []

#------ Calculating Intersections ------


## Change to loop based on h and v size to eliminate duplicates
for i in range (0, SEpoints.shape[0] - 1):
    for j in range (0, SEpoints.shape[0] - 1):
        if (np.array_equal(SEpoints[i], SEpoints[j]) == False):
            
            if (len(intgrid) > 520):
                i = i
            
            ntr = intersection(SEpoints[i][0],SEpoints[i][1],SEpoints[j][0],SEpoints[j][1])

            if (ntr[0] != np.inf and ntr[1] != np.inf):
                if (isOutlier(img.shape,ntr[0],ntr[1]) == False):
                    intgrid.append(ntr)
                    cv2.circle(lineimg,(int(ntr[0]),int(ntr[1])), 5, (0,255,0), -1)

intgrid = np.array(intgrid).astype(int)

#print intgrid

posLim = intgrid.shape[0]/2

pos = 539
x = vsize - 1
y = hsize - 2

grid = np.zeros([x,y])


#intgrid = np.reshape(intgrid,[hsize,vsize,2])

#intgrid[pos],intgrid[pos+1],intgrid[pos+vsize],intgrid[pos+vsize+1]


#for count in range 



contours = np.array( [intgrid[pos],intgrid[pos+1],intgrid[pos+hsize],intgrid[pos+hsize-1]] ) # top left,bottom left,bottom right,top right
img = np.zeros( (200,200) ) # create a single channel 200x200 pixel black image 
cv2.fillPoly(lineimg, pts =[contours], color=(255,0,0))


cv2.imshow('Lines',lineimg)

print("--- %s seconds ---" % (time.time() - start_time))