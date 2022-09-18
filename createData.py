import numpy as np
import cv2
import random
from scipy import ndimage
import glob
import re


'''Functions'''
def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

def outlinecontour(canny, limit):
    npy = canny
    ratio = 255 / np.amax(npy)
    npy = npy * ratio
    npy = npy.astype(np.uint8)
    accumEdged = np.zeros(npy.shape[:2], dtype="uint8")
    # loop over the blue, green, and red channels, respectively
    for chan in cv2.split(npy):
        #chan = (chan*1.2)
        #chan = np.where(chan>255, 255, chan)
        #chan = chan.astype(np.uint8)
        chan = cv2.medianBlur(chan, 9)
        edged = cv2.Canny(chan, 20, 30)
        accumEdged = cv2.bitwise_or(accumEdged, edged)
    cnts = cv2.findContours(accumEdged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    cnts = sorted(cnts, key=lambda c: cv2.arcLength(c, False), reverse=True)
    perimeters = [cv2.arcLength(cnts[i], True) for i in range(len(cnts))]
    res = list(map(lambda i: i < limit, perimeters)).index(True)
    cnts = cnts[:res]
    npy = np.zeros(npy.shape[:2], dtype="uint8")
    npy = cv2.drawContours(npy, cnts, -1, (255,255,255), 1)
    return npy

def shear(array, shearfactor):
    pat, dan = array.shape
    maxshear = int(shearfactor*pat+0.5)
    zeros = np.zeros((pat,pat+maxshear))
    for vertpixels in range(pat):
        shift = int(shearfactor*vertpixels+0.5)
        zeros[vertpixels:vertpixels+1, shift:shift+pat] = array[vertpixels:vertpixels+1, :pat]
    return zeros

# pos and neg need same transformations

def transform(reference, positive, negative, index, patchwidth):
    # 6 different transformations
    transformation = index
    if transformation == 0:  # rotate image
        selection = random.choice(rotate)
        reference = ndimage.rotate(reference, selection, reshape=False)
        further = random.choice(rotatediff)
        positive = ndimage.rotate(positive, selection + further, reshape=False)
        negative = ndimage.rotate(negative, selection + further, reshape=False)
    elif transformation == 1:  # rescale image
        selection = random.choice(scale)
        reference = cv2.resize(reference, (0, 0), fx=selection, fy=selection)
        diff = random.choice(horiscalediff)
        positive = cv2.resize(positive, (0, 0), fx=selection * diff, fy=selection * diff)
        negative = cv2.resize(negative, (0, 0), fx=selection * diff, fy=selection * diff)
        if reference.shape != (patchwidth, patchwidth): #if not correct size, pad borders with 0s until it is
            shape = reference.shape[0]
            zeros = np.zeros((patchwidth,patchwidth))
            zeros[:shape, :shape] = reference
            reference = zeros
        if positive.shape != (patchwidth, patchwidth):
            shape = positive.shape[0]
            zeros = np.zeros((patchwidth,patchwidth))
            zeros[:shape, :shape] = positive
            positive = zeros
        if negative.shape != (patchwidth, patchwidth):
            shape = negative.shape[0]
            zeros = np.zeros((patchwidth,patchwidth))
            zeros[:shape, :shape] = negative
            negative = zeros
    elif transformation == 2:  # scale horizontal
        selection = random.choice(horiscale)
        reference = cv2.resize(reference, (0, 0), fx=selection, fy=1)
        further = random.choice(horiscalediff)
        positive = cv2.resize(positive, (0, 0), fx=selection * further, fy=1)
        negative = cv2.resize(negative, (0, 0), fx=selection * further, fy=1)
        if reference.shape != (patchwidth, patchwidth): #if not correct size, pad borders with 0s until it is
            shape = reference.shape[1]
            zeros = np.zeros((patchwidth, patchwidth))
            zeros[:patchwidth, :shape] = reference
            reference = zeros
        if positive.shape != (patchwidth, patchwidth): #if not correct size, pad borders with 0s until it is
            shape = positive.shape[1]
            zeros = np.zeros((patchwidth, patchwidth))
            zeros[:patchwidth, :shape] = positive
            positive = zeros
        if negative.shape != (patchwidth, patchwidth): #if not correct size, pad borders with 0s until it is
            shape = negative.shape[1]
            zeros = np.zeros((patchwidth, patchwidth))
            zeros[:patchwidth, :shape] = negative
            negative = zeros
    elif transformation == 3:  # shear
        selection = random.choice(horishear)
        reference = shear(reference, selection)
        further = random.choice(horisheardiff)
        positive = shear(positive, selection + further)
        negative = shear(negative, selection + further)
        if reference.shape != (patchwidth, patchwidth):
            shape = reference.shape[1]
            zeros = np.zeros((shape, shape))
            zeros[:patchwidth, :shape] = reference
            reference = zeros
            ratio = patchwidth/shape
            reference = cv2.resize(reference, (0,0), fx=ratio, fy=ratio)
        if positive.shape != (patchwidth, patchwidth):
            shape = positive.shape[1]
            zeros = np.zeros((shape, shape))
            zeros[:patchwidth, :shape] = positive
            positive = zeros
            ratio = patchwidth/shape
            positive = cv2.resize(positive, (0,0), fx=ratio, fy=ratio)
        if negative.shape != (patchwidth, patchwidth):
            shape = negative.shape[1]
            zeros = np.zeros((shape, shape))
            zeros[:patchwidth, :shape] = negative
            negative = zeros
            ratio = patchwidth / shape
            negative = cv2.resize(negative, (0, 0), fx=ratio, fy=ratio)
    elif transformation == 4:  # vertical disparity shift
        reference = reference
        vert = random.choice(vertdisparity)
        positive = positive[vert:]
        negative = negative[vert:]
        if vert == 1:
            positive = np.vstack((positive, np.zeros((1, patchwidth), dtype='uint8')))
            negative = np.vstack((negative, np.zeros((1, patchwidth), dtype='uint8')))

    elif transformation == 5:  # brightness # contrast shift
        selectioncon = random.choice(contrast)
        selectionbright = random.choice(brightness)
        reference = reference * selectioncon + selectionbright
        further = random.choice(contrastdiff), random.choice(brightnessdiff)
        positive = positive * selectioncon * further[0] + selectionbright + further[1]
        negative = negative * selectioncon * further[0] + selectionbright + further[1]
    else:
        print('you messed up buddy')
    return reference.astype(np.uint8), positive.astype(np.uint8), negative.astype(np.uint8)

'''Hyperparameters'''
rotate = np.arange(-28, 29)
rotatediff = np.arange(-3,4)
scale = (0.8, 0.9, 1)
horiscale = (0.8, 0.9, 1)
horiscalediff = (0.9, 1.0)
horishear = (0,0.1)
horisheardiff = (0,0.1, 0.2, 0.3)
vertdisparity = np.arange(0,2)
contrast = (1,1.1)
contrastdiff = (1,1.1)
brightness = np.arange(0,1.3, 0.1)
brightnessdiff = np.arange(0,0.8, 0.1)

'''Run parameters'''
# img0~img22 = middlebury 2015. img23~img416 = kitti, 417-437 =  middlebury 2006, 438-443 = middlebury 2005, 444-445 = middlebury 2003, 446 - 451 = middlebury 2001
start = 0
end = 2 #uses python indexing
patch = 11
box = int((patch-1)/2)
middleburyfactor = 5
kittifactor = 5
middleburynegrange = 6, 18
kittinegrange = 6, 10
standarddeviationref = 25
standarddeviationpos = 25
noaugNAME = str(start)+'-'+str(end)+'.csv'
augNAME = 'aug'+str(start)+'-'+str(end)+'.csv'
print(augNAME)
'''Closing and deleting file contents'''
noaug = noaugNAME
augment = augNAME
# opening the file with w+ mode truncates the file
f = open(noaug, "w+")
f.close()
aug = open(augment, 'w+')
aug.close()

'''Opening files'''
f = open(noaug, "w")
aug = open(augment,'w')

'''Calling all files'''
# natural sort is a function. it sorts files by order.
# ex. sorted() = 1.png, 10.png, 100.png
#     natural sort = 1.png , 2.png, 3.png, 4.png...
images = glob.glob('data/*.npy') #.npy files are disparity maps. See verifyname.py to get a 0-255 version of these maps
images = natural_sort(images) # 'images' is a list

'''Looping through every image pair with a disparity map'''
for imgnum, dispmap in enumerate(images[start:end], start): # number = image number. Make sure disparity map number is same number as images
    #print(imgnum, dispmap)
    data = np.load(dispmap) # loads disparity map
    data = np.asarray(data)
    height, width = data.shape
    #print(height, width)
    imgR = cv2.imread('PNG/im' + str(imgnum) + 'R.png', 0)
    imgL = cv2.imread('PNG/im' + str(imgnum) + 'L.png', 0) # in order to convert them to numpy array

    # for every pixel in image
    for yval in range(box, height-box):#box, height-box
        for xval in range(box+middleburynegrange[1], width-box-middleburynegrange[1]): # to accomodate for max neg shift
            disparity = int(data[yval][xval])

            # filter to meet criteria
            if imgnum < 22: #if middlebury
                if xval % middleburyfactor == 0 and yval % middleburyfactor == 0:  # if factor of 10
                    if xval - disparity - middleburynegrange[1] - box < 0:  # if out of bounds
                        continue
                    elif disparity == 0 or disparity == -1:  # if disparity is occluded
                        continue
                    neg = random.choice(np.hstack((np.arange(-1*middleburynegrange[1], -1*middleburynegrange[0]+1), np.arange(middleburynegrange[0], middleburynegrange[1]+1))))
                else:
                    continue
            else: #if kitti
                if xval% kittifactor == 0 and yval%kittifactor==0: #if factor of 10
                    if xval - disparity -middleburynegrange[1]-box< 0: # if out of bounds
                        continue
                    elif disparity == 0 or disparity == -1: #if disparity is occluded
                        continue
                    neg = random.choice(np.hstack((np.arange(-1*kittinegrange[1], -1*kittinegrange[0]+1), np.arange(kittinegrange[0], kittinegrange[1]+1))))
                else:
                    continue
            pos = random.choice(np.arange(-1,2)) #modifier to actual value
            ref = imgL[yval-box:yval+box+1, xval-box: xval+box+1]
            if np.std(ref) < standarddeviationref: #if fails the standard deviation check
                continue
            opos = imgR[yval-box:yval+box+1, xval-box-disparity+pos: xval+box+1-disparity+pos] #creates a patch*patch sized window around pixel
            oneg = imgR[yval-box:yval+box+1, xval-box-disparity+neg: xval+box+1-disparity+neg]
            if np.std(opos) < standarddeviationpos: # if positive fails standard deviation check
                continue
            #cv2.imshow('pos', opos)
            #cv2.imshow('neg', oneg)
            #cv2.imshow('ref', ref)
            #cv2.waitKey()
            copyopos = np.reshape(opos, (1,patch**2)) # to convert images to a 1xpatch*patch array for csv storage
            copyoneg = np.reshape(oneg, (1,patch**2))
            copyref = np.reshape(ref, (1, patch**2))
            joinref = np.hstack((copyref,copyopos, copyoneg))
            np.savetxt(f, joinref, fmt='%4d', delimiter=',')
            #np.savetxt(aug, joinref, fmt = '%4d', delimiter = ',') #saving the regular transformation into 'aug'

'''
            for transformations in range(0,6):
                auref, auopos, auoneg = transform(ref, opos, oneg, transformations, patch)
                auref = np.reshape(auref, (1, patch ** 2))
                auopos = np.reshape(auopos, (1, patch ** 2))
                auoneg = np.reshape(auoneg, (1, patch ** 2))
                auoneg = np.reshape(auoneg, (1, patch ** 2))
                joinaug = np.hstack((auref,auopos, auoneg))
                np.savetxt(aug, joinaug, fmt='%4d', delimiter=',')
'''
