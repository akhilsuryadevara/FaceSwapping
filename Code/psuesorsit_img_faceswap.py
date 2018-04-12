
# coding: utf-8

# In[1]:


import sys
import numpy
import cv2
import dlib

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

colorBlurFactor = 1
FEATHER_AMOUNT = 11

# JAW_POINTS = list(range(0, 17))
# FACE_POINTS = list(range(17, 68))

LEFT_EYE_POINTS = list(range(42, 48))
RIGHT_EYE_POINTS = list(range(36, 42))

LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_BROW_POINTS = list(range(17, 22))

NOSE_POINTS = list(range(27, 35))
MOUTH_POINTS = list(range(48, 61))

OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS,
]


ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS + RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)

####################################################

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

#A
def get_landmarks(im):
    rects = detector(im, 1)
    
    if len(rects) > 1:
        raise TooManyFaces
    if len(rects) == 0:
        raise NoFaces

    return numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

class TooManyFaces(Exception):
    pass

class NoFaces(Exception):
    pass

def transformation_from_points(points1, points2):
    #This returns the transformation matrix M
   
    points1 = points1.astype(numpy.float64)
    points2 = points2.astype(numpy.float64)

    #translation
    c1 = numpy.mean(points1, axis=0)
    c2 = numpy.mean(points2, axis=0)
    points1 = points1 - c1
    points2 = points2 - c2
    
    #scaling
    s1 = numpy.std(points1)
    s2 = numpy.std(points2)
    points1 = points1/s1
    points2 = points2/s2

    #rotation
    U, S, V = numpy.linalg.svd(points1.T * points2)
    R = (U * V).T
    
    M = numpy.vstack([numpy.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         numpy.matrix([0., 0., 1.])])
    return M

#B
def warp_im(im, M, dshape):
    output_im = numpy.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im

def correct_colours(im1, im2, landmarks1):
    blur_amount = colorBlurFactor * numpy.linalg.norm(
                              numpy.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
                              numpy.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1 #Make it odd value
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    #Prevent dividing by zero
    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

    return (im2.astype(numpy.float64) * im1_blur.astype(numpy.float64) /
                                                im2_blur.astype(numpy.float64))

#C
def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)

def get_face_mask(im, landmarks):
    im = numpy.zeros(im.shape[:2], dtype=numpy.float64)

    for group in OVERLAY_POINTS:
        draw_convex_hull(im,
                         landmarks[group],
                         color=1)

    im = numpy.array([im, im, im]).transpose((1, 2, 0))

    im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

    return im

def process(im1, im2):

    landmarks1 = get_landmarks(im1)
    landmarks2 = get_landmarks(im2)

    M = transformation_from_points(landmarks1[ALIGN_POINTS],
                                   landmarks2[ALIGN_POINTS])

    mask = get_face_mask(im2, landmarks2)
    warped_mask = warp_im(mask, M, im1.shape)
    combined_mask = numpy.max([get_face_mask(im1, landmarks1), warped_mask],
                              axis=0)

    warped_im2 = warp_im(im2, M, im1.shape)
    warped_corrected_im2 = correct_colours(im1, warped_im2, landmarks1)

    output_im = im1 * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask
    return output_im

def nothing(x):
    pass

def save_and_show(name, output_im):
    dst = 'output.jpg'
    cv2.imwrite(dst, output_im)
    #show the output
    out = cv2.imread(dst)
    cv2.imshow(name, out)
    #cv2.waitKey(0)
    

####################################################

#default images
defim1 = "Satoru.jpg"
defim2 = "Prayuth.jpg"

if len(sys.argv) == 1:
    #no argument - use default images
    i1 = defim1
    i2 = defim2
elif len(sys.argv) == 2:
    #only one argument - use default base face
    i1 = defim1
    i2 = sys.argv[1]
else:
    #two arguments - use both
    i1 = sys.argv[1]
    i2 = sys.argv[2]

i1 = defim1
i2 = defim2

im1 = cv2.imread(i1, cv2.IMREAD_COLOR)
im2 = cv2.imread(i2, cv2.IMREAD_COLOR)

si1 = im1
si2 = im2

sw = 0 

output_im = process(im1, im2)
save_and_show('Result Image', output_im)

while 1:
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()
        break
    elif k == ord('i'):
        sw = 0
        cv2.destroyAllWindows()
        colorBlurFactor = 1
        output_im = process(im1, im2)
        save_and_show('Reloaded Orginal Result Image', output_im)
    elif k == ord('b'):
        cv2.destroyAllWindows()
        name = 'Color Blur Factor Adjustable Image'
        cv2.namedWindow(name)  
        cv2.createTrackbar('Color Blur',name,5,50,nothing)
        colorBlurFactor = 10/10.0
        if sw == 1:
            output_im = process(im2, im1)
        else: 
            output_im = process(im1, im2)
        while 1:
            save_and_show(name, output_im)
            ks = cv2.waitKey(1)
            if ks == 27: #ESC
                break
            n = cv2.getTrackbarPos('Color Blur',name)
            colorBlurFactor = n/10.0
            if sw == 1:
                output_im = process(im2, im1)
            else: 
                output_im = process(im1, im2)
            save_and_show(name, output_im)
    elif k == ord('s'):
        cv2.destroyAllWindows()
        if sw == 0:
            output_im = process(im2, im1)
            save_and_show('Result Image', output_im)
            si1 = im2
            si2 = im1
            sw = 1
        else:
            output_im = process(im1, im2)
            save_and_show('Result Image', output_im)
            si1 = im1
            si2 = im2
            sw = 0
    elif k == ord('h'):
        f = open('instructions.txt','r')
        message = f.read()
        print(message)
        f.close()
        
####################################################

