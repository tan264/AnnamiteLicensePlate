import cv2
import numpy as np
import math

GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5) #kích cỡ càng to thì càng mờ
ADAPTIVE_THRESH_BLOCK_SIZE = 19 
ADAPTIVE_THRESH_WEIGHT = 9  

# model duoc train san giup doc anh => ki tu
def initModel():
    npaClassifications = np.loadtxt("model/classificationS.txt", np.float32)
    npaFlattenedImages = np.loadtxt("model/flattened_images.txt", np.float32)
    npaClassifications = npaClassifications.reshape(
    (npaClassifications.size, 1))  
    kNearest = cv2.ml.KNearest_create() 
    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

    return kNearest

# tien xu ly anh, tra ve anh xam va anh nhi phan
def preprocess(imgOriginal):

    imgGrayscale = extractValue(imgOriginal)
    # imgGrayscale = cv2.cvtColor(imgOriginal,cv2.COLOR_BGR2GRAY) nên dùng hệ màu HSV
    # Trả về giá trị cường độ sáng ==> ảnh gray
    imgMaxContrastGrayscale = maximizeContrast(imgGrayscale) # để làm nổi bật biển số hơn, dễ tách khỏi nền
    #cv2.imwrite("imgGrayscalePlusTopHatMinusBlackHat.jpg",imgMaxContrastGrayscale)
    height, width = imgGrayscale.shape

    imgBlurred = np.zeros((height, width, 1), np.uint8)
    imgBlurred = cv2.GaussianBlur(imgMaxContrastGrayscale, GAUSSIAN_SMOOTH_FILTER_SIZE, 0)
    #cv2.imwrite("gauss.jpg",imgBlurred)
    #Làm mịn ảnh bằng bộ lọc Gauss 5x5, sigma = 0

    imgThresh = cv2.adaptiveThreshold(imgBlurred, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)

    #Tạo ảnh nhị phân
    return imgGrayscale, imgThresh

def extractValue(imgOriginal):
    height, width, numChannels = imgOriginal.shape
    imgHSV = np.zeros((height, width, 3), np.uint8)
    imgHSV = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV)

    imgHue, imgSaturation, imgValue = cv2.split(imgHSV)
    
    #màu sắc, độ bão hòa, giá trị cường độ sáng
    #Không chọn màu RBG vì vd ảnh màu đỏ sẽ còn lẫn các màu khác nữa nên khó xđ ra "một màu" 
    return imgValue

# Làm cho độ tương phản lớn nhất 
def maximizeContrast(imgGrayscale):
    #Làm cho độ tương phản lớn nhất 
    height, width = imgGrayscale.shape
    
    imgTopHat = np.zeros((height, width, 1), np.uint8)
    imgBlackHat = np.zeros((height, width, 1), np.uint8)
    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)) #tạo bộ lọc kernel
    
    imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuringElement, iterations = 10) #nổi bật chi tiết sáng trong nền tối
    #cv2.imwrite("tophat.jpg",imgTopHat)
    imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement, iterations = 10) #Nổi bật chi tiết tối trong nền sáng
    #cv2.imwrite("blackhat.jpg",imgBlackHat)
    imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat) 
    imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

    #cv2.imshow("imgGrayscalePlusTopHatMinusBlackHat",imgGrayscalePlusTopHatMinusBlackHat)
    #Kết quả cuối là ảnh đã tăng độ tương phản 
    return imgGrayscalePlusTopHatMinusBlackHat

# Tra ve nhung hinh co duong vien khep kin(co kha nang la bien so xe)
def findContours(edged_img, img):
    contours, hierarchy = cv2.findContours(edged_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # sắp xếp các contour có kích thước lớn lên đầu và chỉ lấy 10 phần tử
    contours = sorted(contours,key=cv2.contourArea,reverse=True)[:10]
    # đưa vào list những contour có hình dạng tứ giác
    listContours = []
    for c in contours:
        # cv2.drawContours(img, [c], -1, (0, 255, 0), 3)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.06 * peri, True)
        [x, y, w, h] = cv2.boundingRect(approx.copy())
        ratio = w / h
        # nếu số đỉnh bằng 4 thì ta thêm vào list
        if (len(approx) == 4) and (0.8 <= ratio <= 1.5 or 4.5 <= ratio <= 6.5):
            listContours.append(approx)

    return listContours

def changeContrast(img):
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl,a,b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced_img

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def compute_skew(src_img, center_thres = 0):
    if len(src_img.shape) == 3:
        h, w, _ = src_img.shape
    elif len(src_img.shape) == 2:
        h, w = src_img.shape
    else:
        print('upsupported image type')
    img = cv2.medianBlur(src_img, 3)
    edges = cv2.Canny(img,  threshold1 = 30,  threshold2 = 100, apertureSize = 3, L2gradient = True)
    lines = cv2.HoughLinesP(edges, 1, math.pi/180, 30, minLineLength=w / 1.5, maxLineGap=h/3.0)
    if lines is None:
        return 1

    min_line = 100
    min_line_pos = 0
    for i in range (len(lines)):
        for x1, y1, x2, y2 in lines[i]:
            center_point = [((x1+x2)/2), ((y1+y2)/2)]
            if center_thres == 1:
                if center_point[1] < 7:
                    continue
            if center_point[1] < min_line:
                min_line = center_point[1]
                min_line_pos = i

    angle = 0.0
    nlines = lines.size
    cnt = 0
    for x1, y1, x2, y2 in lines[min_line_pos]:
        ang = np.arctan2(y2 - y1, x2 - x1)
        if math.fabs(ang) <= 30: # excluding extreme rotations
            angle += ang
            cnt += 1
    if cnt == 0:
        return 0.0
    return (angle / cnt)*180/math.pi

def deskew(src_img, change_cons, center_thres):
    if change_cons == 1:
        return rotate_image(src_img, compute_skew(changeContrast(src_img), center_thres))
    else:
        return rotate_image(src_img, compute_skew(src_img, center_thres))

# def sort_contours(cnts):
#     reverse = False
#     i = 0
#     boundingBoxes = [cv2.boundingRect(c) for c in cnts]
#     (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
#                                         key=lambda b: b[1][i], reverse=reverse))
#     return cnts   