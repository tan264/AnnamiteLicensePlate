import cv2
import numpy as np
import utils

Min_char = 0.01
Max_char = 0.09

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

def detect_image(img, kNearest):
    list_image_detected_plate = []

    imgGrayscaleplate, imgThreshplate = utils.preprocess(img)
    canny_image = cv2.Canny(imgThreshplate, 250, 255)  # Canny Edge
    kernel = np.ones((3, 3), np.uint8)
    dilated_image = cv2.dilate(canny_image, kernel, iterations=1)
    # cv2.imwrite("results/threashImage.jpg", imgThreshplate)
    # cv2.imwrite("results/dilated_image.jpg", dilated_image)
    # cv2.imwrite("results/canny_image.jpg", canny_image)

    list = utils.findContours(dilated_image, img)
    # print(len(list))
    for a in range(0, len(list)):
        # x, y, w, h = cv2.boundingRect(list[a])
        # lấy ảnh biển số xe thông qua ảnh ban đầu
        # cropped_image = img[y:y+h, x:x+w]
        # list_image_detected_plate.append(cropped_image)
        # cv2.imwrite(str(a) + ".jpg", cropped_image)

        # Tiến hành cắt vùng ảnh chứa biển số
        mask = np.zeros(imgGrayscaleplate.shape, np.uint8)
        new_image = cv2.drawContours(mask, [list[a]], 0, 255, -1, )
        # cv2.imwrite("results/mask_image" + str(a) + ".jpg", new_image)

        (x, y) = np.where(mask == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))
        roi = img[topx:bottomx + 1, topy:bottomy + 1] # cắt trên ảnh gốc
        # cv2.imwrite("results/before_rotate.jpg", roi)
        imgThresh = imgThreshplate[topx:bottomx + 1, topy:bottomy + 1] # cắt trên ảnh nhị phân

        # roi = utils.deskew(roi, 0, 0)
        # imgThresh = utils.deskew(imgThresh,0, 0)

        angle = utils.compute_skew(roi) # tính góc nghiêng trên ảnh gốc
        roi = utils.rotate_image(roi, angle) # xoay ảnh gốc
        # cv2.imwrite("results/after_rotate.jpg", roi)
        imgThresh = utils.rotate_image(imgThresh, angle) # xoay ảnh gốc
        # cv2.imwrite("results/thresh_plate.jpg", imgThresh)

        roi = cv2.resize(roi, (0, 0), fx=3, fy=3) # phóng to ảnh lên 3 lần
        imgThresh = cv2.resize(imgThresh, (0, 0), fx=3, fy=3)
        # cropped_image = utils.deskew(cropped_image, 0, 0)

        list_image_detected_plate.append(roi) # thêm vào list các ảnh biển số xe phát hiện được

        # Tiền xử lý vùng ảnh chứa biển số xe
        kerel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thre_mor = cv2.morphologyEx(imgThresh, cv2.MORPH_DILATE, kerel3)
        cont, hier = cv2.findContours(thre_mor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Character segmentation
        char_x_ind = {}
        char_x = []
        height, width, _ = roi.shape
        roiarea = height * width
        # print ("roiarea",roiarea)
        for ind, cnt in enumerate(cont):
            # cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 3)s
            area = cv2.contourArea(cnt)
            (x, y, w, h) = cv2.boundingRect(cont[ind])
            ratiochar = w / h
            if (Min_char * roiarea < area < Max_char * roiarea) and (0.25 < ratiochar < 0.7):
                if x in char_x:  # Sử dụng để dù cho trùng x vẫn vẽ được
                    x = x + 1
                char_x.append(x)
                char_x_ind[x] = ind
        
        # cv2.imwrite("results/thresh_contour_character.jpg", roi)
        # Character recognition
        if len(char_x) in range(7, 10):
            cv2.drawContours(img, [list[a]], -1, (0, 255, 0), 3)

            char_x = sorted(char_x)
            strFinalString = ""
            first_line = ""
            second_line = ""
            for i in char_x:
                (x, y, w, h) = cv2.boundingRect(cont[char_x_ind[i]])
                cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

                imgROI = thre_mor[y:y + h, x:x + w]
                
                imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
                image_name = "results/" + str(a) + str(i) + ".jpg"
                # cv2.imwrite(image_name, imgROI)
                # cv2.imwrite(image_name, imgROIResized)

                npaROIResized = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
                npaROIResized = np.float32(npaROIResized)
                _, npaResults, _, _ = kNearest.findNearest(npaROIResized,k=3)  # call KNN function find_nearest;
                strCurrentChar = str(chr(int(npaResults[0][0])))
                cv2.putText(roi, strCurrentChar, (x, y + 50), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 0), 3)

                if (y < height / 3):  # decide 1 or 2-line license plate
                    first_line = first_line + strCurrentChar
                else:
                    second_line = second_line + strCurrentChar

            # cv2.imwrite("results/bouding_character.jpg", roi)    
            strFinalString = first_line + second_line
            print("\n License Plate " + strFinalString)
            cv2.putText(img, strFinalString, (topy, topx-20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    
    return img, list_image_detected_plate

if __name__ == "__main__":
    originalImg = cv2.imread("images/15.jpg")
    img = cv2.resize(originalImg, dsize=(1920, 1080))
    kNearest = utils.initModel()
    img, list_image_detected_plate = detect_image(img, kNearest)
    for index in range(0, len(list_image_detected_plate)):
        detected_image = cv2.resize(list_image_detected_plate[index], None, fx=0.5, fy=0.5)
        cv2.imshow(str(index), detected_image)

    img = cv2.resize(img, None, fx=0.5, fy=0.5)
    cv2.imshow("image", img)
    cv2.waitKey(0)