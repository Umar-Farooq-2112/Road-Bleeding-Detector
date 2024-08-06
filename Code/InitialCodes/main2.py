import cv2
import numpy as np 


def binary_mask(image):
    if len(image.shape)>2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary_img = np.zeros_like(image)
    binary_img[image!= 0] = 255
    return binary_img
    

def apply_threshold(image,max_threshold):
    different_pixels = np.count_nonzero(image,axis = 1)
    return np.mean(different_pixels)>max_threshold


def get_largest_contours(mask):
    res = np.zeros_like(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    largest_contour = contours[0]
    cv2.drawContours(res, [largest_contour], -1, (255), thickness=cv2.FILLED)    
    for i in range(1,len(contours)):
        if cv2.contourArea(largest_contour)/(2)<cv2.contourArea(contours[i]):
            largest_contour = contours[i]
        else:
            break
        cv2.drawContours(res, [largest_contour], -1, (255), thickness=cv2.FILLED)    
    return res

def filter_color(image,color):
    mask = np.zeros(image.shape[:2],dtype=np.uint8)
    mask[np.all(image == color,axis=-1)] = 255
    return mask




def detectRoadBleed(image,mask,pothole,threshold = 30,dark_spots_threshold = 80):
    
    if image is None or mask is None or pothole is None:
        return None
    
    bin_mask = binary_mask(mask)
    
    
    if (len(image.shape)>2):
        gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image.copy()
    
    _, binary_image = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY_INV)
    binary_image = cv2.bitwise_and(bin_mask,binary_image)
    # cv2.imshow("thresh",resize_image(binary_image))

    
    shadow_filter = (get_largest_contours(binary_image))
    cracks_filter = filter_color(mask,(0,0,255))    
    if len(pothole.shape)>2:
        pothole_filter = cv2.cvtColor(pothole,cv2.COLOR_BGR2GRAY)
        pothole_filter = binary_mask(pothole_filter)
    else:
        pothole_filter = pothole
        
    final_filter = cv2.bitwise_or(cracks_filter,pothole_filter)        
    final_filter = cv2.bitwise_or(final_filter,shadow_filter)

    _, binary_thresholded_image = cv2.threshold(gray_image, dark_spots_threshold, 255, cv2.THRESH_BINARY_INV)
    binary_thresholded_image = cv2.bitwise_and(bin_mask,binary_thresholded_image)
    
    resultant_image = binary_mask(binary_thresholded_image)
    resultant_image = cv2.subtract(resultant_image,final_filter)

    return apply_threshold(resultant_image,threshold)
