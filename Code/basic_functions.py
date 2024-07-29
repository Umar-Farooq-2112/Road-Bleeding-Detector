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


def get_largest_contour(mask):
    res = np.zeros_like(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(res, [largest_contour], -1, (255), thickness=cv2.FILLED)
    return res

def filter_color(image,color):
    mask = np.zeros(image.shape[:2],dtype=np.uint8)
    mask[np.all(image == color,axis=-1)] = 255
    return mask

