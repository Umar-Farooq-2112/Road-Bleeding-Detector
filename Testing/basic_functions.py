import cv2
import numpy as np

def join_images_horizontally(images):
    heights = [img.shape[0] for img in images]
    max_height = max(heights)
    for i in range(len(images)):
        if images[i].shape[0] != max_height:
            # Resize image to have the same height as the tallest image
            scale = max_height / images[i].shape[0]
            new_width = int(images[i].shape[1] * scale)
            images[i] = cv2.resize(images[i], (new_width, max_height))

    joined_image = np.hstack(images)
    return joined_image


def binary_mask(image):
    if len(image.shape)>2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary_img = np.zeros_like(image)
    binary_img[image!= 0] = 255
    return binary_img
    

def apply_threshold(image,max_threshold):
    different_pixels = np.count_nonzero(image,axis = 1)
    return np.mean(different_pixels)>max_threshold


def resize_image(image, scale_percent=20):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)    

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

