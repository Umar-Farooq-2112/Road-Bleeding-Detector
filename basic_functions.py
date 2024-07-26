import cv2
import numpy as np

def best_fit_line(mask):        
    y = first_nonzero_indices(mask)
    x = (np.arange(len(y)))[y != -1]
    y = y[y!=-1]
    if np.all(y == y[0]):
        return False,(y[0],None)
    
    coefficients = np.polyfit(x, y, 1)
    return True,coefficients

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


def crop_required_image(mask,crop_ratio=0.1):
    for i,item in enumerate(mask):
        if np.count_nonzero(item)!=0:
            break
    for j in range(mask.shape[0]-1,-1,-1):
        if np.count_nonzero(mask[j])!=0:
            break
    mask = mask[i:j,:]
    height = mask.shape[0]
    mask = mask[int(height*crop_ratio):height-int(height*crop_ratio),:]
    return mask,i,j


def first_nonzero_indices(arr):
    indices = np.argmax(arr != 0, axis=1)
    all_zero_rows = np.all(arr == 0, axis=1)
    indices[all_zero_rows] = -1
    return indices


def resize_image(image, scale_percent=20):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)    

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

