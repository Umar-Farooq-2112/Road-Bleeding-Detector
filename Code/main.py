import cv2
import numpy as np 
from basic_functions import binary_mask,filter_color,apply_threshold,get_largest_contour

    

def detectRoadBleed(image,mask,pothole,threshold = 15):
    
    if image is None or mask is None or pothole is None:
        return None
    
    bin_mask = binary_mask(mask)
    
    
    if (len(image.shape)>2):
        gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image.copy()
    
    _, binary_image = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)
    binary_image = cv2.bitwise_and(bin_mask,binary_image)
    shadow_filter = cv2.bitwise_not(get_largest_contour(binary_image))

    pothole_filter = binary_mask(pothole)
    cracks_filter = filter_color(mask,(0,0,255))
    
    pothole_filter = cv2.cvtColor(pothole.copy(),cv2.COLOR_BGR2GRAY)
    _,pothole_filter = cv2.threshold(pothole_filter,np.mean(pothole_filter)+2*np.std(pothole_filter),255,cv2.THRESH_BINARY)
    pothole_filter = binary_mask(pothole_filter)

    final_filter = cv2.bitwise_or(cracks_filter,pothole_filter)        
    final_filter = cv2.bitwise_or(final_filter,shadow_filter)

    _, binary_thresholded_image = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY_INV)
    binary_thresholded_image = cv2.bitwise_and(bin_mask,binary_thresholded_image)
    
    kernel = np.ones((3, 3), np.uint8)
    resultant_image = cv2.morphologyEx(binary_thresholded_image,cv2.MORPH_OPEN,kernel,iterations=5)
    resultant_image = binary_mask(resultant_image)
    
    resultant_image = cv2.subtract(resultant_image,final_filter)
    return apply_threshold(resultant_image,threshold)


# ## Ignore the Code Below
# masks = []
# names = []
# potholes = []
# for i in range (0,10):
#     names.append(f'Dataset/images2/0000000{i}.png')
#     masks.append(f'Dataset/masks2/0000000{i}.png')
#     potholes.append(f'Dataset/dm2/0000000{i}.png')
# for i in range (0,6):
#     names.append(f'Dataset/images2/0000001{i}.png')
#     masks.append(f'Dataset/masks2/0000001{i}.png')
#     potholes.append(f'Dataset/dm2/0000001{i}.png')


# for i in range(len(names)):
#     image = cv2.imread(names[i])
#     mask = cv2.imread(masks[i])
#     pothole = cv2.imread(potholes[i])
#     print(f"{names[i]}:    {detectRoadBleed(image,mask,pothole,20)}")


# image = cv2.imread('Dataset/images2/00000001.png')
# mask = cv2.imread('Dataset/masks2/00000001.png')
# pothole = cv2.imread('Dataset/dm2/00000001.png')
# detect_road_bleed(image,mask,pothole)


