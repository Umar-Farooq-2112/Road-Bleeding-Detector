import cv2
import numpy as np 
from basic_functions import binary_mask,filter_color,resize_image,join_images_horizontally,apply_threshold




    

def detect_road_bleed(image,mask,pothole,shadows=None,threshold = 15):
    bin_mask = binary_mask(mask)
    
    if (len(image.shape)>2):
        gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image.copy()
    
    _, binary_thresholded_image = cv2.threshold(gray_image, 50, 255, cv2.THRESH_TOZERO_INV)
    binary_thresholded_image = cv2.bitwise_and(bin_mask,binary_thresholded_image)
    
    
    kernel = np.ones((3, 3), np.uint8)
    resultant_image = cv2.morphologyEx(binary_thresholded_image,cv2.MORPH_OPEN,kernel,iterations=5)
    resultant_image = binary_mask(resultant_image)
    opened = resultant_image.copy()

    pothole_filter = binary_mask(pothole)    
    cracks_filter = filter_color(mask,(0,0,255))
    # shadow_filter = binary_mask(shadows)
    # extra code
    dm = cv2.cvtColor(pothole.copy(),cv2.COLOR_BGR2GRAY)
    _,dm = cv2.threshold(dm,150,255,cv2.THRESH_TOZERO)
    dm = binary_mask(dm)
    # cv2.imshow("dm",resize_image(dm))
    final_filter = cv2.bitwise_or(cracks_filter,dm)
    
    
    
    # final_filter = cv2.bitwise_or(cracks_filter,pothole_filter)
    
    resultant_image = cv2.subtract(resultant_image,final_filter)
    # print(np.count_nonzero(resultant_image))
    
    
    
    w = image.shape[0]
    
    # joined1 = join_images_horizontally([cracks_filter,pothole_filter,shadow_filter])
    joined1 = join_images_horizontally([cracks_filter,pothole_filter])
    cv2.putText(joined1,"Cracks Filter",(10,300),1,15,150,5)
    cv2.putText(joined1,"Pothole Filter",(w-int(0.4*w),300),1,15,150,5)
    # cv2.putText(joined1,"Shadow Filter",(int(1.5*w),300),1,15,150,5)
    
    joined2 = join_images_horizontally([opened,final_filter,resultant_image])
    cv2.putText(joined2,"Black Spots",(0,300),1,15,150,5)
    cv2.putText(joined2,"Final Filter",(w-int(0.4*w),300),1,15,150,5)
    cv2.putText(joined2,"Result",(int(1.5*w),300),1,15,150,5)
    
    cv2.imshow('Cracks+Potholes+Shadows',resize_image(joined1))
    cv2.imshow('Image+Filter+Result',resize_image(joined2))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return apply_threshold(resultant_image,threshold)
        
    
    
# image = cv2.imread('image.png')
# mask = cv2.imread('mask.png')
# pothole = cv2.imread('depthmap.png')
# detect_road_bleed(image,mask,pothole)
image = cv2.imread('Dataset/images2/00000001.png')
mask = cv2.imread('Dataset/masks2/00000001.png')
pothole = cv2.imread('Dataset/dm2/00000001.png')
detect_road_bleed(image,mask,pothole)