import cv2
import numpy as np 
from basic_functions import binary_mask,filter_color,resize_image,join_images_horizontally,apply_threshold

def get_largest_contour(mask):
    res = np.zeros_like(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(res, [largest_contour], -1, (255), thickness=cv2.FILLED)
    return res


    

def detect_road_bleed(image,mask,pothole,threshold = 15):
    bin_mask = binary_mask(mask)
    
    
    if (len(image.shape)>2):
        gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image.copy()
    
    _, binary_image = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY)
    binary_image = cv2.bitwise_and(bin_mask,binary_image)
    shadow_filter = cv2.bitwise_not(get_largest_contour(binary_image))

    pothole_filter = binary_mask(pothole)
    cracks_filter = filter_color(mask,(0,0,255))
    
    #Temporary code
    pothole_filter = cv2.cvtColor(pothole.copy(),cv2.COLOR_BGR2GRAY)
    print(np.mean(pothole_filter))
    print(np.std(pothole_filter))
    _,pothole_filter = cv2.threshold(pothole_filter,np.mean(pothole_filter)+2*np.std(pothole_filter),255,cv2.THRESH_BINARY)
    pothole_filter = binary_mask(pothole_filter)

    final_filter = cv2.bitwise_or(cracks_filter,pothole_filter)        
    final_filter = cv2.bitwise_or(final_filter,shadow_filter)

    _, binary_thresholded_image = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY_INV)
    binary_thresholded_image = cv2.bitwise_and(bin_mask,binary_thresholded_image)
    
    kernel = np.ones((3, 3), np.uint8)
    resultant_image = cv2.morphologyEx(binary_thresholded_image,cv2.MORPH_OPEN,kernel,iterations=5)
    resultant_image = binary_mask(resultant_image)
    opened = resultant_image.copy()
    # cv2.imshow("res",resize_image(binary_thresholded_image))

    
    
    resultant_image = cv2.subtract(resultant_image,final_filter)
    # cv2.imshow('resultant_image',resize_image(resultant_image))
    w = image.shape[0]
    
    joined1 = join_images_horizontally([cracks_filter,pothole_filter,shadow_filter])
    cv2.putText(joined1,"Cracks Filter",(10,300),1,15,150,5)
    cv2.putText(joined1,"Pothole Filter",(w-int(0.4*w),300),1,15,150,5)
    cv2.putText(joined1,"Shadow Filter",(int(1.5*w),300),1,15,150,5)
    
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