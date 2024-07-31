import cv2
import numpy as np 
from basic_functions import binary_mask,filter_color,resize_image,join_images_horizontally,apply_threshold


def detectRoadBleed(image,mask,pothole,threshold = 15,dark_spots_threshold = 50,name = ''):
    
    if image is None or mask is None or pothole is None:
        return None
    
    bin_mask = binary_mask(mask)
    
    
    if (len(image.shape)>2):
        gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image.copy()

    pothole_filter = binary_mask(pothole)
    cracks_filter = filter_color(mask,(0,0,255))
    
    # # pothole_filter = cv2.cvtColor(pothole.copy(),cv2.COLOR_BGR2GRAY)
    # # _,pothole_filter = cv2.threshold(pothole_filter,np.mean(pothole_filter)+2*np.std(pothole_filter),255,cv2.THRESH_BINARY)
    # # pothole_filter = binary_mask(pothole_filter)

    final_filter = cv2.bitwise_or(cracks_filter,pothole_filter)        

    _, binary_thresholded_image = cv2.threshold(gray_image, dark_spots_threshold, 255, cv2.THRESH_BINARY_INV)
    binary_thresholded_image = cv2.bitwise_and(bin_mask,binary_thresholded_image)
    
    # cv2.imshow("Binary Image displaying Black Spots",resize_image(binary_thresholded_image))
    
    kernel = np.ones((3, 3), np.uint8)
    resultant_image = cv2.morphologyEx(binary_thresholded_image,cv2.MORPH_OPEN,kernel,iterations=5)
    resultant_image = binary_mask(resultant_image)
    opened = resultant_image.copy()

    cv2.imshow('Opened Binary Image',resize_image(opened))
    
    resultant_image = cv2.subtract(resultant_image,final_filter)
    
    # w = image.shape[0]
    # joined1 = join_images_horizontally([cracks_filter,pothole_filter,shadow_filter])
    # cv2.putText(joined1,"Cracks Filter",(10,300),1,15,150,5)
    # cv2.putText(joined1,"Pothole Filter",(w-int(0.4*w),300),1,15,150,5)
    # cv2.putText(joined1,"Shadow Filter",(int(1.5*w),300),1,15,150,5)
    
    # joined2 = join_images_horizontally([opened,final_filter,resultant_image])
    # cv2.putText(joined2,"Black Spots",(0,300),1,15,150,5)
    # cv2.putText(joined2,"Final Filter",(w-int(0.4*w),300),1,15,150,5)
    # cv2.putText(joined2,"Result",(int(1.5*w),300),1,15,150,5)
    
    # cv2.imwrite(f'{name}_filters.png',resize_image(joined1))
    # cv2.imwrite(f'{name}_results.png',resize_image(joined2))
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
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
#     print(f"{names[i]}:    {detectRoadBleed(image,mask,pothole,20,f"Results/{i}")}")



# image = cv2.imread('Dataset/images2/00000007.png')
# mask = cv2.imread('Dataset/masks2/00000007.png')
# pothole = cv2.imread('Dataset/dm2/00000007.png')

# detectRoadBleed(image,mask,pothole,20,"Temp")


