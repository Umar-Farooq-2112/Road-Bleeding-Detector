import cv2
import numpy as np 
from basic_functions import binary_mask,filter_color,resize_image,join_images_horizontally,apply_threshold,get_largest_contour
from skimage import measure


def detectRoadBleed(image,mask,pothole,threshold = 15,dark_spots_threshold = 50,name = ''):
    
    if image is None or mask is None or pothole is None:
        return None
    
    bin_mask = binary_mask(mask)
    
    
    if (len(image.shape)>2):
        gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image.copy()
    
    _, binary_image = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY_INV)
    binary_image = cv2.bitwise_and(bin_mask,binary_image)
    cv2.imshow("Binary_Image",resize_image(binary_image))

        # Label connected components
    labels = measure.label(binary_image, connectivity=2)
    props = measure.regionprops(labels)

    h, w = binary_image.shape
    left_connected = False
    right_connected = False

    # Check each region
    for prop in props:
        # Check if the region touches the left edge
        if any(coord[1] == 0 for coord in prop.coords):
            left_connected = True

        # Check if the region touches the right edge
        if any(coord[1] == w - 1 for coord in prop.coords):
            right_connected = True

        # If both conditions are met, they are connected
        if left_connected and right_connected:
            break


    print(left_connected)
    print(right_connected)

    
    # flood_mask = np.zeros((binary_image.shape[0]+2,binary_image.shape[1]+2),np.uint8)
    # h, w = binary_image.shape

    
    # cv2.floodFill(binary_image, flood_mask, (0, 0), 255)
    # # cv2.floodFill(binary_image,flood_mask,(0,0),255,100,255)
    # cv2.floodFill(binary_image, flood_mask, (0, h-1), 255)
    #### Invert the flood_mask
    # flood_mask = cv2.bitwise_not(flood_mask[1:-1, 1:-1])
    # cv2.imshow("Flood Mask",resize_image(flood_mask))
    
    # result = cv2.bitwise_and(binary_image, binary_image, mask=flood_mask)

    # # result = cv2.bitwise_not(result)
    
    
    # # cv2.imshow("Binary_Image",resize_image(binary_image))
    # cv2.imshow("res",resize_image(result))
    
    
    
    
    
    
    # shadow_filter = cv2.bitwise_not(get_largest_contour(binary_image))
    # shadow_filter = (get_largest_contour(binary_image))
    # cv2.imshow("SIN",resize_image(shadow_filter))

    # pothole_filter = binary_mask(pothole)
    # cracks_filter = filter_color(mask,(0,0,255))
    
    # # pothole_filter = cv2.cvtColor(pothole.copy(),cv2.COLOR_BGR2GRAY)
    # # _,pothole_filter = cv2.threshold(pothole_filter,np.mean(pothole_filter)+2*np.std(pothole_filter),255,cv2.THRESH_BINARY)
    # # pothole_filter = binary_mask(pothole_filter)

    # final_filter = cv2.bitwise_or(cracks_filter,pothole_filter)        
    # final_filter = cv2.bitwise_or(final_filter,shadow_filter)

    # _, binary_thresholded_image = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY_INV)
    # binary_thresholded_image = cv2.bitwise_and(bin_mask,binary_thresholded_image)
    
    # # cv2.imshow("Binary Image displaying Black Spots",resize_image(binary_thresholded_image))
    
    # kernel = np.ones((3, 3), np.uint8)
    # resultant_image = cv2.morphologyEx(binary_thresholded_image,cv2.MORPH_OPEN,kernel,iterations=5)
    # resultant_image = binary_mask(resultant_image)
    # opened = resultant_image.copy()

    # cv2.imshow('Opened Binary Image',resize_image(opened))
    
    # resultant_image = cv2.subtract(resultant_image,final_filter)
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
    
    # return apply_threshold(resultant_image,threshold)





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



image = cv2.imread('Dataset/images2/00000007.png')
mask = cv2.imread('Dataset/masks2/00000007.png')
pothole = cv2.imread('Dataset/dm2/00000007.png')

detectRoadBleed(image,mask,pothole,20,"Temp")


