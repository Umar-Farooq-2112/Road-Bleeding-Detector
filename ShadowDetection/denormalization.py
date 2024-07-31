import numpy as np
import pandas as pd
import cv2
import os
from basic_functions import resize_image,join_images_horizontally


def join_images(images):
    if len(images) != 6:
        raise ValueError("Input array must contain exactly 6 images.")
    
    # Join the first 3 images horizontally
    first_horizontal = np.hstack(images[:3])
    
    # Join the last 3 images horizontally
    second_horizontal = np.hstack(images[3:])
    
    # Join the two resulting images vertically
    final_image = np.vstack([first_horizontal, second_horizontal])
    
    return final_image

def denormalized_image_coordinates(
    norm_coords: np.ndarray, width: int, height: int
) -> np.ndarray:
    size = max(width, height)
    p = np.empty((len(norm_coords), 2))
    p[:, 0] = norm_coords[:, 0] * size - 0.5 + width / 2.0
    p[:, 1] = norm_coords[:, 1] * size - 0.5 + height / 2.0
    return p

def readtracks(trackpath):
    df = pd.read_csv(trackpath+'tracks.csv',delimiter='\t',skiprows=[0],names=['image','track_id','feature_index', 'normalized_x', 'normalized_y', 'size' ,'R','G','B','na','nas'])
    df = df.drop(['na','nas'],axis=1)
    xypoints = np.vstack([df['normalized_x'].to_numpy(),df['normalized_y'].to_numpy()])
    xypoints = denormalized_image_coordinates(xypoints.T,1920,1080)

    images = df['image']
    df = df.rename(columns={'normalized_x':'x','normalized_y':'y'})
    df['x'] = xypoints[:,0]
    df['y'] = xypoints[:,1]
    return df

# path = 'Dataset/Reconstructions/'
# subsections = []
# for i in range(0,10):
#     subsections.append(f'{path}subsection00{i}/')
# for i in range(10,100):
#     subsections.append(f'{path}subsection0{i}/')
# for i in range(100,116):
#     subsections.append(f'{path}subsection{i}/')

# for x,subsection in enumerate(subsections):
#     images_dir = subsection+'images/'    
#     df = readtracks(subsection)
    
#     images = []
#     files = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
#     for file in files:        
#         points = df[df['image'] == file][['x','y']].values
#         image = cv2.imread(images_dir+file)
#         for i in points:
#             cv2.circle(image,(int(i[0]),int(i[1])),5,(255,255,255),5)
#         images.append(resize_image(image))
#     if len(images)>0:
#         cv2.imwrite(f"Results/{x}.png",join_images(images))
#     else:
#         print(f"{len(files)}+{len(images)}")
            
        

df = readtracks('section/')

# print(df.columns)

points = df[["x",'y']]


# print(points)

