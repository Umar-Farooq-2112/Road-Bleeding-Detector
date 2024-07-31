import numpy as np
import pandas as pd


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


# df = readtracks(pathToTracks)

#     #logIt(df)
#     track_id1 = df[df['image'] == img1]['track_id'].values
