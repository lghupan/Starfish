import numpy as np

def slice_image_4d(img, idx):
    if idx == 0: return img[:,:112,:112,:]
    if idx == 1: return img[:,112:,:112,:]
    if idx == 2: return img[:,:112,112:,:]
    if idx == 3: return img[:,112:,112:,:]

def combine_image_4d(img_p):
    img_p_shape = img_p[0].shape
    img = np.zeros((img_p_shape[0], 2*img_p_shape[1], 2*img_p_shape[2], img_p_shape[3]))
    img[:,:112,:112,:] = img_p[0]
    img[:,112:,:112,:] = img_p[1]
    img[:,:112,112:,:] = img_p[2]
    img[:,112:,112:,:] = img_p[3]
    return img