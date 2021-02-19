import os, sys, glob, ray
import numpy as np
from PIL import Image
ray.init()

dataset_fn = "./cars_224_224_color.npy"
train_list = sorted( glob.glob("./cars/split_train/*.jpg") )
test_list = sorted( glob.glob("./cars/split_test/*.jpg") )

if os.path.exists(dataset_fn):
    x_train, x_test = np.load(dataset_fn, allow_pickle=True)
    print("Loaded prepared dataset: " + dataset_fn)
else:
    
    @ray.remote
    def proc_img(f):
        #  width, height in PIL
        img = Image.open(f).convert('RGB').resize((224,224), resample=Image.LANCZOS)
        return np.array(img)
    
    x_train = np.array(ray.get([proc_img.remote(x) for x in train_list]))
    x_test  = np.array(ray.get([proc_img.remote(x) for x in  test_list]))
    
    np.save(dataset_fn, [x_train, x_test])

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

print(x_train.shape, x_test.shape) #