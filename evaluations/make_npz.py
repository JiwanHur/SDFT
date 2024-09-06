import numpy as np
import PIL.Image as Image
import pdb
import blobfile as bf

data_path = '/mnt/metface/256/'

img_arr = []
for img in (bf.listdir(data_path)):  
    img = Image.open(f'{data_path}{img}').resize((256,256), resample=Image.BOX).convert("RGB")
    img_arr.append(np.array(img))

img_np = np.array(img_arr)

np.savez(f'{data_path}../samples_256.npz', img_np)