# %%
import cv2
from matplotlib import pyplot as plt
import numpy as np

def resize_img(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    #return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return cv2.resize(img, dim, interpolation = cv2.INTER_NEAREST)

# %%
potsdam_folder_rgb = '/media/zoltan/ext/ai/isprs/Potsdam/2_Ortho_RGB'
potsdam_folder_lbl = '/media/zoltan/ext/ai/isprs/Potsdam/5_Labels_all'
img_resize_pcnt = 25
crop_size_x, crop_size_y = 256, 256

# %%
import glob
img_rgb_file_paths = glob.glob(potsdam_folder_rgb + '/*.tif')

# %%
data = []
for img_rgb_path in img_rgb_file_paths:
    print('Processing: ', img_rgb_path)
    img_rgb = cv2.imread(img_rgb_path)
    img_rgb_filename = img_rgb_path.split('/')[-1]
    img_lbl_filename = img_rgb_filename.replace('RGB', 'label')
    img_lbl = cv2.imread(potsdam_folder_lbl + '/' + img_lbl_filename)

    img_rgb = resize_img(img_rgb, img_resize_pcnt)
    img_lbl = resize_img(img_lbl, img_resize_pcnt)

    #plt.figure(figsize=(10,10))
    #plt.imshow(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

    for start_x in range(0, img_rgb.shape[0], crop_size_x):
        for start_y in range(0, img_rgb.shape[1], crop_size_y):
            cropped_image = img_rgb[start_x:start_x+crop_size_x, start_y:start_y+crop_size_y, :]
            cropped_label = img_lbl[start_x:start_x+crop_size_x, start_y:start_y+crop_size_y, :]
            if cropped_image.shape[0] == crop_size_x and cropped_image.shape[1] == crop_size_y:
                data.append({
                    'meta': {
                        'rgb_filename': img_rgb_filename,
                        'lbl_filename': img_lbl_filename
                    },
                    'rgb': cropped_image,
                    'lbl': cropped_label,
                })

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,10))
ax1.imshow(cv2.cvtColor(data[2]['rgb'], cv2.COLOR_RGB2BGR))
ax2.imshow(cv2.cvtColor(data[2]['lbl'], cv2.COLOR_RGB2BGR))

# %%
np.savez('/home/zoltan/Documents/isprs_potsdam.npz', data=data)

# %%
loaded_data = np.load('/home/zoltan/Documents/isprs_potsdam.npz', allow_pickle=True)

# %%
