#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 10:42:32 2023

@author: mikel
"""
import os
from pydicom import dcmread
from pydicom.encaps import encapsulate
import matplotlib.pyplot as plt
from skimage.transform import resize
import numpy as np
from PIL import Image
import cv2

from generate_images import remove_pectoral_muscle, generate_pectoral_muscle_mask, apply_clahe
from general_cropper import square_img
from muscle_density import apply_otsu

current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)

input_path = f'{current_directory}/data/dicom'
output_path = f'{current_directory}/data/np_files'

imgs_path = f'{current_directory}/data/imgs'
masks_path = f'{current_directory}/data/masks'

# images = ['2.16.840.1.113669.632.25.1.103024.20200330123457658.3','2.16.840.1.113669.632.25.1.103024.20200330123624691.3', '2.16.840.1.113669.632.25.1.103024.20200401134008460.3', '2.16.840.1.113669.632.25.1.103024.20200401134232052.3']
# images = ['2.16.840.1.113669.632.25.1.103024.20200401134232052.3']

for file in os.listdir(input_path):
    print(file)
    
    var_metadato = dcmread(f'{input_path}/{file}')
    
    m_type = var_metadato[0x0054,0x0220].value[0]
    
    if m_type.CodeValue == 'R-10242':
        print('bad')
        continue
    
    img = var_metadato.pixel_array
    
    img = square_img(img)

    mask = generate_pectoral_muscle_mask(img)
    
    otsu_img = apply_otsu(mask)

    img = apply_clahe(img)

    resized_image = resize(img, (512, 512), anti_aliasing=True)
    resized_mask = resize(otsu_img, (512, 512), anti_aliasing=True)

    resized_image = cv2.normalize(resized_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    resized_mask= cv2.normalize(resized_mask, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # resized_image = np.stack([resized_image] * 3, axis=-1)
    # resized_mask = np.stack([resized_mask] * 3, axis=-1)

    plt.imshow(resized_image)
    plt.show()

    plt.imshow(resized_mask, cmap='gray')
    plt.show()

    print("aceptar?")
    op = input()

    if(op == 'N' or op == 'n'):
        continue

    im = Image.fromarray(resized_image)
    im.save(f'{imgs_path}/{file}.png', "PNG")

    im = Image.fromarray(resized_mask)
    im.save(f'{masks_path}/{file}.png', "PNG")

    img = resized_image.reshape([resized_image.shape[0], resized_image.shape[1], 1])
    mask = resized_mask.reshape([resized_mask.shape[0], resized_mask.shape[1], 1])

    # img = np.rollaxis(img, 0, 3)
    # mask = np.rollaxis(mask, 0, 3)

    # mask[mask > np.mean(mask)] = 1
    
    plt.imshow(mask)
    plt.show()

    np.savez_compressed(f'{output_path}/{file}.npz', img=img, mask=mask)

    # os.system(f'cp {input_path}/{file} {imgs_path}/{file}')  
    # os.system(f'cp {input_path}/{file} {masks_path}/{file}')  





