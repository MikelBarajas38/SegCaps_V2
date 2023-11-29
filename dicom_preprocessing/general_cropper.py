# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 10:09:22 2023

@author: Eduardo
"""

from pydicom import dcmread
import matplotlib.pyplot as plt
import os
import numpy as np

def recortar_imagen_umbral_LO(img, umbral):
    
    # Obtener coordenadas de valores que no son 'negros'
    coord = np.transpose(np.where(img > umbral))

    # Columnas
    col_ini = np.min(coord[:, 1])
    col_fin = np.max(coord[:, 1])

    # Renglones
    N = round(img.shape[0] / 2)
    margen = 800
    img_copia = np.where(img > umbral, 1, 0)
    
    suma = np.sum(img_copia[0:N-margen,:], axis=1, dtype=None, out=None, keepdims=False)
    ren_ini = np.argmin(suma)
    
    suma = np.sum(img_copia[N+margen:,:], axis=1, dtype=None, out=None, keepdims=False)
    ren_fin = np.argmin(suma)
    ren_fin = ren_fin + N + margen
    
    img_recortada = img[ren_ini:ren_fin+1:, col_ini:col_fin+1]
    
    return img_recortada

def recortar_imagen_umbral(img, umbral):

    # Obtener coordenadas de valores que no son negras
    coord = np.transpose(np.where(img > umbral))

    # Columnas
    col_ini = np.min(coord[:, 1])
    col_fin = np.max(coord[:, 1])

    # Renglones
    ren_ini = np.min(coord[:, 0])
    ren_fin = np.max(coord[:, 0])
        
    img_recortada = img[ren_ini:ren_fin+1,col_ini:col_fin+1]

    return img_recortada

def square_img(img):

    height, width = img.shape
    
    min_axis = width
    diff = height - width

    avg_top = np.mean(img[diff:, :], axis=(0, 1))
    avg_bottom = np.mean(img[:-diff, :], axis=(0, 1))

    if avg_top < avg_bottom:
        img = img[diff:, :]
    else:
        img = img[:-diff, :]

    return img
