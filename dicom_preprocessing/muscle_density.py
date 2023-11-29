#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 11:19:45 2023

@author: mikel
"""

import cv2

def apply_clahe(img):
    """Apply CLAHE filter using GPU"""

    clahe = cv2.createCLAHE(clipLimit = 100)  # crete clahe parameters

    img_umat = cv2.UMat(img)  # send img to gpu

    img_umat = clahe.apply(img_umat)

    # Normalize image [0, 255]
    img_umat = cv2.normalize(img_umat, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return img_umat.get()  # recover img from gpu


def apply_otsu(img):
    clahe = apply_clahe(img)
    _, binary_mask = cv2.threshold(clahe, 0, clahe.max(), cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # binary_mask = ~binary_mask
    return binary_mask