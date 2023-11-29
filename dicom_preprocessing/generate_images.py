# @author: mikel
# reference: https://research.edgehill.ac.uk/ws/portalfiles/portal/20123898/IEEETBME2016.pdf

import cv2
import numpy as np
from scipy.signal import find_peaks, peak_prominences

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from Side import get_side, Side

def is_valid(img, x, y):
    N, M = img.shape
    return x >= 0 and x < M and y >= 0 and y < N 

def get_distance(a, b):
    return abs(a - b)

def region_growing_avg(img, mask, tolerance, seed):
    y = seed[0]
    x = seed[1]
    
    mask[y, x] = 1
    mean = np.mean(img[np.where(mask == 1)])
    
    plt.title('mask')
    plt.imshow(mask)
    plt.show()
    
    if is_valid(img, x+1, y) and mask[y, x+1] == 0 and get_distance(mean, img[y, x+1] <= tolerance):
        region_growing_avg(img, mask, tolerance, [y, x+1])
    
    if is_valid(img, x-1, y) and mask[y, x-1] == 0 and get_distance(mean, img[y, x-1] <= tolerance):
        region_growing_avg(img, mask, tolerance, [y, x-1])
        
    if is_valid(img, x, y+1) and mask[y+1, x] == 0 and get_distance(mean, img[y+1, x] <= tolerance):
        region_growing_avg(img, mask, tolerance, [y+1, x])
        
    if is_valid(img, x, y-1) and mask[y-1, x] == 0 and get_distance(mean, img[y-1, x] <= tolerance):
        region_growing_avg(img, mask, tolerance, [y-1, x])


def is_muscle_start(img, top_point):
    row = get_mean_ntok_rows(img, 0, 20)
    x = round(top_point)
    left = np.mean(row[x-100:x])
    right = np.mean(row[x:x+100])

    if get_side(img) == Side.LEFT:
        print(f'diff = {left - right}')
        return left - right > 200
    else:
        print(f'diff = {right - left}')
        return right - left > 200
#----------------------------------------------------------------------------------------------------------

def apply_simple_tresholding_mask(img):
    UMBRAL = 100
    norm_img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    _, binary_mask = cv2.threshold(norm_img, UMBRAL, norm_img.max(), cv2.THRESH_BINARY)
    return binary_mask

def apply_clahe(img):
    """Apply CLAHE filter using GPU"""

    clahe = cv2.createCLAHE(clipLimit = 40)  # crete clahe parameters

    img_umat = cv2.UMat(img)  # send img to gpu

    img_umat = clahe.apply(img_umat)

    # Normalize image [0, 255]
    img_umat = cv2.normalize(img_umat, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return img_umat.get()  # recover img from gpu

def get_max_inscribed_circle(img):
    mask = apply_simple_tresholding_mask(img)
    
    #make borders black
    mask[0] = 0
    mask[::,0] = 0
    mask[-1] = 0
    mask[::, -1] = 0
    
    #adjustment
    if img.shape[1] < 1800:
        for i in range(500):
            mask[i] = 0

    dist_map = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    _, radius, _, center = cv2.minMaxLoc(dist_map)
    
    return center, round(radius)

def get_mean_ntok_rows(img, n, k):
    img = img[n:k, :]
    mean_values = img.mean(axis=0)
    return mean_values

def get_nth_greatest_val(arr, n):
    sorted_arr = arr.copy()
    sorted_arr.sort()
    return sorted_arr[-n]

def get_n_most_prominent_mins(mins, prominence, n):
    prominence_val = get_nth_greatest_val(prominence, 2)
    return mins[prominence >= prominence_val]

def get_muscle_starting_point(clahe_img, k):
    M,N = clahe_img.shape
    row = get_mean_ntok_rows(clahe_img, 0, k)
    x = np.linspace(0, N, num=N)
    
    mins, _ =find_peaks(-row)
    prominence = peak_prominences(-row, mins)[0]
    most_prominent_mins = get_n_most_prominent_mins(mins, prominence, n=1)
    
    '''
    plt.plot(x[most_prominent_mins], row[most_prominent_mins], '*')
    plt.plot(x, row)
    
    plt.xlabel('pixel')
    plt.ylabel('valor')
    plt.title(f'mean of the first {k} rows')
    plt.show()
    '''
    
    if get_side(clahe_img) is Side.RIGHT:
        return x[most_prominent_mins[0]], 0
    else:
        return x[most_prominent_mins[1]], 0

def get_muscle_ending_point(img):
    #find MIC
    center, radius = get_max_inscribed_circle(img)
    if get_side(img) is Side.RIGHT:
        return center[0] + radius, center[1]
    else:
        return center[0] - radius, center[1]
    

def remove_pectoral_muscle(img):
    
    #get CLAHE
    clahe_img = apply_clahe(img)
    
    #blur to prevent false-positive 
    clahe_img = cv2.GaussianBlur(clahe_img, (25, 25), 0)
    
    top_point = get_muscle_starting_point(clahe_img, k=20)
    bottom_point = get_muscle_ending_point(img)
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(clahe_img, cmap='gray')
    ax1.axis("off")
    ax2.axis("off")
    ax2.imshow(clahe_img, cmap='gray')
    ax2.plot(top_point[0], 50, '*', markersize = 10)
    
    center, radius = get_max_inscribed_circle(img)
    circle = Circle((center[0], center[1]), radius, edgecolor='white', facecolor='none', linestyle='dashed')
    ax2.add_patch(circle)
    
    ax2.plot(center[0] + radius, center[1], 'x', markersize = 10 )
    ax2.plot(center[0] - radius, center[1], 'x', markersize = 10 )
    
    T = top_point[0] / 4 if get_side(img) is Side.LEFT else (top_point[0] - img.shape[1]) / 4
    ax2.plot([top_point[0], bottom_point[0]], [top_point[1], bottom_point[1]], linestyle='dashed')
    ax2.plot([top_point[0] + T , bottom_point[0] + T], [top_point[1], bottom_point[1]], linestyle='dashed')
    ax2.plot([top_point[0] - T , bottom_point[0] - T], [top_point[1], bottom_point[1]], linestyle='dashed')

    plt.show()
    
    slope = (bottom_point[1] - top_point[1]) /  (bottom_point[0] - top_point[0])
    
    '''
    if get_side(img) is Side.RIGHT:
        y_intercept = -(slope*bottom_point[0]) + bottom_point[1]
        for x in range(M - round(top_point[0])):
            inv_x = M - x - 1
            line_y = round(slope * inv_x + y_intercept)
            img[:line_y, inv_x] = 0
    else:
        y_intercept = bottom_point[1]
        for x in range(round(top_point[0])):
            line_y = round(slope * x + y_intercept)
            img[:line_y,x] = 0
    
    '''
    
    if get_side(img) is Side.RIGHT:
        y_intercept = -(slope*bottom_point[0]) + bottom_point[1]
        for x in range(img.shape[1]):
            line_y = slope * x + y_intercept
            for y in range(img.shape[0]):                
                # Crop the pixel if it lies to the right of the line
                if y < line_y:
                    img[y, x] = 0  # Set pixel to black color
    else:
        y_intercept = bottom_point[1]
        for x in range(img.shape[1]):
            line_y = slope * x + y_intercept
            for y in range(img.shape[0]):
                # Crop the pixel if it lies to the right of the line
                if y < line_y:
                    img[y, x] = 0  # Set pixel to black color
    
    '''
    mask = img.copy()
    if get_side(img) is Side.RIGHT:
        y_intercept = -(slope*bottom_point[0]) + bottom_point[1]
        for x in range(img.shape[1]):
            line_y_1 = slope * (x + T) + y_intercept
            line_y_2 = slope * (x - T) + y_intercept
            for y in range(img.shape[0]):                
                # Crop the pixel if it lies to the right of the line
                if y > line_y_1 and y < line_y_2:
                    mask[y, x] = 0  # Set pixel to black color
                else:
                    mask[y, x] = -1
                    
    else:
        y_intercept = bottom_point[1]
        for x in range(img.shape[1]):
            line_y_1 = slope * (x + T) + y_intercept
            line_y_2 = slope * (x - T) + y_intercept                
            for y in range(img.shape[0]):                
                # Crop the pixel if it lies to the right of the line
                if y > line_y_1 and y < line_y_2:
                    mask[y, x] = 0  # Set pixel to black color
                else:
                    mask[y, x] = -1
    
    plt.title('mask')
    plt.imshow(mask, cmap='gray')
    plt.show()
    
    if is_muscle_start(img, top_point[0]):
        region_growing_avg(img, mask, 10, [round(top_point[1]), round(top_point[0])])
     
    
    mask[mask == -1] = 0
    
    plt.title('mask 2')
    plt.imshow(mask, cmap='gray')
    plt.show()
    '''
    
    return img

def generate_pectoral_muscle_mask(img):
    
    #get CLAHE
    clahe_img = apply_clahe(img)
    
    #blur to prevent false-positive 
    clahe_img = cv2.GaussianBlur(clahe_img, (25, 25), 0)
    
    #Get POIs and the offset (to include all of the muscle)
    top_point = get_muscle_starting_point(clahe_img, k=20)
    bottom_point = get_muscle_ending_point(img)
    T = top_point[0] / 4 if get_side(img) is Side.LEFT else (top_point[0] - img.shape[1]) / 4

    '''
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(clahe_img, cmap='gray')
    ax1.axis("off")
    ax2.axis("off")
    ax2.imshow(clahe_img, cmap='gray')
    ax2.plot(top_point[0], 50, '*', markersize = 10)
    
    center, radius = get_max_inscribed_circle(img)
    circle = Circle((center[0], center[1]), radius, edgecolor='white', facecolor='none', linestyle='dashed')
    ax2.add_patch(circle)
    
    ax2.plot(center[0] + radius, center[1], 'x', markersize = 10 )
    ax2.plot(center[0] - radius, center[1], 'x', markersize = 10 )
    
    ax2.plot([top_point[0], bottom_point[0]], [top_point[1], bottom_point[1]], linestyle='dashed')
    ax2.plot([top_point[0] + T , bottom_point[0] + T], [top_point[1], bottom_point[1]], linestyle='dashed')
    ax2.plot([top_point[0] - T , bottom_point[0] - T], [top_point[1], bottom_point[1]], linestyle='dashed')

    plt.show()
    '''
    
    slope = (bottom_point[1] - (top_point[1])) /  (bottom_point[0] - top_point[0])
    
    mask = img.copy()

    if get_side(img) is Side.RIGHT:
        y_intercept = -(slope*bottom_point[0]) + bottom_point[1] - T
        for x in range(img.shape[1]):
            line_y = slope * x + y_intercept
            for y in range(img.shape[0]):                
                if y >= line_y:
                    mask[y, x] = 0  # Set pixel to black color

    else:
        y_intercept = bottom_point[1] + T
        for x in range(img.shape[1]):
            line_y = slope * x + y_intercept
            for y in range(img.shape[0]):
                if y >= line_y:
                    mask[y, x] = 0  # Set pixel to black color
    
    return mask

