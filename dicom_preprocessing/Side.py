from enum import Enum

Side = Enum('Side', ['RIGHT', 'LEFT'])

def get_side(img):
    N = round(img.shape[1] / 2)
    img_left = img[:,0:N]
    img_right = img[:,N:]
    
    if img_left.mean() > img_right.mean():
        return Side.LEFT
    else:
        return Side.RIGHT   