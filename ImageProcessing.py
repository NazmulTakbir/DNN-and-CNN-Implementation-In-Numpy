import numpy as np
import cv2

def contrast_grey(img):
    high = np.percentile(img, 90)
    low  = np.percentile(img, 10)
    if high == low:
        high = np.percentile(img, 95)
        low  = np.percentile(img, 5)
    return (high-low)/(high+low), high, low

def adjust_contrast_grey(img, target = 0.4):
    contrast, high, low = contrast_grey(img)
    if high == low:
        return img, False
    if contrast < target:
        img = img.astype(int)
        ratio = 200./(high-low)
        img = (img - low + 25)*ratio
        img = np.maximum(np.full(img.shape, 0) ,np.minimum(np.full(img.shape, 255), img)).astype(np.uint8)
    return img, True

def black_and_white(img, thres):
    img[img < thres] = 0
    img[img >= thres] = 255
    return img

def crop(img, thres=253, margin=5):
    row_means = np.mean(img, axis=1)
    col_means = np.mean(img, axis=0)
    firstcol = lastcol = firstrow = lastrow = 0

    for i in range(len(row_means)):
        if row_means[i] < thres:
            firstrow = max(i-margin, 0)
            break

    for i in range(len(row_means)-1, 0, -1):
        if row_means[i] < thres:
            lastrow = min(i+margin, img.shape[0])
            break

    for i in range(len(col_means)):
        if col_means[i] < thres:
            firstcol = max(i-margin, 0)
            break

    for i in range(len(col_means)-1, 0, -1):
        if col_means[i] < thres:
            lastcol = min(i+margin, img.shape[1])
            break

    croppedimg = img[firstrow:lastrow, firstcol:lastcol]
    return croppedimg

def crop_and_contrast(img_path, img_size):
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = np.array(img)
        if np.mean(img) < 128:
            img = 255 - img 
        img, success = adjust_contrast_grey(img)
        if not success:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img.shape != (img_size, img_size):
                img = cv2.resize(img, (img_size, img_size))
            img = np.array(img)
            return img
        else:
            img = black_and_white(img, 128)
            img = crop(img) 
            # img = np.asnumpy(img)
            img = cv2.resize(img, (img_size, img_size))
            img = np.array(img)
            return img
    except Exception as e: 
        print(img_path, e)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img.shape != (img_size, img_size):
            img = cv2.resize(img, (img_size, img_size))
        img = np.array(img)
        return img