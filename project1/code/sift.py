import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from preprocess import classes

sift = cv2.xfeatures2d.SIFT_create()

def sift_keypoints(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    keypoints, _ = sift.detectAndCompute(img, None)

    img_kp = cv2.drawKeypoints(img, keypoints, 
                               outImage = np.array([]), # I don't know why this should be here
                               color = (0, 0, 255), # Draw blue images
                               flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS # Not sure about
                              )
    return img_kp


def get_patch_centers(img, step_size):
    """
    Given an input image, devide this image into patches with size stepsize 
    and calculates the patch centers and returns the coordinates of the centers.

    input:
    img: an image
    step_size: integer

    Returns
    center_coordinates a list with cv2.KeyPoints() of all the patch centers.
    """
    # tiles = [img[x:x+M,y:y+N,:] for x in range(0,img.shape[0],M) for y in range(0,img.shape[1],N)]
    # kp = [cv2.KeyPoint(x, y, M) for y in range(0, img.shape[0], M) for x in range(0, img.shape[1], M)]
    center_coordinates = [cv2.KeyPoint(x+(step_size//2), y+(step_size//2), step_size) for x in range(0,img.shape[0],step_size) for y in range(0,img.shape[1],step_size)]
    return center_coordinates

def sift_dense(img):
    kp = get_patch_centers(img, 8)
    img_kp = cv2.drawKeypoints(img, kp, img, color = (0, 0, 255)) # Draw blue images
    return img_kp

def sift_plot(img_kps, img_dens):
    s = 3
    fig = plt.figure(figsize=(s*5, s*4))
    axs = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(4, 5),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )

    # for ax, im in zip(grid, image_data):
    #     # Iterating over the grid returns the Axes.
    #     ax.imshow(im)

    for ax, img in zip(axs, np.concatenate((img_kps, img_dens))):
        ax.imshow(img)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])

    for ax, class_name in zip(axs[:5], classes):
        ax.set_title(class_name)

    ax = axs[5]
    ax.set_ylabel('Key points', fontsize = 15)
    ax.yaxis.set_label_coords(-0.1, 1.)

    ax = axs[15]
    ax.set_ylabel('Densely sampled regions', fontsize = 15)
    ax.yaxis.set_label_coords(-0.1, 1.)

    return fig