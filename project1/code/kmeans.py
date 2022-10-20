import scipy.cluster.vq
import sklearn
from sklearn.cluster import KMeans
from tqdm import tqdm as tqdm
import matplotlib.pyplot as plt

import cv2
import numpy as np


from preprocess import RELEVANT_CLASSES, IMAGE_SHAPE, N_CLASSES, classes
from store import load, save

sift = cv2.xfeatures2d.SIFT_create()

def select_subsets(images_ordered, labels_ordered, percentage1, percentage2):
    """
    percentage1  Part that is used to build visual vocabulary
    percentage2  Part that is used to calcualte visual dictionary (note the minus sign in the last line of select_subsets)
    """
    assert len(images_ordered.shape) == 5
    N_images = np.prod( images_ordered.shape[:2] )

    n1 = int(N_images * percentage1/100 / N_CLASSES)
    n2 = int(N_images * percentage2/100 / N_CLASSES)

    images_subset1 = images_ordered[:,:n1]
    images_subset1 = images_subset1.reshape(-1, *IMAGE_SHAPE)
    images_subset2 = images_ordered[:,-n2:]
    
    labels_subset1 = labels_ordered[:,:n1]
    labels_subset1 = labels_subset1.reshape(-1)
    labels_subset2 = labels_ordered[:,-n2:]
    
    return images_subset1, labels_subset1, images_subset2, labels_subset2



def kmeans_model(features, vocab_size, batched = True):
    """
    Initalizes and fits the features with kmeans     

    features     The features to fit
    vocab_size   Should be set to 10000
    """
    if batched:
        kmeans = sklearn.cluster.MiniBatchKMeans(n_clusters = vocab_size) # This line is giving warnings 
    else:
        kmeans = sklearn.cluster.KMeans(n_clusters = vocab_size)

    kmeans.fit(features)
    model = kmeans

    return model

def calc_class_features(imgs, disable_tqdm = False):
    """
    Calculates and reshapes the descriptors (features) of given images
    """
    features = np.array([])
    for img in tqdm(imgs, disable = disable_tqdm, desc = 'Calculating img kps'):
        kp, des = sift.detectAndCompute(img, None)
        if des is not None:
            features = np.append( features, des )
    features = np.reshape(features, (len(features)//128, 128))
    return features

def predict_class_features(features_list, model):
    """
    Calculate the predictions of the classed features with the LEARNED model
    
    features_list     A list where each item are features of a single class
    model             The learned model
    """

    preds = []
    for features in tqdm(features_list, desc = 'Predicting features from kmeans model'):
        pred = model.predict(features)
        preds.append(pred)
        
    return preds

def plot_class_features(percentages, vocab_size):
    rows = len(percentages)

    fig, axs = plt.subplots(rows, N_CLASSES, 
                            figsize = (15, 3*rows),
                            tight_layout = True
                           )
    for axs_row, percentage1 in tqdm(zip(axs, percentages), total = rows):
        percentage2 = 100 - percentage1

        preds = load('class_features', percentage1, percentage2)

        for ax, pred, class_name in zip(axs_row, preds, classes):
            ax.hist(pred, bins = np.arange(vocab_size), density = True)
            ax.set_ylim(0, 0.007)

            # Labels
            if all(axs_row == axs[0]): ax.set_title(class_name, fontsize = 20)
            if all(axs_row != axs[-1]): 
                ax.xaxis.set_ticklabels([])
            if ax != axs_row[0]:
                ax.yaxis.set_ticklabels([])

        axs_row[0].set_ylabel(f'{percentage1}%', fontsize = 20)
    return fig