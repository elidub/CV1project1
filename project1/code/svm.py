import numpy as np
import sklearn
from tqdm import tqdm as tqdm

from preprocess import RELEVANT_CLASSES, IMAGE_SHAPE, N_CLASSES, classes
from store import load, save
from kmeans import calc_class_features

def calc_hist(img, model, bins):
    
    assert img.shape == IMAGE_SHAPE
    
    img = img.reshape(1, *IMAGE_SHAPE)
    
    features = calc_class_features(img, disable_tqdm = True)
    if len(features) > 0: # If statement in case there are no features
        pred = model.predict(features) 
        hist, _ = np.histogram(pred, bins = bins, density = True)
    else:
        hist = np.zeros(len(bins)-1)
        print('No features found in img!')
    return hist

def labels_binary(labels, index):
    return (labels == index)*1.

def get_classifiers(hists, labels):
    fits = []
    for index in tqdm(range(N_CLASSES), desc = 'Fitting SVMs'):
        clf = sklearn.svm.SVC()
        fit = clf.fit(hists, labels_binary(labels, index))
        fits.append(fit)
        
    return fits

def rank(preds, labels):
    preds_indices = np.argsort(preds, axis = 1)
    
    preds_ranked = np.zeros_like(preds)
    preds_binary = np.zeros_like(preds)
    
    labels_ranked       = np.zeros_like(preds)
    labels_ranked_binary= np.zeros_like(preds)
    
    for index in range(N_CLASSES):
        labels_ranked[index]        = np.flip(labels[preds_indices[index]])
        labels_ranked_binary[index] = labels_binary(labels_ranked[index], index)
        
        preds_ranked[index] = np.flip(preds[index][preds_indices[index]])
        preds_binary[index] = np.array([0 if d < 0 else 1 for d in preds_ranked[index]])
    return preds_ranked, preds_binary, preds_indices, labels_ranked, labels_ranked_binary


def predict_labels(percentage1, percentage2, vocab_size, model, images_train, labels_train, images_test, labels_test):
    bins = np.arange(vocab_size)
    
    hist_test_filename = f'hists_test_{vocab_size}'
    hist_train_filename = f'hists_train_{vocab_size}'
    classifers_filename = f'classifiers_{vocab_size}'
    preds_train_all_filename = f'preds_train_all_{vocab_size}'
    preds_test_all_filename = f'preds_test_all_{vocab_size}'
    
    # Getting histograms and classifiers
    hists_train = np.array([calc_hist(img, model, bins) for img in 
                            tqdm(images_train, desc = 'Calculating train hists')])
    hists_test = np.array([calc_hist(img, model, bins) for img in 
                           tqdm(images_test, desc = 'Calculating test hists')])
    classifiers = get_classifiers(hists_train, labels_train)
    
    # Predictions of labels
    preds_train = np.array([classifier.decision_function(hists_train) for classifier in 
                            tqdm(classifiers, desc = 'Classifying train set')])
    preds_test = np.array([classifier.decision_function(hists_test) for classifier in 
                           tqdm(classifiers, desc = 'Classifying test set')])
    
    
    preds_train_all = rank(preds_train, labels_train)
    preds_test_all = rank(preds_test, labels_test)
    
    save(hists_test, hist_test_filename, percentage1, percentage2)
    save(hists_train, hist_train_filename, percentage1, percentage2)
    save(classifiers, classifers_filename, percentage1, percentage2)
    save(preds_train_all, preds_train_all_filename, percentage1, percentage2)
    save(preds_test_all, preds_test_all_filename, percentage1, percentage2)
    