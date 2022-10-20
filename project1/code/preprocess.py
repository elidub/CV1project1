import numpy as np

from stl10_input import DATA_DIR, HEIGHT, WIDTH, DEPTH, \
                        DATA_PATH_TRAIN, LABEL_PATH_TRAIN, DATA_PATH_TEST, LABEL_PATH_TEST, \
                        read_all_images, read_labels, keep_relevant_images

RELEVANT_CLASSES = np.array([1, 2, 9, 7, 3])
classes = ['Airplane', 'Bird', 'Ship', 'Horse', 'Car']
IMAGE_SHAPE = (HEIGHT, WIDTH, DEPTH)
N_CLASSES = len(RELEVANT_CLASSES)


### Preprocess images and labels

def import_images(data_path, label_path):
    images = read_all_images(data_path)

    labels = read_labels(label_path)
    used_labels, used_images = keep_relevant_images(images, labels, RELEVANT_CLASSES)
    return used_labels, used_images

def order_images(images, labels):
    N_labels = len(labels)
    N = int(N_labels / N_CLASSES)
    
    images_array = np.zeros((len(RELEVANT_CLASSES), N, *IMAGE_SHAPE), dtype = images.dtype)
    for i, class_index in enumerate(RELEVANT_CLASSES):
        image_indices = np.where(labels == class_index)[0].reshape(1, -1)
        images_array[i] = images[tuple(image_indices)] # Select the images from the indices
        
    labels_array = np.array([np.full(shape = (N), fill_value = i) for i in range(N_CLASSES)])

    return images_array, labels_array

def prepare_labels(labels):
    
    for i, class_index in enumerate(RELEVANT_CLASSES):
        labels[labels == class_index] = i + 10 # First add 10 so new and old indices don't get mixed
    labels -= 10 # Subtract 10 after for loop
    
    return labels

def preprocess(n_train = None, n_test = None):
    images_train, labels_train = import_images(DATA_PATH_TRAIN, LABEL_PATH_TRAIN)
    images_test, labels_test = import_images(DATA_PATH_TEST, LABEL_PATH_TEST)
    images_train_ordered, labels_train_ordered = order_images(images_train, labels_train)
    labels_test = prepare_labels(labels_test)
    
    if (n_train is not None) and (n_test is not None):
        images_train_ordered = images_train_ordered[:,:n_train]
        labels_train_ordered = labels_train_ordered[:,:n_train]
        images_test = images_test[:n_test]
        labels_test = labels_test[:n_test]
    
    return images_train_ordered, labels_train_ordered, images_test, labels_test
