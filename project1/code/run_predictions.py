from preprocess import preprocess, IMAGE_SHAPE
from store import load, save

from kmeans import select_subsets, kmeans_model, calc_class_features 
from svm import predict_labels


def run_predictions(percentage1, vocab_size):
    percentage2 = 100 - percentage1
    
     # Select subsets
    images_train_subset1, labels_train_subset1, images_train_subset2, labels_train_subset2 = select_subsets(images_train_ordered, labels_train_ordered, percentage1, percentage2)

    images_train_subset2 = images_train_subset2.reshape(-1, *IMAGE_SHAPE)
    labels_train_subset2 = labels_train_subset2.reshape(-1)

    # Calculate features for subsets
    features1 = calc_class_features(images_train_subset1)

    # Calculate and predict
    model = kmeans_model(features = features1, vocab_size = vocab_size)

    # Save
    save(model, f'kmeans_model_{vocab_size}', percentage1, percentage2)

    predict_labels(
        percentage1, percentage2, vocab_size,
        model,
        images_train_subset2, labels_train_subset2,
        images_test, labels_test
    )

if __name__ =='__main__':
    n_train, n_test = None, None #50, 100
    
#     percentages = [30, 40, 50, 60]
    percentages = [40, 50, 60]
    
    vocab_sizes = [500, 1000, 1500, 2000]
#     vocab_sizes = [vocab_sizes[1]]
    
    images_train_ordered, labels_train_ordered, images_test, labels_test = preprocess(n_train, n_test)
    
    for percentage1 in percentages:
        for vocab_size in vocab_sizes:
            print(f'\n\n\t##### percentage1 = {percentage1}, vocab_size = {vocab_size} #####\n\n')
            run_predictions(percentage1, vocab_size)
        
    