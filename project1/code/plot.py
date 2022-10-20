import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from preprocess import IMAGE_SHAPE, N_CLASSES, classes

def plot_confusion_matrix(preds_binary, labels_binary):

    fig, axs = plt.subplots(1, N_CLASSES, figsize = (15, 3), sharey = True, sharex = True)
    bins_c = np.arange(-0.5, 2.5)

    for index, (ax, pred_binary, label_binary) in enumerate(zip(axs, preds_binary, labels_binary)):

        hist, _, _, im = ax.hist2d(pred_binary, label_binary, bins = bins_c)

        for i in range(len(bins_c)-1):
            for j in range(len(bins_c)-1):
                ax.text(bins_c[j]+0.5,bins_c[i]+0.5, hist.T[i,j], 
                        color="w", ha="center", va="center", fontweight="bold")

        ax.set_title(classes[index])
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        
    axs[0].set_ylabel('True value', fontsize = 12)
    axs[N_CLASSES // 2].set_xlabel('Predicted value', fontsize = 12)
    return fig

def plot_pred_distribution(preds_ranked, labels_ranked_binary):

    fig, axs = plt.subplots(1, N_CLASSES, figsize = (15, 3), sharey = True, sharex = True)

    for index, (ax, pred_ranked, label_ranked_binary) in enumerate(zip(axs, preds_ranked, labels_ranked_binary)):
        
        ax.scatter(x = np.arange(len(pred_ranked)),
                   y = pred_ranked,
                   c = label_ranked_binary,
                   alpha = 0.1
                  )

        ax.set_title(classes[index])
        ax.plot([0, len(pred_ranked)], [0, 0], 'k:')
        
    axs[0].set_ylabel('Relative cerntainty', fontsize = 12)
    axs[N_CLASSES // 2].set_xlabel('Image index', fontsize = 12)
    return fig    


def plot_and_save_dots():
    fig, ax = plt.subplots(1, 1, figsize = (1, 1))
    ax.text(0.5, .5, '...', horizontalalignment='center',
         verticalalignment='center', transform=ax.transAxes, fontweight = 'bold', fontsize = 20)
    ax.axis('off')
    ax.set_aspect('equal')
    fig.savefig('dots.png', dpi = 96, pad_inches=0.1)
    plt.close(fig)

    
def plot_top_images(images, preds_indices):
    s = 2
    fig = plt.figure(figsize=(s*10, s*5))
    axs = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(5, 11),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )


    imgs = np.array([[
        images[preds_indices[index, -5:]], # best imgs
        images[np.flip(preds_indices[index, :5])] # worst imgs
    ] for index in range(N_CLASSES)]).reshape(-1, *IMAGE_SHAPE)

    center_indices = np.arange(5, 55, 11)
    axs_sel = np.delete(axs, center_indices)

    for ax, img in zip(axs_sel, imgs):
        ax.imshow(img)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set(frame_on=False)  # New

    img_dots = plt.imread('dots.png')#[:96, :96]
    for ax in np.take(axs, np.arange(5, 55, 11)):
        ax.imshow(img_dots)
        ax.axis('off')


    axs[2].set_title('Top 5', fontweight = 'bold', fontsize = 20)
    axs[8].set_title('Bottom 5', fontweight = 'bold', fontsize = 20)

    for i, class_name in zip(np.arange(0, 55, 11), classes):
        axs[i].set_ylabel(class_name)
    
    return fig