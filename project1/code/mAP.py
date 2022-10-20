import numpy as np
from preprocess import N_CLASSES

def fc(preds, labels):
    """
    Generator to check if a prediction is correct.

    parameters:

    pred_label: prediction label
    true_label: the true label
    
    Output:
    list in the form of 
    """
    
    i = 1
    out_num = 0 
    for j, (p,t) in enumerate(zip(preds, labels)):
        if p == t == 1:
            out_num += (i/(j+1))
            i += 1
    return(out_num)

def calculate_AP(pred_ranked_binary, label_ranked_binary):
    mc = label_ranked_binary.sum()
#     print(mc)
    AP = 1/mc * fc(pred_ranked_binary, label_ranked_binary);
    return AP
    
def calculate_mAP(preds_ranked_binary, labels_ranked_binary):
    APs = np.zeros(N_CLASSES)
    for index, pred_ranked_binary, label_ranked_binary in zip(range(N_CLASSES), preds_ranked_binary, labels_ranked_binary):
        AP = calculate_AP(pred_ranked_binary, label_ranked_binary)   
        APs[index] = AP
        
#     assert all(APs > 0), print(APs)
#     print('APs = ', APs)
    mAP = np.mean(APs)
    return mAP