import numpy as np
import matplotlib.pyplot as plt

au_match = [12, 17, 20, 26, 4, 6, 15, 1, 25, 2, 5, 9]
au_description = ["Lip Corner Puller", # AU12
                  "Chin Raiser", # AU17
                  "Lip Stretcher",  # AU20 
                  "Jaw Drop", # AU26
                  "Brow Lowerer", # AU4
                  "Cheek Raiser", # AU6
                  "Lip Corner Depressor", # AU15
                  "Inner Brow Raiser", # AU1
                  "Lips Parts", # AU25
                  "Outer Brow Raiser", # AU2
                  "Upper Lid Raiser", # AU5
                  "Nose Wrinkler" # AU9
                  ]

def ROC(log, au, au_projection, save_file):
    specificity = np.array(log['specificity'][au])
    sensitivity = np.array(log['sensitivity'][au])
    x = np.linspace(start=0, stop=1, num=100)
    plt.figure()
    plt.plot(1 - specificity, sensitivity, label='trained classifier')
    plt.plot(x, x, label='random classifier')
    plt.title('ROC for AU{} : {}'.format(au_match[au_projection[au]], au_description[au_projection[au]]))
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.legend()
    plt.savefig(save_file)

def SOTA_order(order, result_array, au_projection):
    result = []
    for au in order:
        found = False
        i = 0
        while not found:
            if result_array.shape[0] == 8:
                found = (au_match[au_projection[i]] == au)
            if result_array.shape[0] == 12:
                found = (au_match[i] == au)
            i+=1
        found_i = i-1
        result.append(result_array[found_i])
    return result

def trunc(x, decimals):
    x_str = str(x).split('.')
    if len(x_str) == 1:
        return x_str[0]
    return x_str[0] + '.' + x_str[1][:decimals]

def F1Score(log, au, au_projection):
    f1_score = np.max(np.array(log['f1score'][au]))
    print('AU{}: {}%'.format(au_match[au_projection[au]], trunc(f1_score * 100, 1)))
    return f1_score

def AUC(log, au, au_projection):
    auc = log['AUC'][au]
    print('AU{}: {}'.format(au_match[au_projection[au]], trunc(auc, 3)))
    return auc

def SOTA_order(order, result_array, au_projection):
    result = []
    for au in order:
        found = False
        i = 0
        while not found:
            found = (au_match[au_projection[i]] == au)
            i += 1
        found_i = i-1
        result.append(result_array[found_i])
    return result

def metrics(logs, au):
    n_thresholds = len(logs[0]['f1score'][au])
    f1_folds = np.zeros((len(logs), n_thresholds))
    auc_folds = np.zeros((len(logs),))
    for i in range(len(logs)):
        f1_folds[i] = logs[i]['f1score'][au]
        auc_folds[i] = logs[i]['AUC'][au]

    return np.array([np.mean(np.max(f1_folds, axis=1)), np.mean(auc_folds)])
