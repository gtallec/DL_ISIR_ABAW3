
def true_positives(tp, tn, fp, fn):
    return tp

def true_negatives(tp, tn, fp, fn):
    return tn

def false_positives(tp, tn, fp, fn):
    return fp

def false_negatives(tp, tn, fp, fn):
    return fn

def accuracy(tp, tn, fp, fn):
    return (tp + tn) / (fn + tp + fp + tn + 1e-7)

def precision(tp, tn, fp, fn):
    return (tp + 1e-7) / (tp + fp + 1e-7)

def recall(tp, tn, fp, fn):
    return (tp + 1e-7) / (tp + fn + 1e-7)

def specificity(tp, tn, fp, fn):
    return (tn + 1e-7) / (tn + fp + 1e-7)

def sensitivity(tp, tn, fp, fn):
    return (tp + 1e-7) / (tp + fn + 1e-7)

def balanced_accuracy(tp, tn, fp, fn):
    return (1/2) * (specificity(tp, tn, fp, fn) + sensitivity(tp, tn, fp, fn))

def f1score(tp, tn, fp, fn):
    prec = precision(tp, tn, fp, fn)
    rec = recall(tp, tn, fp, fn)
    return 2 * prec * rec / (prec + rec + 1e-7)

def abaw2(tp, tn, fp, fn):
    return (1/2) * (accuracy(tp, tn, fp, fn) + f1score(tp, tn, fp, fn))

SUPPORTED_CLASSIFICATION_FUNCTIONS = {"accuracy": accuracy,
                                      "tp": true_positives,
                                      "tn": true_negatives,
                                      "fp": false_positives,
                                      "fn": false_negatives,
                                      "sensitivity": sensitivity,
                                      "baccuracy": balanced_accuracy,
                                      "precision": precision,
                                      "recall": recall,
                                      "specificity": specificity,
                                      "f1score": f1score,
                                      "abaw2": abaw2}

def get_classification_function(name):
    return SUPPORTED_CLASSIFICATION_FUNCTIONS[name]
