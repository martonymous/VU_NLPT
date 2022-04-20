# The original code only outputs the accuracy and the loss.
# Process the file model_output.tsv and calculate precision, recall, and F1 for each class

import numpy as np

def load_model_outputfile(file_name, C_or_N='C'):
    outputs, labels = [], []
    with open(file_name, "r", encoding='windows-1252') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i] != '----------\n':
                parts = lines[i].rstrip('\n').split('\t')
                if parts[1] == 'N': labels.append(0)
                else:               labels.append(1)

                if parts[2] == 'N': outputs.append(0)
                else:               outputs.append(1)

    outputs, labels = np.array(outputs), np.array(labels)
    if C_or_N == 'N':
        outputs = 1 - outputs
        labels  = 1 - labels
    label_weight = np.sum(labels == 1)

    return outputs, labels, label_weight


def precision(outputs, labels):
    # calculate true positives and false positives
    tp = np.sum(np.logical_and(outputs == 1, labels == 1))
    fp = np.sum(np.logical_and(outputs == 1, labels == 0))

    if (tp+fp) != 0:
        return tp/(tp+fp)
    else:
        return np.nan


def recall(outputs, labels):
    # calculate true positives and false negatives
    tp = np.sum(np.logical_and(outputs == 1, labels == 1))
    fn = np.sum(np.logical_and(outputs == 0, labels == 1))

    if (tp+fn) != 0:
        return tp/(tp+fn)
    else:
        return np.nan


def f1(outputs, labels):
    rec = recall(outputs, labels)
    prec = precision(outputs, labels)

    if prec == np.nan or rec == np.nan or (prec+rec) == 0:
        return np.nan
    else:
        return 2*((prec*rec)/(prec+rec))


if __name__ == '__main__':
    filenames = {
        'random': './baseline predictions/test_rand_pred.tsv',
        'majority': './baseline predictions/test_maj_pred.tsv',
        'frequency': './baseline predictions/test_freq_pred.tsv',
        'length': './baseline predictions/test_len_pred.tsv',
        'lstm': 'experiments/base_model/model_output.tsv'
    }

    f1s = {
        'random': [],
        'majority': [],
        'frequency': [],
        'length': [],
        'lstm': []
    }

    label_weights = {
        'random': [],
        'majority': [],
        'frequency': [],
        'length': [],
        'lstm': []
    }

    for category in ['N', 'C']:
        print(f'--- {category} ---')
        for file in filenames:

            out, lab, label_weight = load_model_outputfile(filenames[file], category)
            prec = precision(out, lab)
            rec  = recall(out, lab)
            f1_score = f1(out, lab)

            f1s[file].append(f1_score)
            label_weights[file].append(label_weight)

            print(f'{file}:\n\nPrecision : {prec}\nRecall    : {rec}\nF1-score  : {f1_score}\n\n\n')

    print('Weighted F1-scores\n')
    for model in f1s:
        weighted_f1 = ((f1s[model][0] * label_weights[model][0]) + (f1s[model][1] * label_weights[model][1])) / (label_weights[model][0] + label_weights[model][1])
        print(f'{model}:  {weighted_f1}')

