# Implement four baselines for the task.
# Majority baseline: always assigns the majority class of the training data
# Random baseline: randomly assigns one of the classes. Make sure to set a random seed and average the accuracy over 100 runs.
# Length baseline: determines the class based on a length threshold
# Frequency baseline: determines the class based on a frequency threshold

from model.data_loader import DataLoader
import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np
from numpy.random import RandomState
from wordfreq import word_frequency

# Each baseline returns predictions for the test data. The length and frequency baselines determine a threshold using the development data.

def majority_baseline(train_sentences, train_labels, testinput, testlabels):

    #cleanup
    train_tokens = []
    for sent in train_sentences:
        for word in sent.rstrip('\n').split(' '): train_tokens.append(word)
    t_labels = []
    for sent in train_labels:
        for label in sent.rstrip('\n').split(' '): t_labels.append(label)

    #Training
    train_df = pd.DataFrame(zip(train_tokens,t_labels),columns=["Train Token","Train Label"])
    major_class = train_df["Train Label"].value_counts().index[0]
    pred_classes = [element for element in train_df["Train Label"].value_counts().index]

    #Testing
    test_tokens = []
    for sent in testinput:
        for word in sent.rstrip('\n').split(' '): test_tokens.append(word)
    gold_labels = []
    for sent in testlabels:
        for label in sent.rstrip('\n').split(' '): gold_labels.append(label)

    pred_labels = [major_class for n in range(len(test_tokens))]

    #Results
    conf_m = confusion_matrix(gold_labels, pred_labels,labels=pred_classes)
    accuracy = (conf_m[0][0]+conf_m[1][1])/conf_m.sum()
    return accuracy, conf_m

def random_baseline(train_sentences, train_labels, testinput, testlabels):

    #cleanup
    train_tokens = []
    for sent in train_sentences:
        for word in sent.rstrip('\n').split(' '): train_tokens.append(word)
    t_labels = []
    for sent in train_labels:
        for label in sent.rstrip('\n').split(' '): t_labels.append(label)

    #Training
    train_df = pd.DataFrame(zip(train_tokens,t_labels),columns=["Train Token","Train Label"])
    pred_classes = [element for element in train_df["Train Label"].value_counts().index]
    rng = RandomState(69)

   #Testing
    test_tokens = []
    for sent in testinput:
        for word in sent.rstrip('\n').split(' '): test_tokens.append(word)
    gold_labels = []
    for sent in testlabels:
        for label in sent.rstrip('\n').split(' '): gold_labels.append(label)

    #Results
    accuracy = []
    for n in range(100):
        pred_labels = [rng.choice(pred_classes, 1) for n in range(len(test_tokens))]
        conf_m = confusion_matrix(gold_labels, pred_labels,labels=pred_classes)
        accuracy.append((conf_m[0][0]+conf_m[1][1])/conf_m.sum())
    accuracy = np.mean(accuracy)
    return accuracy, conf_m

def freq_baseline(train_sentences, train_labels, testinput, testlabels, threshold, flip):

    #cleanup
    train_tokens = []
    for sent in train_sentences:
        for word in sent.rstrip('\n').split(' '): train_tokens.append(word)
    t_labels = []
    for sent in train_labels:
        for label in sent.rstrip('\n').split(' '): t_labels.append(label)

    #Training
    train_df = pd.DataFrame(zip(train_tokens,t_labels),columns=["Train Token","Train Label"])
    pred_classes = [element for element in train_df["Train Label"].value_counts().index]

    #Testing
    test_tokens = []
    for sent in testinput:
        for word in sent.rstrip('\n').split(' '): test_tokens.append(word)
    gold_labels = []
    for sent in testlabels:
        for label in sent.rstrip('\n').split(' '): gold_labels.append(label)
    pred_classes_og = pred_classes
    if flip : pred_classes.reverse()
    pred_labels = [pred_classes[0] if word_frequency(test_tokens[n], 'en', wordlist='best', minimum=0.0)>(threshold/100000) else pred_classes[1] for n in range(len(test_tokens))]

    #Results
    conf_m = confusion_matrix(gold_labels, pred_labels,labels=pred_classes_og)
    accuracy = (conf_m[0][0]+conf_m[1][1])/conf_m.sum()
    return accuracy, conf_m

def length_baseline(train_sentences, train_labels, testinput, testlabels, threshold, flip):

    #cleanup
    train_tokens = []
    for sent in train_sentences:
        for word in sent.rstrip('\n').split(' '): train_tokens.append(word)
    t_labels = []
    for sent in train_labels:
        for label in sent.rstrip('\n').split(' '): t_labels.append(label)

    #Training
    train_df = pd.DataFrame(zip(train_tokens,t_labels),columns=["Train Token","Train Label"])
    pred_classes = [element for element in train_df["Train Label"].value_counts().index]

    #Testing
    test_tokens = []
    for sent in testinput:
        for word in sent.rstrip('\n').split(' '): test_tokens.append(word)
    gold_labels = []
    for sent in testlabels:
        for label in sent.rstrip('\n').split(' '): gold_labels.append(label)
    pred_classes_og = pred_classes
    if flip : pred_classes.reverse()
    pred_labels = [pred_classes[0] if len(test_tokens[n])>threshold else pred_classes[1] for n in range(len(test_tokens))]

    #Results
    conf_m = confusion_matrix(gold_labels, pred_labels,labels=pred_classes_og)
    accuracy = (conf_m[0][0]+conf_m[1][1])/conf_m.sum()
    return accuracy, conf_m


if __name__ == '__main__':
    train_path = "data/preprocessed/train"
    dev_path = "data/preprocessed/val"
    test_path = "data/preprocessed/test"

    # Note: this loads all instances into memory. If you work with bigger files in the future, use an iterator instead.

    with open(train_path + "/sentences.txt", encoding="utf8") as sent_file:
        train_sentences = sent_file.readlines()

    with open(train_path + "/labels.txt", encoding="utf8") as label_file:
        train_labels = label_file.readlines()

    with open(dev_path + "/sentences.txt", encoding="utf8") as dev_file:
        dev_sentences = dev_file.readlines()

    with open(dev_path + "/labels.txt", encoding="utf8") as dev_label_file:
        dev_labels = dev_label_file.readlines()
    with open(test_path + "/sentences.txt", encoding="utf8") as testfile:
        testinput = testfile.readlines()

    with open(test_path + "/labels.txt", encoding="utf8") as test_label_file:
        testlabels = test_label_file.readlines()

    dev_majority_accuracy, dev_majority_predictions = majority_baseline(train_sentences, train_labels,dev_sentences, dev_labels)
    dev_random_accuracy, dev_random_predictions = random_baseline(train_sentences, train_labels,dev_sentences, dev_labels)
    dev_freq_accuracy, dev_freq_predictions = freq_baseline(train_sentences, train_labels,dev_sentences, dev_labels, 4, False) #threshold between 0-40 #flip T/F
    dev_length_accuracy, dev_length_predictions = length_baseline(train_sentences, train_labels,dev_sentences, dev_labels, 8, True) #threshold between 1-15 #flip T/F

    test_majority_accuracy, test_majority_predictions = majority_baseline(train_sentences, train_labels, testinput,testlabels)
    test_random_accuracy, test_random_predictions = random_baseline(train_sentences, train_labels, testinput, testlabels)
    test_freq_accuracy, test_freq_predictions = freq_baseline(train_sentences, train_labels, testinput, testlabels, 4, False) #threshold between 0-40 #flip T/F
    test_length_accuracy, test_length_predictions = length_baseline(train_sentences, train_labels, testinput, testlabels, 8, True) #threshold between 1-15 #flip T/F

    print("dev major", dev_majority_accuracy)
    print("dev rand", dev_random_accuracy)
    print("dev freq", dev_freq_accuracy)
    print("dev len", dev_length_accuracy)
    print("test major", test_majority_accuracy)
    print("test rand", test_random_accuracy)
    print("test freq", test_freq_accuracy)
    print("test len", test_length_accuracy)

    #ORDER IS 'N' 'C' !!!! POSITIVE IS N HERE

    #region filewrite
    a_file = open("baseline predictions/dev_major.txt", "w")
    for row in dev_majority_predictions:
        np.savetxt(a_file, row)
    a_file.close()

    a_file = open("baseline predictions/dev_random.txt", "w")
    for row in dev_random_predictions:
        np.savetxt(a_file, row)
    a_file.close()

    a_file = open("baseline predictions/dev_freq.txt", "w")
    for row in dev_freq_predictions:
        np.savetxt(a_file, row)
    a_file.close()

    a_file = open("baseline predictions/dev_length.txt", "w")
    for row in dev_length_predictions:
        np.savetxt(a_file, row)
    a_file.close()

    a_file = open("baseline predictions/test_major.txt", "w")
    for row in test_majority_predictions:
        np.savetxt(a_file, row)
    a_file.close()

    a_file = open("baseline predictions/test_random.txt", "w")
    for row in test_random_predictions:
        np.savetxt(a_file, row)
    a_file.close()

    a_file = open("baseline predictions/test_freq.txt", "w")
    for row in test_freq_predictions:
        np.savetxt(a_file, row)
    a_file.close()

    a_file = open("baseline predictions/test_length.txt", "w")
    for row in test_length_predictions:
        np.savetxt(a_file, row)
    a_file.close()
    #endregion