# Implement four baselines for the task.
# Majority baseline: always assigns the majority class of the training data
# Random baseline: randomly assigns one of the classes. Make sure to set a random seed and average the accuracy over 100 runs.
# Length baseline: determines the class based on a length threshold
# Frequency baseline: determines the class based on a frequency threshold

from model.data_loader import DataLoader
import os
from sklearn.metrics import confusion_matrix
import numpy as np
from numpy.random import RandomState
import matplotlib.pyplot as plt
import spacy
import pandas as pd
from wordfreq import word_frequency
import seaborn as sns

pd.set_option('display.float_format', str)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('expand_frame_repr', False)
nlp = spacy.load('en_core_web_sm')

f_path = "data/original/english/WikiNews_Train.tsv"
dataset = pd.read_csv(f_path, sep='\t',header=None)
# print(dataset.head(5))

token_twords = [nlp(text) for text in dataset.iloc[:,4]] #tokenized target words
#Results
print("instances 1 : ",(dataset.iloc[:,9]==1).sum())    #count the number of 1s
print("instances 0 : ",(dataset.iloc[:,9]==0).sum())    #count the number of 0s
print("prob label\n",dataset.iloc[:,10].describe())     #get numerical summary
print("median : ",dataset.iloc[:,10].median())          #get median
print("more than 1 token : ",len([token for token in token_twords if len(token)>1]))       #get the number of instances with more than 1 token
print("max number of tokens : ", len(max(token_twords,key=len)))        #get the instance with the max number of tokens

sc_tokens = [] #single complex tokens
for bin, prob, token in zip(dataset.iloc[:,9],dataset.iloc[:,10],token_twords): #looping through the binary,probability and tokens together
    if bin == 1 and len(token) == 1:
        sc_tokens.append([token[0].text,prob,len(token[0]),word_frequency(token[0].text, 'en', wordlist='best', minimum=0.0),token[0].pos_])    #save the actual word,probability,length,frequency and POS tag

df_tokens = pd.DataFrame(sc_tokens,columns=["Token","Prob","Len","Freq","POS"])
# print(df_tokens)

#Results
print("Correlation Len:Prob :",df_tokens["Len"].corr(df_tokens["Prob"],method='pearson')) #pearson correlation for length and probability
print("Correlation Freq:Prob :",df_tokens["Freq"].corr(df_tokens["Prob"],method='pearson')) #pearson correlation for frequency and probability
plt.figure()
sns.scatterplot(data=df_tokens,x="Len",y="Prob")
plt.figure()
sns.scatterplot(data=df_tokens,x="Freq",y="Prob")
plt.figure()
sns.scatterplot(data=df_tokens,x="POS",y="Prob")
# plt.show() #uncomment to see the plots

# Each baseline returns predictions for the test data. The length and frequency baselines determine a threshold using the development data.

def majority_baseline(train_sentences, train_labels, testinput, testlabels, filename):

    #cleanup : removing newlines, collecting tokens and labels
    train_tokens = []
    for sent in train_sentences:
        for word in sent.rstrip('\n').split(' '): train_tokens.append(word)
    t_labels = []
    for sent in train_labels:
        for label in sent.rstrip('\n').split(' '): t_labels.append(label)

    #Training   : get the majority class
    train_df = pd.DataFrame(zip(train_tokens,t_labels),columns=["Train Token","Train Label"])
    major_class = train_df["Train Label"].value_counts().index[0]
    pred_classes = [element for element in train_df["Train Label"].value_counts().index] #prediction classes

    #Testing : collecting the gold data, creating predictions
    test_tokens = []
    for sent in testinput:
        for word in sent.rstrip('\n').split(' '): test_tokens.append(word)
    gold_labels = []
    for sent in testlabels:
        for label in sent.rstrip('\n').split(' '): gold_labels.append(label)

    pred_labels = [major_class for n in range(len(test_tokens))]

    #Results : create confusion matrix and compute accuracy
    conf_m = confusion_matrix(gold_labels, pred_labels,labels=pred_classes)
    accuracy = (conf_m[0][0]+conf_m[1][1])/conf_m.sum() #TP+TN/All

    output_predictions(filename, test_tokens, gold_labels, pred_labels) #save the predictions for further evaluation

    return accuracy, conf_m

def random_baseline(train_sentences, train_labels, testinput, testlabels, filename):

    #cleanup : removing newlines, collecting tokens and labels
    train_tokens = []
    for sent in train_sentences:
        for word in sent.rstrip('\n').split(' '): train_tokens.append(word)
    t_labels = []
    for sent in train_labels:
        for label in sent.rstrip('\n').split(' '): t_labels.append(label)

    #Training : get prediction classes and setup random generator
    train_df = pd.DataFrame(zip(train_tokens,t_labels),columns=["Train Token","Train Label"])
    pred_classes = [element for element in train_df["Train Label"].value_counts().index]
    rng = RandomState(69)

    #Testing : collecting the gold data
    test_tokens = []
    for sent in testinput:
        for word in sent.rstrip('\n').split(' '): test_tokens.append(word)
    gold_labels = []
    for sent in testlabels:
        for label in sent.rstrip('\n').split(' '): gold_labels.append(label)

    #Results : generate a randomclass a 100 times, measure accuracy and finally average them
    accuracy = []
    for n in range(100):
        pred_labels = [rng.choice(pred_classes, 1) for n in range(len(test_tokens))]
        conf_m = confusion_matrix(gold_labels, pred_labels,labels=pred_classes)
        accuracy.append((conf_m[0][0]+conf_m[1][1])/conf_m.sum())
    accuracy = np.mean(accuracy)

    output_predictions(filename, test_tokens, gold_labels, pred_labels) #save the predictions for further evaluation

    return accuracy, conf_m

def freq_baseline(train_sentences, train_labels, testinput, testlabels, threshold, flip, filename):

    #cleanup : removing newlines, collecting tokens and labels
    train_tokens = []
    for sent in train_sentences:
        for word in sent.rstrip('\n').split(' '): train_tokens.append(word)
    t_labels = []
    for sent in train_labels:
        for label in sent.rstrip('\n').split(' '): t_labels.append(label)

    #Training : get prediction classes
    train_df = pd.DataFrame(zip(train_tokens,t_labels),columns=["Train Token","Train Label"])
    pred_classes = [element for element in train_df["Train Label"].value_counts().index]

    #Testing :collecting gold data and make predictions
    test_tokens = []
    for sent in testinput:
        for word in sent.rstrip('\n').split(' '): test_tokens.append(word)
    gold_labels = []
    for sent in testlabels:
        for label in sent.rstrip('\n').split(' '): gold_labels.append(label)
    pred_classes_og = pred_classes
    if flip : pred_classes.reverse() #flip the threshold meaning
    pred_labels = [pred_classes[0] if word_frequency(test_tokens[n], 'en', wordlist='best', minimum=0.0)>(threshold/100000) else pred_classes[1] for n in range(len(test_tokens))] #if frequency above threshold then first class, else second class

    #Results : create confusion matrix and compute accuracy
    conf_m = confusion_matrix(gold_labels, pred_labels,labels=pred_classes_og)
    accuracy = (conf_m[0][0]+conf_m[1][1])/conf_m.sum()

    output_predictions(filename, test_tokens, gold_labels, pred_labels) #save the predictions for further evaluation

    return accuracy, conf_m

def length_baseline(train_sentences, train_labels, testinput, testlabels, threshold, flip, filename):

    #cleanup : removing newlines, collecting tokens and labels
    train_tokens = []
    for sent in train_sentences:
        for word in sent.rstrip('\n').split(' '): train_tokens.append(word)
    t_labels = []
    for sent in train_labels:
        for label in sent.rstrip('\n').split(' '): t_labels.append(label)

    #Training: get prediction classes
    train_df = pd.DataFrame(zip(train_tokens,t_labels),columns=["Train Token","Train Label"])
    pred_classes = [element for element in train_df["Train Label"].value_counts().index]

    #Testing : collect gold data and make predictions
    test_tokens = []
    for sent in testinput:
        for word in sent.rstrip('\n').split(' '): test_tokens.append(word)
    gold_labels = []
    for sent in testlabels:
        for label in sent.rstrip('\n').split(' '): gold_labels.append(label)
    pred_classes_og = pred_classes
    if flip : pred_classes.reverse() #flip the threshold meaning
    pred_labels = [pred_classes[0] if len(test_tokens[n])>threshold else pred_classes[1] for n in range(len(test_tokens))] #if frequency above threshold then first class, else second class

    #Results : create confusion matrix and compute accuracy
    conf_m = confusion_matrix(gold_labels, pred_labels,labels=pred_classes_og)
    accuracy = (conf_m[0][0]+conf_m[1][1])/conf_m.sum()

    output_predictions(filename, test_tokens, gold_labels, pred_labels) #save the predictions for further evaluation

    return accuracy, conf_m


def output_predictions(outfile, word, gold, prediction): #save the predictions for further evaluation
    with open(outfile, "w", encoding='windows-1252') as f:
        for i in range(len(word)):
            f.write("\t".join([word[i], gold[i], prediction[i][0]]))
            f.write("\n")


if __name__ == '__main__':
    train_path = "data/preprocessed/train"
    dev_path = "data/preprocessed/val"
    test_path = "data/preprocessed/test"

    if not os.path.exists('./baseline predictions'):
        os.mkdir('./baseline predictions')

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

    dev_majority_accuracy, dev_majority_predictions = majority_baseline(train_sentences, train_labels,dev_sentences, dev_labels, './baseline predictions/maj_pred.tsv')
    dev_random_accuracy, dev_random_predictions = random_baseline(train_sentences, train_labels,dev_sentences, dev_labels, './baseline predictions/rand_pred.tsv')
    dev_freq_accuracy, dev_freq_predictions = freq_baseline(train_sentences, train_labels,dev_sentences, dev_labels, 4, False, './baseline predictions/freq_pred.tsv') #threshold between 0-40 #flip T/F
    dev_length_accuracy, dev_length_predictions = length_baseline(train_sentences, train_labels,dev_sentences, dev_labels, 8, True, './baseline predictions/len_pred.tsv') #threshold between 1-15 #flip T/F

    test_majority_accuracy, test_majority_predictions = majority_baseline(train_sentences, train_labels, testinput,testlabels, './baseline predictions/test_maj_pred.tsv')
    test_random_accuracy, test_random_predictions = random_baseline(train_sentences, train_labels, testinput, testlabels, './baseline predictions/test_rand_pred.tsv')
    test_freq_accuracy, test_freq_predictions = freq_baseline(train_sentences, train_labels, testinput, testlabels, 4, False, './baseline predictions/test_freq_pred.tsv') #threshold between 0-40 #flip T/F
    test_length_accuracy, test_length_predictions = length_baseline(train_sentences, train_labels, testinput, testlabels, 8, True, './baseline predictions/test_len_pred.tsv') #threshold between 1-15 #flip T/F

    print("dev major", dev_majority_accuracy)
    print("dev rand", dev_random_accuracy)
    print("dev freq", dev_freq_accuracy)
    print("dev len", dev_length_accuracy)
    print("test major", test_majority_accuracy)
    print("test rand", test_random_accuracy)
    print("test freq", test_freq_accuracy)
    print("test len", test_length_accuracy)

    #ORDER IS 'N' 'C' for confusion matrix
