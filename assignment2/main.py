import pandas as pd
import numpy as np
from utils import *
from tokenization_task import *
import seaborn as sns
import matplotlib.pyplot as plt
import random
from sklearn.metrics import classification_report

# ENTER PATH TO LOCAL CHECKPOINT HERE, If BERT-CHECKPOINT IS SAVED, EG:
BERT_CHECKPOINT = 'outputs/checkpoint-16560-epoch-20'

def random_baseline(train: pd.DataFrame, test: pd.DataFrame):
    train_labels = train['labels'].unique()
    pred = [random.choice(train_labels) for i in range(test.shape[0])]
    print('\nClassification Report - Random Baseline\n', classification_report(test['labels'], pred, target_names=['Not Offensive', 'Offensive']))

def majority_baseline(train: pd.DataFrame, test: pd.DataFrame):
    maj_label = train['labels'].value_counts().argmax()
    pred = [maj_label] * test.shape[0]
    print('\nClassification Report - Majority Baseline\n', classification_report(test['labels'], pred, target_names=['Not Offensive', 'Offensive']))


if __name__ == '__main__':

    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)

    """ Assignment 2A """

    # load data
    train, test, diagnostic = load_data()

    # get class distributions and examples
    print('\nClass distribution:\n', train['labels'].value_counts())
    print('\nRelative distribution:\n', train['labels'].value_counts(normalize=True))
    sns.countplot(data=train, x='labels')

    plt.show()

    train_sample_off = train[train['labels'] == 1].sample(n=1, random_state=random_seed)
    train_sample_not = train[train['labels'] == 0].sample(n=1, random_state=random_seed)

    print('EXAMPLE - Offensive:     ', train_sample_off['text'].values)
    print('EXAMPLE - Not Offensive: ', train_sample_not['text'].values)


    # run baseline predictions
    random_baseline(train, test)
    majority_baseline(train, test)

    # simple transformers doesn't use this column for trianing anyway
    train = train.drop('id', axis=1)

    # load and retrain bert
    if BERT_CHECKPOINT:
        berty = load_bert(BERT_CHECKPOINT)
    else:
        berty = load_bert()
    berty.train_model(train)

    # Evaluate the model
    result, model_outputs, wrong_predictions = berty.eval_model(test)

    print(result)
    print(model_outputs)
    print(wrong_predictions)

    run_task4()
