import csv
import spacy
import pandas as pd
from .perturb import Perturb
import numpy as np

def main():
    np.random.seed(42)
    data=pd.read_csv('D:/Master/Year_1/Quarter_5/NLP/Assignment 2/checklist/checklist/olid/olid-subset-diagnostic-tests.csv', index_col=0,encoding='utf-8')
    #train_data=pd.read_csv('D:/Master/Year_1/Quarter_5/NLP/Assignment 2/checklist/checklist/olid/olid-train.csv', index_col=0,encoding='utf-8')
    #test_data =pd.read_csv('D:/Master/Year_1/Quarter_5/NLP/Assignment 2/checklist/checklist/olid/olid-test.csv', index_col=0, encoding='utf-8')
    #nlp = spacy.load('en_core_web_sm')
    #pdata = list(nlp.pipe(data["text"]))

    tdata=list(data["text"])
    pert_tweets = []
    for tweet in tdata:
        pert_tweets.append(Perturb.add_typos(tweet, 4))
    #print(pert_tweets)

    pert_frame={
        'text':pert_tweets,
        'labels':list(data["labels"])
    }
    pert_frame=pd.DataFrame(pert_frame)
    pert_frame.to_csv('perturbed2.csv', index=False, encoding='utf-8')




if __name__ == "__main__":
    main()