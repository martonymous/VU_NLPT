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
print(dataset.head(5))

token_twords = [nlp(text) for text in dataset.iloc[:,4]] #tokenized target words
#Results
# print("instances 1 : ",(dataset.iloc[:,9]==1).sum())
# print("instances 0 : ",(dataset.iloc[:,9]==0).sum())
# print("prob label\n",dataset.iloc[:,10].describe())
# print("median : ",dataset.iloc[:,10].median())
# print("more than 1 token : ",len([token for token in token_twords if len(token)>1]))
# print("max number of tokens : ", len(max(token_twords,key=len)))

sc_tokens = [] #single complex tokens
for bin, prob, token in zip(dataset.iloc[:,9],dataset.iloc[:,10],token_twords):
    if bin == 1 and len(token) == 1:
        sc_tokens.append([token[0].text,prob,len(token[0]),word_frequency(token[0].text, 'en', wordlist='best', minimum=0.0),token[0].pos_])

df_tokens = pd.DataFrame(sc_tokens,columns=["Token","Prob","Len","Freq","POS"])
# print(df_tokens)

#Results
# print(df_tokens["Len"].corr(df_tokens["Prob"],method='pearson'))
# print(df_tokens["Freq"].corr(df_tokens["Prob"],method='pearson')
# plt.figure()
# sns.scatterplot(data=df_tokens,x="Len",y="Prob")
# plt.figure()
# sns.scatterplot(data=df_tokens,x="Freq",y="Prob")
# plt.figure()
# sns.scatterplot(data=df_tokens,x="POS",y="Prob")
# plt.show()