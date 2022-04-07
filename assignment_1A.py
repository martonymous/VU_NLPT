import spacy
from collections import Counter

with open('data/preprocessed/train/sentences.txt', 'r', encoding='utf8') as f:
    text = f.read()

# load simple English model and use it to analyze sentences
nlp = spacy.load('en_core_web_sm')
doc = nlp(text)

all_words = [tokens.text for tokens in doc if not tokens.is_punct and not tokens.is_stop]
all_words_len = 0
for word in all_words:
    all_words_len += len(word)

word_frequencies = Counter()
word_frequencies.update(all_words)
types = word_frequencies.keys()


sentences = doc.sents
num_sentences = 0
for sentence in sentences:
    sent_words = [token.text for token in sentence if not token.is_stop and not token.is_punct]  # this isn't really necessary for this part
    num_sentences += 1


print("Number of Tokens:            ", len(doc))
print("Number of Words:             ", len(all_words))
print("Number of Types:             ", len(types))
print("Number of Sentences:         ", num_sentences)
print("Average nr. Words/Sentence:  ", len(all_words)/num_sentences)
print("Average Word Length:         ", all_words_len/len(all_words))
