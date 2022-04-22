# Implement linguistic analyses using spacy
# Run them on data/preprocessed/train/sentences.txt
import spacy
from collections import Counter
from spacy import displacy

nlp = spacy.load('en_core_web_sm')
with open("data/preprocessed/train/sentences.txt", encoding="utf8") as file:
    train_sentences = file.read()
    train_sentences = train_sentences.replace('\n',' ')
# print(train_sentences)
nlp_data = nlp(train_sentences)


""" TOKENIZATION """
word_frequencies = Counter()

for token in nlp_data:
    if not (token.is_punct or token.is_space):          #Count every token which is not a punctuation or space
        word_frequencies.update([token.text])

num_tokens = len(nlp_data)
num_types = len(word_frequencies.keys())
num_words = sum(word_frequencies.values())
num_sent_words = num_words/len([sentence for sentence in nlp_data.sents])       #Number of words divided by number of sentences
num_length = sum([len(word) for word in word_frequencies.keys()])/num_types     #Sum of word characters divided by number of words

#RESULTS
print("# of tokens", num_tokens,
      "\n# of words", num_words,
      "\n# of types", num_types,
      "\n# of words / sentence", num_sent_words,
      "\navg length of words", num_length)


""" Word Classes """
all_word_freq = Counter()
word_classes = Counter()
all_words_tagged = {}
for token in nlp_data:
    keyname = str(token.tag_)+" "+str(token.pos_)       #Get POS tags
    all_words_tagged[token.text] = keyname              #Classify words
    word_classes.update([keyname])                      #Count POS tags
    all_word_freq.update([token.text])                  #Count All words

w_class_summary = []
for tag in sorted(word_classes, key=word_classes.get, reverse=True):        #Go through decr sorted POS tags
    temp_list = []
    # print("searched tag",tag)
    temp_list.extend([tag,word_classes[tag],round((word_classes[tag]/sum(word_classes.values())),2)," 3 most freq tags:"])     #Get tag, tag occurence, frequency
    temp_words = []
    for word in sorted(all_word_freq, key=all_word_freq.get, reverse=True):     #Go through decr sorted counted words
        if all_words_tagged[word] == tag :                                      #Get the 3 most frequented words with this tag
            # print("found tag",all_words_tagged[word])
            temp_words.append(word)
        if len(temp_words) == 3 :
            temp_list.extend(temp_words)
            break
    temp_list.append(" infreq tag:")
    for word in sorted(all_word_freq, key=all_word_freq.get, reverse=False):    #Go through incr sorted counted words
        if all_words_tagged[word] == tag :                                      #Get the least freq word with this tag
            # print("found tag",all_words_tagged[word])
            temp_list.append(word)
            break
    w_class_summary.append(temp_list)                                           #Store results and exit after 10
    if len(w_class_summary) == 10 : break

# print(all_word_freq, "\n")

#RESULTS
for POS in w_class_summary:
    print(*POS)

""" N-Grams """
token_bgram = Counter()
token_tgram = Counter()
fg_pos_bgram = Counter()
fg_pos_tgram = Counter()
uni_pos_bgram = Counter()
uni_pos_tgram = Counter()

for sentence in nlp_data.sents:
    for word in range(len(sentence)-1): #Go through the sentences except the last word and count bigrams
        token_bgram.update([sentence[word].text+" "+sentence[word+1].text])
        fg_pos_bgram.update([sentence[word].tag_+" "+sentence[word+1].tag_])
        uni_pos_bgram.update([sentence[word].pos_+" "+sentence[word+1].pos_])
    for word in range(len(sentence)-2): #Go through the sentences except the last 2 words and count trigrams
        token_tgram.update([sentence[word].text+" "+sentence[word+1].text+" "+sentence[word+2].text])
        fg_pos_tgram.update([sentence[word].tag_+" "+sentence[word+1].tag_+" "+sentence[word+2].tag_])
        uni_pos_tgram.update([sentence[word].pos_+" "+sentence[word+1].pos_+" "+sentence[word+2].pos_])

#Results
print("Token bigrams\n",token_bgram.most_common(3),"\nToken trigrams\n",token_tgram.most_common(3),
      "\nFine-grained POS bigrams\n",fg_pos_bgram.most_common(3),"\nFine-grained POS trigrams\n",fg_pos_tgram.most_common(3),
      "\nUniversal POS bigrams\n",uni_pos_bgram.most_common(3),"\nUniversal POS trigrams\n",uni_pos_tgram.most_common(3))

""" Lemmatization """
all_lemma_dict = {}
fall_lemma_dict = {}
#this part may have been an overkill
for token in nlp_data:
    lemma_pos = nlp(token.lemma_)[0].pos_   #get the lemma's POS
    if token.lemma_ in all_lemma_dict : #if the normalized form or token not in dict and the token's POS same as lemma's POS then we put it in the dict, excluding the lemma itself
        if token.norm_ not in all_lemma_dict[token.lemma_] and token.text not in all_lemma_dict[token.lemma_] and (token.text != token.lemma_) and (token.pos_ == lemma_pos):
            all_lemma_dict[token.lemma_].append(token.text)
    elif (token.text != token.lemma_) :
        all_lemma_dict[token.lemma_] = [token.text]

remove = [k for k in all_lemma_dict if len(all_lemma_dict[k])<3]    #clean up the lemmas that are under 3
for k in remove: del all_lemma_dict[k]

#Choosen lemma : fall (I think 4th key)
for inflection in all_lemma_dict['fall'] :
    for sentence in nlp_data.sents :    #get an example sentence for every inflection while avoiding duplicates
        if inflection in sentence.text and sentence.text not in fall_lemma_dict.values():
            fall_lemma_dict[inflection] = sentence.text
            break

#Results
# print(all_lemma_dict)
print(fall_lemma_dict)

"""Named Entity Recognition"""
entity_labels = Counter()
num_ner = len(nlp_data.ents) #number of entities
for entity in nlp_data.ents :   #count every entity
    entity_labels.update([entity.label_])
# print(entity_labels)

#Results
print("Different labels:",len(entity_labels.keys()))
print("Named entities:",num_ner)

five_sentences = ""
k = 0
for sentence in nlp_data.sents: #get the first 5 sentences and pass it to the NER pipeline
    five_sentences+= sentence.text
    k+=1
    if k == 5 : break
nlp_five_sentences = nlp(five_sentences)
displacy.serve(nlp_five_sentences, style='ent') #Starts a server of the rendered NER,Go to localhost:5000

