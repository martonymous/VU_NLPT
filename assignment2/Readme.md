Authors: Marton Fejer, Gergo Pandurcsek, carol Rameder
Date: 20-05-2022

Code submission for NLP assignment 2

For part A, we performed all training and evaluation locally (instead of using colab) to speed up training. We run the entire task A from main.py, whereby the current default is load the model from a checkpoint (located in './outputs') and load the data from './data'. All relevant results are printed out when run.

For part B exercise 5 the perturbation is added to the data using add_perturbation.py.

The differences between the initial dataset and the modified (pertrubed) one are analyzed in NLP2.ipynb. The accuracy measurements were calculated manually with the numbers in the confussion matrix. 

For part B exercise 6 and 7, the code is in Assi2B_67.ipynb. During the implementation the predictions were saved so now the program doesn't need to load the model anymore. The negations were added by using the package's perturb.py add_negation function, but it had to be modified in order to work. The modifications made at the start of the for loop:

        for sentence in doc.sents:
            if len(sentence) < 3:
                continue
            try:
                root_id = [x.i for x in sentence if x.dep_ == 'ROOT'][0]
                root = doc[root_id]
            except:
                continue
