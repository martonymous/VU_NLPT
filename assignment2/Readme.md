For part A, we performed all training and evaluation locally (instead of using colab) to speed up training. We run the entire task A from main.py, whereby the current default is load the model from a checkpoint (located in './outputs') and load the data from './data'. All relevant results are printed out when run.

For part B exercise 5 the perturbation is added to the data using add_perturbation.py. 

The differences between the initial dataset and the modified (pertrubed) one are analyzed in NLP2.ipynb. The accuracy measurements were calculated manually with the numbers in the confussion matrix. 
