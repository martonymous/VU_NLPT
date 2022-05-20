import pandas as pd
from simpletransformers.classification import ClassificationModel, ClassificationArgs


def load_data():
    olid_train = pd.read_csv('data/olid-train.csv')
    olid_test = pd.read_csv('data/olid-test.csv')
    olid_diagnostic = pd.read_csv('data/olid-subset-diagnostic-tests.csv')
    return olid_train, olid_test, olid_diagnostic


def load_bert(local_path=None):
    if not local_path:
        model_args = ClassificationArgs(num_train_epochs=20, train_batch_size=16, learning_rate=0.000001, overwrite_output_dir=True)
        bert = ClassificationModel('bert', 'bert-base-cased', args=model_args)
    else:
        bert = ClassificationModel('bert', local_path)
    return bert
