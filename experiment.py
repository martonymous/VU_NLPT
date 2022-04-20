from TODO_detailed_evaluation import *
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == '__main__':
    """
        Run experiment with embedding size
    """
    def init_vars():
        f1s = {
            'a': [],
            'b': [],
            'c': [],
            'd': [],
            'e': [],
            'f': []
        }

        label_weights = {
            'a': [],
            'b': [],
            'c': [],
            'd': [],
            'e': [],
            'f': []
        }
        return f1s, label_weights

    # define experiment parameters
    emb_d, lrs = [3, 10, 25, 50, 100, 250], [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03]
    experiments, weighted_f1s = [emb_d, lrs], {'embedding_size': [], 'learning_rate': []}

    for j, experiment in enumerate(experiments):
        print(f'Running experiment: {list(weighted_f1s.keys())[j]}')
        f1s, label_weights = init_vars()
        for i, key in enumerate(f1s):

            # update params file
            with open("./experiments/base_model/params.json", "r+") as a_file:
                params = json.load(a_file)
                if j == 0:
                    params['learning_rate'] = 0.001
                    params["embedding_dim"] = experiment[i]
                    params["lstm_hidden_dim"] = experiment[i]
                else:
                    params['learning_rate'] = experiment[i]
                    params["embedding_dim"] = 50
                    params["lstm_hidden_dim"] = 50
                a_file.close()

            with open("./experiments/base_model/params.json", "w") as a_file:
                json.dump(params, a_file, indent=4)
                a_file.close()

            print('Training Model...\n')
            trainer = open('train.py')
            train_file = trainer.read()
            exec(train_file)

            print('Evaluating Model...\n')
            evaluator = open('evaluate.py')
            evaluate_file = evaluator.read()
            exec(evaluate_file)

            print('Getting metrics...\n')
            for class_label in ['N', 'C']:
                out, lab, label_weight = load_model_outputfile('experiments/base_model/model_output.tsv', class_label)
                f1_score = f1(out, lab)
                f1s[key].append(f1_score)
                label_weights[key].append(label_weight)

        for key in f1s:
            weighted_f1s[list(weighted_f1s.keys())[j]].append(((f1s[key][0] * label_weights[key][0]) + (f1s[key][1] * label_weights[key][1])) / (label_weights[key][0] + label_weights[key][1]))

    weighted_f1s['embedding_size'] = np.array(weighted_f1s['embedding_size'])
    weighted_f1s['learning_rate'] = np.array(weighted_f1s['learning_rate'])
    f1_scores = pd.DataFrame(weighted_f1s)

    sns.lineplot(data=f1_scores)
    plt.show()
    
