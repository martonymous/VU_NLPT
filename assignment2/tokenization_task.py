from utils import *


def run_task4():
    # load data
    train, test, diagnostic = load_data()

    token_counter = 0               # number tokens after subword splitting
    token_counter2 = 0              # number of subword tokens
    subword_counter = 0             # number of subwords
    token_split_counter = 0         # number of words split into subwords
    prev_token_is_subword = False

    model = load_bert('outputs/checkpoint-16560-epoch-20')

    result, model_outputs, wrong_predictions = model.eval_model(test)

    print(result)
    print(model_outputs)
    print(wrong_predictions)

    for i in range(train.shape[0]):
        tokens = model.tokenizer.tokenize(train['text'][i])
        if i <= 150 and i >= 50:
            print('\n', train['text'][i], '\ntokens: ', tokens)

        for token in tokens:
            token_counter += 1

            if token.startswith('##'):
                subword_counter += 1
                token_counter2 += 1
                if not prev_token_is_subword:
                    subword_counter += 1
                    token_split_counter += 1

                prev_token_is_subword = True
            else:
                prev_token_is_subword = False

    print('\nNr. of Tokens after splitting:                       ', token_counter)
    print('Nr. of Tokens before splitting:                      ', token_counter - token_counter2)
    print('Nr. of Subwords:                                     ', subword_counter)
    print('Number of tokens that have been split into subwords: ', token_split_counter)
    print('Average number of subwords per token (all tokens):   ', token_counter / (token_counter - token_counter2))
    print('Average number of subwords per token (split tokens): ', subword_counter / token_split_counter)

    voc = model.tokenizer.vocab
    lst = sorted(list(voc.keys()), key=len)

    print('\nLongest (sub)words and their length')
    for i in range(-10, 0):
        print(lst[i], len(lst[i]))
