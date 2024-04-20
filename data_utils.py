import pandas as pd
import json
import pickle
import numpy as np
from utils import ROOT_DIR

def load_toxicData():
    train_df = pd.read_csv('./Data/ToxicBias/Train.csv')
    train_df['bias'] = train_df['bias'].apply(lambda x: 1 if x=='bias' else 0)
    train_sentences = list()
    for i in range(train_df.shape[0]):
        d = {'sentence': train_df['text'], 'label': train_df['bias'], 'idx': i}
        train_sentences.append(d)
    train_labels = list(train_df['bias'])

    test_df = pd.read_csv('./data/ToxicBias/Test.csv')
    test_df['bias'] = test_df['bias'].apply(lambda x: 1 if x=='bias' else 0)
    test_sentences = list()
    for i in range(test_df.shape[0]):
        d = {'sentence': test_df['text'], 'label': test_df['bias'], 'idx': i}
        test_sentences.append(d)
    test_labels = list(test_df['bias'])
        
    return train_sentences, train_labels, test_sentences, test_labels


def custom_load_dataset(params, change_params=True):
    """
    Load train and test data
    :param params: experiment parameter, which contains dataset spec
    :return: train_x, train_y, test_x, test_y
    """

    if params['dataset'] == 'toxicbias':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_toxicData()
        if change_params:
            params['prompt_prefix'] = "In this task, you are given sentences. The task is to classify a sentence as \"Positive\" if the sentence has race, religion, political, lgbtq, gender bias or as \"Negative\" if the sentence is unbiased.\n\n"
            params["q_prefix"] = "Sentence: "
            params["a_prefix"] = "Bias: "
            params['label_dict'] = {0: ['Negative'], 1: ['Positive']}
            params['inv_label_dict'] = {'Negative': 0, 'Positive': 1}
            params['task_format'] = 'classification'
            params['num_tokens_to_predict'] = 1
    else:
        raise NotImplementedError
    print('train set length: ', len(orig_train_sentences), ' test set length: ', len(orig_test_sentences), flush=True)
    return orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels
