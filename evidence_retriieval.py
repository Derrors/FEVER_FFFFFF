# _*_ coding: utf-8 _*_

import numpy as np
import json
import os
import pickle

from document_retrieval import doc_ret
from sentence_retrieval import sent_ret
from fever_io import (load_doc_lines, load_paper_dataset, load_split_trainset,
                      titles_to_jsonl_num)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def evi_ret(data=dict(), n_docs=5, n_sents=5):
    with open('./data/preprocessed_data/edocs.bin', 'rb') as rb:
        edocs = pickle.load(rb)

    with open('./results/doc_ret/doc_ret_model.bin', 'rb') as rb:
        dmodel = pickle.load(rb)

    t2jnum = titles_to_jsonl_num()

    with open('./results/sent_ret/sent_ret_model.bin', 'rb') as rb:
        lmodel = pickle.load(rb)

    docs = doc_ret(data, edocs, model=dmodel, best=n_docs)
    lines = load_doc_lines(docs, t2jnum)
    evidence = sent_ret(data, docs, lines, model=lmodel, best=n_sents)

    return docs, evidence


def to_fever_format(data, docs, evidence):
    data2 = data.copy()
    for instance in data2:
        cid = instance['id']
        instance['predicted_pages'] = list()
        instance['predicted_sentences'] = list()
        instance['scored_sentences'] = list()
        for doc, score in docs[cid]:
            instance['predicted_pages'].append(doc)
        for doc, line, score in evidence[cid]:
            instance['predicted_sentences'].append([doc, line])
            instance['scored_sentences'].append([doc, line, score])
    return data2


def run_evi_ret(config):
    train, dev = load_paper_dataset(config['train_input'], dev=config['dev_input'])

    for split, data in [('train', train), ('dev', dev)]:
        if split == 'train':
            out_file = config['train_output']
        if split == 'dev':
            out_file = config['dev_output']

        docs, evidence = evi_ret(data, n_docs=config['n_docs'], n_sents=config['n_sents'])
        pred = to_fever_format(data, docs, evidence)

        with open(out_file, 'w') as w:
            for example in pred:
                w.write(json.dumps(example, cls=NpEncoder) + '\n')


if __name__ == '__main__':
    config = {}
    run_evi_ret(config)
