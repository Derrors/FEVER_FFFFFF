# _*_ coding: utf-8 _*_

import numpy as np
import argparse
import json
import os
import pickle

from doc_ir import doc_ir
from fever_io import (load_doc_lines, load_paper_dataset, load_split_trainset, titles_to_jsonl_num)
from line_ir import line_ir


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


def get_evidence(data=dict(), n_docs=5, n_sents=5):
    with open('./data/edocs.bin', 'rb') as rb:
        edocs = pickle.load(rb)

    with open('./data/doc_ir_model.bin', 'rb') as rb:
        dmodel = pickle.load(rb)

    t2jnum = titles_to_jsonl_num()

    with open('./data/line_ir_model.bin', 'rb') as rb:
        lmodel = pickle.load(rb)

    docs = doc_ir(data, edocs, model=dmodel, best=n_docs)
    lines = load_doc_lines(docs, t2jnum)
    evidence = line_ir(data, docs, lines, model=lmodel, best=n_sents)

    return docs, evidence


def fever_predictions(data, evidence):
    data2 = data.copy()
    for instance in data2:
        cid = instance['id']
        instance['predicted_evidence'] = list()
        instance['predicted_label'] = instance['label']
        for doc, line, _ in evidence[cid]:
            instance['predicted_evidence'].append([doc, line])
    return data2


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


def fever_score():
    train, dev = load_split_trainset(9999)
    docs, evidence = get_evidence(dev)
    from scorer import fever_score
    pred = fever_predictions(dev, evidence)
    strict_score, acc_score, pr, rec, f1 = fever_score(pred)

    print(strict_score, acc_score, pr, rec, f1)


def run_ir(config):
    train, dev = load_paper_dataset(config['train_input'], dev=config['dev_input'])

    for split, data in [('train', train), ('dev', dev)]:
        if split == 'train':
            out_file = config['train_output']
        if split == 'dev':
            out_file = config['dev_output']

        docs, evidence = get_evidence(data, n_docs=config['n_docs'], n_sents=config['n_sents'])
        pred = to_fever_format(data, docs, evidence)

        with open(out_file, 'w') as w:
            for example in pred:
                w.write(json.dumps(example, cls=NpEncoder) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('save ir results')
    parser.add_argument('--train_input')
    parser.add_argument('--dev_input')
    parser.add_argument('--train_output')
    parser.add_argument('--dev_output')
    parser.add_argument('--n_docs', type=int, default=5, help='how many documents to retrieve')
    parser.add_argument('--n_sents', type=int, default=5, help='how many setences to retrieve')
    args = parser.parse_args()
    print(args)

    train, dev = load_paper_dataset(train=args.train_input, dev=args.dev_input)
    # train, dev = load_split_trainset(9999)
    for split, data in [('train', train), ('dev', dev)]:
        if split == 'train':
            out_file = args.train_output
        if split == 'dev':
            out_file = args.dev_output
        if os.path.exists(out_file):
            print('file {} exists. skipping ir...'.format(out_file))
            continue

        docs, evidence = get_evidence(data, n_docs=args.n_docs, n_sents=args.n_sents)
        pred = to_fever_format(data, docs, evidence)

        with open(out_file, 'w') as w:
            for example in pred:
                w.write(json.dumps(example) + '\n')
