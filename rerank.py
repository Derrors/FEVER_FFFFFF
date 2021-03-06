# _*_ coding: utf-8 _*_

import json

import numpy as np


def load_jsonl(path, key=None):
    out = list()
    if key is None:
        with open(path, 'r') as f:
            for line in f:
                out.append(json.loads(line))
    else:
        with open(path, 'r') as f:
            for line in f:
                out.append(json.loads(line)[key])
    return out


def run_rerank(config):
    train_and_dev = [
        ('train_rte_predictions', 'train_aggregated_labels', 'train_predicted_evidence', 'train_reranked_evidence'),
        ('dev_rte_predictions', 'dev_aggregated_labels', 'dev_predicted_evidence', 'dev_reranked_evidence')
    ]
    for (rte, aggregate, predict, rerank) in train_and_dev:
        rte_predictions = load_jsonl(config[rte], key='predicted_labels')
        aggregated_labels = load_jsonl(config[aggregate], key='predicted')
        predicted_evidences = load_jsonl(config[predict], key='predicted_sentences')
        ids = load_jsonl(config[predict], key='id')

        predictions = []

        for (id, ev, rte_labels, aggr_label) in zip(ids, predicted_evidences, rte_predictions, aggregated_labels):
            if len(rte_labels) == 1:
                rte_labels = rte_labels[0]
            predictions.append({'id': id, 'rte_preds': rte_labels, 'predicted_label': aggr_label, 'predicted_evidence': ev[: config['n_sentences']]})

        out_preds = list()
        for pred in predictions:
            if len(pred['rte_preds']) != len(pred['predicted_evidence']):
                pred['rte_preds'] = pred['rte_preds'][: len(pred['predicted_evidence'])]

            # no reranking if num of rte preds are lower than 5
            if len(pred['rte_preds']) > 5 and pred['predicted_label'] != 'NOT ENOUGH INFO':
                correct_ev_flags = (pred['predicted_label'] == np.array(pred['rte_preds']))
                correct_ev_args = np.reshape(np.argwhere(correct_ev_flags == True), (-1))
                incorrect_ev_args = np.reshape(np.argwhere(correct_ev_flags == False), (-1))

                correct_evs = [pred['predicted_evidence'][idx] for idx in correct_ev_args]
                incorrect_evs = [pred['predicted_evidence'][idx] for idx in incorrect_ev_args]
                out_ev = (correct_evs + incorrect_evs)
            else:
                out_ev = pred['predicted_evidence']

            out_dict = {
                'id': pred['id'],
                'predicted_sentences': out_ev
            }
            out_preds.append(out_dict)

        with open(config[rerank], 'w') as f:
            for out_pred in out_preds:
                f.write(json.dumps(out_pred) + '\n')


if __name__ == '__main__':
    config = {}
    run_rerank(config)
