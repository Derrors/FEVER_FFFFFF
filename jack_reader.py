# _*_ coding: utf-8 _*_

import argparse

from tqdm import tqdm

from converter import convert_label, titles_to_jsonl_num
from fever_io import (get_evidence_sentence_list, load_doclines, read_jsonl,
                      save_jsonl)
from jack import readers
from jack.core import QASetting
from util import abs_path


def read_ir_result(path, n_sentences=5):
    '''
    读取句子检索的结果
    '''
    short_evidences_counter = 0

    instances = read_jsonl(path)
    for instance in instances:
        if len(instance['predicted_sentences']) < n_sentences:
            short_evidences_counter += 1
        instance['predicted_sentences'] = instance['predicted_sentences'][: n_sentences]        # 只保留前 n 个句子
    print('short_evidences: {} / {}'.format(short_evidences_counter, len(instances)))

    t2jnum = titles_to_jsonl_num(wikipedia_dir=abs_path('./data/wiki-pages/wiki-pages/'), doctitles=abs_path('./data/doctitles'))

    titles = list()
    # 获取所有标题的列表
    for instance in instances:
        titles.extend([title for title, _ in instance['predicted_sentences']])

    t2l2s = load_doclines(titles, t2jnum)

    # 证据语句
    for instance in instances:
        instance['evidence'] = get_evidence_sentence_list(instance['predicted_sentences'], t2l2s)

    return instances


def reshape(preds_list, preds_length):
    '''
    >> preds_list = [obj, obj, obj, obj, obj, obj]
    >> preds_length = [3, 1, 2]
    >> reshape(preds_list, preds_length)
    [[obj, obj, obj], [obj], [obj, obj]]
    '''
    reshaped = list()
    pointer = 0

    for length in preds_length:
        preds = preds_list[pointer: pointer + length]
        pointer += length
        reshaped.append(preds)

    return reshaped


def flatten(_2d_list):
    flattened = list()

    for _list in _2d_list:
        flattened.extend(_list)

    return flattened


def predict(reader, all_settings, batch_size):
    preds_list = list()

    for pointer in tqdm(range(0, len(all_settings), batch_size)):
        batch_settings = all_settings[pointer: pointer + batch_size]
        n_settings = [len(settings) for settings in batch_settings]
        preds_list.extend(reshape(reader(flatten(batch_settings)), n_settings))

    return preds_list


def save_predictions(instances, preds_list, path, scores_for_all_candidates=True):
    '''
    保存预测的结果
    '''
    store = list()
    assert len(instances) == len(preds_list)

    for instance, preds in zip(instances, preds_list):
        cid = instance['id']
        claim = instance['claim']
        pred_sents = instance['evidence']

        # 取每个样例预测的所有标签及其得分
        if scores_for_all_candidates:
            pred_labels_list = [[pred.text for pred in preds_instance] for preds_instance in preds]
            scores = [[float(pred.score) for pred in preds_instance] for preds_instance in preds]
        # 取每个样例的第一个标签及得分
        else:
            pred_labels = [pred[0].text for pred in preds]
            scores = [float(pred[0].score) for pred in preds]

        # 保存为字典形式
        dic = {
            'id': cid,
            'scores': scores,
            'claim': claim,
            'predicted_sentences': pred_sents
        }

        if 'label' in instance:
            dic['label'] = instance['label']

        if scores_for_all_candidates:
            dic['predicted_labels'] = [[convert_label(pred_label, inverse=True) for pred_label in pred_labels] for pred_labels in pred_labels_list],
        else:
            dic['predicted_labels'] = [convert_label(pred_label, inverse=True) for pred_label in pred_labels]

        # scores of ir part
        if 'scored_sentences' in instance:
            dic['ev_scores'] = instance['scored_sentences']

        store.append(dic)
    save_jsonl(store, path)


def run_rte(config):
    reader = readers.reader_from_file(config['saved_reader'], dropout=0.0)

    for in_file, out_file in [('train_input_file', 'train_predicted_labels_and_scores'), ('dev_input_file', 'dev_predicted_labels_and_scores'), ('test_input_file', 'test_predicted_labels_and_scores')]:
        all_settings = list()
        instances = read_ir_result(config['in_file'], n_sentences=config['n_sentences'])

        for instance in instances:
            evidence_list = instance['evidence']
            claim = instance['claim']
            settings = [QASetting(question=claim, support=[evidence]) for evidence in evidence_list]
            all_settings.append(settings)

        preds_list = predict(reader, all_settings, config['batch_size'])
        save_predictions(instances, preds_list, path=config['out_file'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser('read claim/evidence and output verdict')
    parser.add_argument('in_file', help='input file path for rte (e.g., dev.sentences.p5.s5.jsonl)')
    parser.add_argument('save_preds', help='specify file name to save prediction')
    parser.add_argument('--saved_reader', help='path to saved reader directory')
    parser.add_argument('--only_use_topev', action='store_true', help='only use top evidence for prediction')
    parser.add_argument('--n_sentences', type=int, default=5, help='how many sentences to read for prediction')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for inference')
    args = parser.parse_args()

    print('loading reader from file:', args.saved_reader)
    dam_reader = readers.reader_from_file(args.saved_reader, dropout=0.0)

    results = list()
    preds_length = list()
    all_settings = list()

    instances = read_ir_result(args.in_file, n_sentences=args.n_sentences)

    for instance in instances:
        evidence_list = instance['evidence']
        claim = instance['claim']
        settings = [QASetting(question=claim, support=[evidence]) for evidence in evidence_list]
        all_settings.append(settings)

    preds_list = predict(dam_reader, all_settings, args.batch_size)
    save_predictions(instances, preds_list, path=args.save_preds)
