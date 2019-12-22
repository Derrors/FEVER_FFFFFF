# _*_ coding: utf-8 _*_

'''
将 FEVER 数据集格式转换为 SNLI 数据格式
'''

import argparse
import json
import os
from collections import Counter

from tqdm import tqdm

from analyse import compare_evidences
from fever_io import (get_evidence_sentence_list, load_doclines, read_jsonl,
                      save_jsonl, titles_to_jsonl_num)
from util import abs_path

current_dir = os.path.dirname(os.path.abspath(__file__))


def convert_label(label, inverse=False):
    '''
    将 fever 标签与 snli 标签相互转换
    '''
    # fever 转 snli
    fever2snli = {
        'SUPPORTS': 'entailment',
        'REFUTES': 'contradiction',
        'NOT ENOUGH INFO': 'neutral'
    }
    # snli 转 fever
    snli2fever = {snli: fever for fever, snli in fever2snli.items()}

    if inverse:
        assert label in snli2fever
        return snli2fever[label]                # 转换为 fever 标签
    else:
        assert label in fever2snli
        return fever2snli[label]                # 转换为 snli 标签


def snli_format(id, pair_id, label, evidence, claim):
    '''
    格式化
    '''
    instance = {
        'captionID': id,
        'pairID': pair_id,
        'gold_label': label,
        'sentence1': evidence,
        'sentence2': claim
    }
    return instance


def sampling(converted_instances):
    '''
    采样：平衡样本中的类别比例
    '''
    labels = list()
    for instance in converted_instances:
        labels.append(instance['gold_label'])

    nei_num = Counter(labels)[convert_label('NOT ENOUGH INFO')]         # 'NEI' 样本的数量
    instances_num = len(converted_instances)                            # 样本总数
    samples_num = (instances_num - nei_num)                             # 非 'NEI' 样本数量

    if samples_num == 0:
        return converted_instances

    sample_count = 0
    sampled_instances = list()
    for label, instance in zip(labels, converted_instances):
        if label != convert_label('NOT ENOUGH INFO'):
            sampled_instances.append(instance)              # 'SUP'、'REF' 类别的实例
        elif sample_count < samples_num:
            sampled_instances.append(instance)              # 'NEI' 类别的实例
            sample_count += 1

    return sampled_instances


def evidence_format(evidences):
    '''
    拼接 evidence: 'sentence1 sentence2 sentence3 ...'
    '''
    return ' '.join(evidences)


def convert_instance(instance, t2l2s, use_ir_prediction, n_sentences):
    '''
    将单个实例转换为一个或多个实例
    参数：
    instance: instance of FEVER dataset.
    t2l2s: output of titles_to_jsonl_num

    返回值：
    converted_instances：list of converted instances
    '''

    converted_instances = []
    # instance['evidence']: [[[hoge, hoge, title, line_num], [hoge, hoge, title, line_num]], [[..],[..],..], ...]
    if use_ir_prediction:

        evidence_line_num = [[title, line_num] for title, line_num in instance['predicted_sentences'][: n_sentences] if title in t2l2s]
        contained_flags = compare_evidences(instance['evidence'], evidence_line_num)

        for eidx, ((title, line_num), contained) in enumerate(zip(evidence_line_num, contained_flags)):
            label = instance['label'] if (instance['label'] != 'NOT ENOUGH INFO' and contained) else 'NOT ENOUGH INFO'
            converted_instances.append(snli_format(
                id='{}-{}'.format(instance['id'], str(eidx)),
                pair_id='{}-{}'.format(instance['id'], str(eidx)),
                label=convert_label(label),
                evidence=evidence_format(get_evidence_sentence_list([(title, line_num)], t2l2s)),
                claim=instance['claim']))
        converted_instances = sampling(converted_instances)

    else:
        for eidx, evidence_set in enumerate(instance['evidence']):
            # 先将 evidences 转换为元组形式
            evidence_line_num = [(title, line_num) for _, _, title, line_num in evidence_set if title in t2l2s]
            if not evidence_line_num:
                continue
            # 转换为 SNLI 格式
            converted_instances.append(snli_format(
                id='{}-{}'.format(instance['id'], str(eidx)),
                pair_id='{}-{}'.format(instance['id'], str(eidx)),
                label=convert_label(instance['label']),
                evidence=evidence_format(get_evidence_sentence_list(evidence_line_num, t2l2s)),
                claim=instance['claim']))
    return converted_instances


def convert(instances, use_ir_prediction=False, n_sentences=5):
    '''
    将 FEVER 数据格式转换成 SNLI 数据格式
    参数:
    instances: list of dictionary of FEVER format

    返回值:
    instances: list of dictionary of jack SNLI format
    '''
    # 记录所有在 evidence 中出现过的 title
    all_titles = list()

    for instance in tqdm(instances, desc='process for NEI'):
        # 标签为 'NEI' 的实例, 对 evidences 形式进行转换
        if instance['label'] == 'NOT ENOUGH INFO':
            evidences = instance['predicted_sentences'][: n_sentences]      # evidences：[(title, line_num), (title, line_num), ...]
            evidences = [[['dummy', 'dummy', title, line_num]] for title, line_num in evidences]      # 转换格式
            instance['evidence'] = evidences
        if use_ir_prediction:
            titles = [title for title, _ in instance['predicted_sentences'][: n_sentences]]
        else:
            titles = [title for evidence_set in instance['evidence'] for _, _, title, _ in evidence_set]
        all_titles.extend(titles)

    print('loading wiki data...')

    # 建立 title 到 jsonl 文件的查找字典
    t2jnum = titles_to_jsonl_num(wikipedia_dir=abs_path('./data/wiki-pages/'), doctitles=abs_path('./data/doctitles'))
    t2l2s = load_doclines(all_titles, t2jnum)               # title -> line_id -> line_text

    converted_instances = list()
    for instance in tqdm(instances, desc='conversion'):
        converted_instances.extend(convert_instance(instance, t2l2s, use_ir_prediction=use_ir_prediction, n_sentences=n_sentences))
    return converted_instances


def run_convert(config):
    for src, tar in [['train_input', 'train_output'], ['dev_input', 'dev_output']]:
        instances = read_jsonl(config[src])
        snli_format_instances = convert(instances, use_ir_prediction=config['use_ir_pred'], n_sentences=config['n_sentences'])
        save_jsonl(snli_format_instances, config[tar], skip_if_exists=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('src')
    parser.add_argument('tar')
    parser.add_argument('--use_ir_pred', action='store_true')
    parser.add_argument('--n_sentences', default=5, type=int)
    parser.add_argument('--convert_test', action='store_true')
    # parser.add_argument('--testset', help='turn on when you convert test data', action='store_true')
    args = parser.parse_args()
    print(args)

    if args.convert_test:
        test_in = '''[{'id': 15812, 'verifiable': 'VERIFIABLE', 'label': 'REFUTES', 'claim': 'Peggy Sue Got Married is a Egyptian film released in 1986.', 'evidence': [[[31205, 37902, 'Peggy_Sue_Got_Married', 0], [31205, 37902, 'Francis_Ford_Coppola', 0]], [[31211, 37908, 'Peggy_Sue_Got_Married', 0]]], 'predicted_pages': ['Peggy_Sue_Got_Married_-LRB-musical-RRB-', 'Peggy_Sue_Got_Married_-LRB-song-RRB-', 'Peggy_Sue_Got_Married', 'Peggy_Sue', 'Peggy_Sue_-LRB-band-RRB-'], 'predicted_sentences': [['Peggy_Sue_Got_Married', 0], ['Peggy_Sue_Got_Married_-LRB-musical-RRB-', 0], ['Peggy_Sue_Got_Married_-LRB-song-RRB-', 0], ['Peggy_Sue', 0], ['Peggy_Sue_Got_Married_-LRB-musical-RRB-', 2]]}, {'id': 229289, 'verifiable': 'NOT VERIFIABLE', 'label': 'NOT ENOUGH INFO', 'claim': 'Neal Schon was named in 1954.', 'evidence': [[[273626, null, null, null]]], 'predicted_pages': ['Neal_Schon', 'Neal', 'Named', 'Was_-LRB-Not_Was-RRB-', 'Was'], 'predicted_sentences': [['Neal_Schon', 0], ['Neal_Schon', 6], ['Neal_Schon', 5], ['Neal_Schon', 1], ['Neal_Schon', 2]]}, {'id': 15711, 'verifiable': 'VERIFIABLE', 'label': 'SUPPORTS', 'claim': 'Liverpool F.C. was valued at $1.55 billion at one point.', 'evidence': [[[31112, 37788, 'Liverpool_F.C.', 11]]], 'predicted_pages': ['Liverpool_F.C.', 'Liverpool_F.C._-LRB-Montevideo-RRB-', 'Liverpool_F.C._-LRB-Superleague_Formula_team-RRB-', 'Liverpool_F.C._-LRB-disambiguation-RRB-', 'Liverpool'], 'predicted_sentences': [['Liverpool_F.C.', 11], ['Liverpool', 0], ['Liverpool', 9], ['Liverpool', 10], ['Liverpool', 8]]}]'''

        print('input:\n', test_in)
        fever_format = json.loads(test_in)
        snli_format_instances = convert(fever_format, use_ir_prediction=args.use_ir_pred, n_sentences=args.n_sentences)
        print('\noutput:\n', json.dumps(snli_format_instances, indent=4))

    else:
        if os.path.exists(args.tar):
            print('WARNING: file {} alreadly exists'.format(args.tar))

        instances = read_jsonl(args.src)
        snli_format_instances = convert(instances, use_ir_prediction=args.use_ir_pred, n_sentences=args.n_sentences)
        save_jsonl(snli_format_instances, args.tar, skip_if_exists=True)
