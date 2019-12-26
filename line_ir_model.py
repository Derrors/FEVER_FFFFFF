# _*_ coding: utf-8 _*_

import argparse
import pickle
from collections import Counter
from random import random, shuffle

import numpy as np
from nltk import word_tokenize
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from doc_ir import doc_ir
from doc_ir_model import doc_ir_model
from fever_io import load_doc_lines, load_paper_dataset, titles_to_jsonl_num
from line_ir import line_features, line_hits, line_ir
from util import normalize_title


class line_ir_model:
    def __init__(self, line_features=line_features):
        # line_ir 使用逻辑回归 (Logistic Regression) 模型
        self.model = LogisticRegression(C=100000000, solver='sag', max_iter=100000)
        featurelist = sorted(list(line_features({'dummy'}, 'dummy', {'dummy'}, 'dummy', {'dummy'}, 0, 0).keys()))       # 所有特征的列表
        self.f2v = {f: i for i, f in enumerate(featurelist)}                # 对每个特征进行编号

    def fit(self, X, y):
        '''
        根据训练数据来拟合模型
        '''
        self.model.fit(X, y)

    def prob(self, x):
        '''
        使用模型进行预测
        '''
        return self.model.predict_proba(x)[0, 1]

    def process_instance(self, c_toks={'dummy'}, t='dummy', t_toks={'dummy'}, line='dummy', l_toks={'dummy'}, lid=0, tscore=0, obsnum=0, array=np.zeros(shape=(1, 1)), dtype=np.float32):
        '''
        将实例转换为对应的特征向量
        '''
        features = line_features(c_toks, t, t_toks, line, l_toks, lid, tscore)
        for f in features:
            array[obsnum, self.f2v[f]] = float(features[f])

    def score_instance(self, c_toks={'dummy'}, t='dummy', t_toks={'dummy'}, line='dummy', l_toks={'dummy'}, lid=0, tscore=0):
        '''
        使用训练好的模型来预测得分
        '''
        x = np.zeros(shape=(1, len(self.f2v)), dtype=np.float32)
        self.process_instance(c_toks, t, t_toks, line, l_toks, lid, tscore, 0, x)
        return self.prob(x)

    def process_train(self, selected, train):
        '''
        训练数据预处理
        '''
        obs = len(selected) * 2                                 # 全体样本的数量
        nvars = len(self.f2v)                                   # 特征向量维度
        X = np.zeros(shape=(obs, nvars), dtype=np.float32)
        y = np.zeros(shape=(obs), dtype=np.float32)
        obsnum = 0

        # 将每个样本转换为特征向量
        for example in tqdm(train):
            cid = example['id']
            if cid in selected:
                claim = example['claim']
                c_toks = set(word_tokenize(claim.lower()))
                for yn in selected[cid]:
                    [title, lid, line, tscore] = selected[cid][yn]
                    t_toks = normalize_title(title)
                    t = ' '.join(t_toks)
                    t_toks = set(t_toks)
                    l_toks = set(word_tokenize(line.lower()))
                    self.process_instance(c_toks, t, t_toks, line, l_toks, lid, tscore, obsnum, X)
                    y[obsnum] = float(yn)
                    obsnum += 1

        assert obsnum == obs
        return X, y


def select_lines(docs, t2jnum, train):
    '''
    在训练数据中进行采样，并生成负样本

    返回值：
    seleted[cid][yn] = [title, l_id, l_txt, score]
    '''
    selected = dict()
    rlines = load_doc_lines(docs, t2jnum)
    samp_size = 20000                           # 采样数量

    tots = {'SUPPORTS': 0, 'REFUTES': 0}        # 全体训练集情况
    sofar = {'SUPPORTS': 0, 'REFUTES': 0}       # 记录当前采样情况

    examples = Counter()
    for example in train:
        cid = example['id']
        label = example['label']

        if label == 'NOT ENOUGH INFO':
            continue

        # 对该样本提取相关的所有证据
        all_evidence = [evi for evi_set in example['evidence'] for evi in evi_set]
        evi_set = set()                         # 该样本包含的证据文档标题的集合
        for evi in all_evidence:
            evi_d = evi[2]                      # 证据文档标题
            if evi_d is not None:
                evi_set.add(evi_d)

        flag = False                            # 标记检索得到的文档是否在样本的证据集中
        for doc, score in docs[cid]:            # docs: 文档检索的结果, doc：文档标题, score: 标题得分
            if doc in evi_set:
                flag = True
        if flag:
            tots[label] += 1                    # 记录全体训练集中文档检索正确的样本数
            examples[label] += 1

    for example in train:
        cid = example['id']
        label = example['label']

        if label == 'NOT ENOUGH INFO':
            continue

        # 对该样本提取相关的所有证据
        all_evidence = [evi for evi_set in example['evidence'] for evi in evi_set]
        lines = dict()                          # evi_d -> evi_line
        for evi in all_evidence:
            evi_d = evi[2]                      # 证据标题
            evi_line = evi[3]                   # 证据所在行号
            if evi_d is not None:
                if evi_d not in lines:
                    lines[evi_d] = set()
                lines[evi_d].add(evi_line)      # 证据信息对应的行号的集合

        flag = False                            # 标记检索到的文档是否在样本证据中
        for doc, score in docs[cid]:
            if doc in lines:
                flag = True
        if flag:
            prob = (samp_size - sofar[label]) / (tots[label])           # 目前还未采样的比例,也就是采样该样本的概率
            if random() < prob:
                ylines = list()
                nlines = list()
                for title, score in docs[cid]:
                    for l_id in rlines[title]:
                        l_txt = rlines[title][l_id]
                        if title in lines and l_id in lines[title]:
                            ylines.append([title, l_id, l_txt, score])          # 正样本
                        elif l_txt != '':
                            nlines.append([title, l_id, l_txt, score])          # 负样本
                selected[cid] = dict()
                for yn, ls in [(1, ylines), (0, nlines)]:
                    shuffle(ls)
                    selected[cid][yn] = ls[0]
                sofar[label] += 1
            tots[label] -= 1

    with open('./data/line_ir_lines', 'w') as w:
        for cid in selected:
            for yn in selected[cid]:
                [title, l_id, l_txt, score] = selected[cid][yn]
                w.write(str(cid) + '\t' + str(yn) + '\t' + title + '\t' + str(l_id) + '\t' + str(l_txt) + '\t' + str(score) + '\n')

    for l in sofar:
        print(l, sofar[l])

    return selected


def load_selected(fname='./data/line_ir_lines'):
    '''
    加载采样数据
    '''
    selected = dict()

    with open(fname) as f:
        for line in tqdm(f):
            fields = line.rstrip('\n').split('\t')
            cid = int(fields[0])
            yn = int(fields[1])
            title = fields[2]
            l_id = int(fields[3])
            l_txt = fields[4]
            score = float(fields[5])

            if cid not in selected:
                selected[cid] = dict()
            selected[cid][yn] = [title, l_id, l_txt, score]

    return selected


if __name__ == '__main__':
    parser = argparse.ArgumentParser('perform ir for sentences')
    parser.add_argument('--best', type=int, default=5, help='how many setences to retrieve')
    args = parser.parse_args()
    print(args)

    train, dev = load_paper_dataset()
    # train, dev = load_split_trainset(9999)

    with open('./data/edocs.bin', 'rb') as rb:
        edocs = pickle.load(rb)

    with open('./data/doc_ir_model.bin', 'rb') as rb:
        dmodel = pickle.load(rb)

    t2jnum = titles_to_jsonl_num()

    try:
        with open('./data/line_ir_model.bin', 'rb') as rb:
            model = pickle.load(rb)                             # 加载模型参数
    except BaseException:
        try:
            selected = load_selected()                          # 加载采样数据
        except BaseException:
            docs = doc_ir(train, edocs, model=dmodel)
            selected = select_lines(docs, t2jnum, train)

        model = line_ir_model()
        X, y = model.process_train(selected, train)             # 训练模型
        model.fit(X, y)

        with open('./data/line_ir_model.bin', 'wb') as wb:
            pickle.dump(model, wb)

    docs = doc_ir(dev, edocs, model=dmodel)                     # 进行文档检索
    lines = load_doc_lines(docs, t2jnum)
    evidence = line_ir(dev, docs, lines, best=args.best, model=model)       # 进行句子检索
    line_hits(dev, evidence)                                    # 评估结果
