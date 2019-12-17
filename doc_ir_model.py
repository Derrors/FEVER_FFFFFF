# _*_ coding: utf-8 _*_

import os
import pickle
import argparse
import numpy as np

from tqdm import tqdm
from random import random, shuffle
from collections import Counter
from nltk.corpus import gazetteers, names
from nltk import word_tokenize, sent_tokenize
from sklearn.linear_model import LogisticRegression

from doc_ir import *
from util import edict, pdict, normalize_title, load_stoplist
from fever_io import titles_to_jsonl_num, load_split_trainset, load_paper_dataset


class doc_ir_model:
    def __init__(self, phrase_features=phrase_features):
        # doc_ir 使用逻辑回归 (Logistic Regression) 模型
        self.model = LogisticRegression(C=100000000, solver='sag', max_iter=100000)
        feature_list = sorted(list(phrase_features('dummy', 0, 'dummy', 'dummy').keys()))       # 所有特征的列表
        self.f2v = {f: i for i, f in enumerate(feature_list)}                                   # 对每个特征进行编号

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
    
    def process_instance(self, phrase='dummy', start=0, title='dummy', claim='dummy', obsnum=0, array=np.zeros(shape=(1, 1)), dtype=np.float32):
        '''
        将实例转换为对应的特征向量
        '''
        features = phrase_features(phrase, start, title, claim)         # 获取语句的特征
        for f in features:
            array[obsnum, self.f2v[f]] = float(features[f])             # 将语句的特征转换为特征向量

    def score_instance(self, phrase='dummy', start=0, title='dummy', claim='dummy'):
        '''
        使用训练好的模型来预测得分
        '''
        x = np.zeros(shape=(1, len(self.f2v)), dtype=np.float32)
        self.process_instance(phrase, start, title, claim, 0, x)
        return self.prob(x)

    def process_train(self, selected, train):
        '''
        训练数据预处理
        '''
        obs = len(selected) * 2
        ndim = len(self.f2v)                                            # 特征变量维度
        X = np.zeros(shape=(obs, ndim), dtype=np.float32)
        y = np.zeros(shape=(obs), dtype=np.float32)
        obsnum = 0
        for example in tqdm(train):
            cid = example['id']
            if cid in selected:
                claim = example['claim']
                for yn in selected[cid]:
                    [title, phrase, start] = selected[cid][yn]
                    self.process_instance(phrase, start, title, claim, obsnum, X)
                    y[obsnum] = float(yn)
                    obsnum += 1
        assert obsnum == obs
        return X, y 
        

def count_labels(train):
    '''
    在选择的文档中统计标签数量
    '''
    supports = 0
    refutes = 0

    print('counting labels...')

    for instance in tqdm(train):
        if instance['label'] == 'NOT ENOUGH INFO':
            continue
        if instance['label'] == 'SUPPORTS':
            supports += 1
        else:
            refutes += 1

    counts = {'SUPPORTS': supports, 'REFUTES': refutes}
    print('result:', counts)
    return counts


def select_docs(train):
    '''
    在训练数据中进行采样负样本

    返回值：
    seleted[cid][yn] = [title, phrase, start]
    '''
    samp_size = 25000
    tots = {'SUPPORTS': 0, 'REFUTES': 0}
    sofar = {'SUPPORTS': 0, 'REFUTES': 0}

    # 读取文档标题字典
    try:
        with open('./data/edocs.bin','rb') as rb:
            if os.path.getsize('./data/edocs.bin') < 100:
                raise RuntimeError('Size of edocs.bin is too small. It may be an empty file.')
            edocs = pickle.load(rb)
    except:
        t2jnum = titles_to_jsonl_num()
        edocs = title_edict(t2jnum)
        with open('./data/edocs.bin','wb') as wb:
            pickle.dump(edocs, wb)

    examples = Counter()
    id2titles = dict()

    for example in train:
        cid = example['id']
        claim = example['claim']
        label = example['label']

        if label == 'NOT ENOUGH INFO':
            continue
        
        # 构建训练集中的证据对应的文档集
        all_evidence = [evi for evi_set in example['evidence'] for evi in evi_set]
        docs = set()
        for evi in all_evidence:
            evi_doc = evi[2]
            if evi_doc != None:
                docs.add(evi_doc)

        # 将 claim 中存在的 title 转换为对应的句子
        t2phrases = find_titles_in_claim(example['claim'], edocs)
        id2titles[cid] = t2phrases
        flag = False
        for title in t2phrases:
            if title in docs:
                flag = True

        # 如果 claim 中出现的 title 全部存在于证据对应的文档集中，即 claim 可通过其中出现的 title 来获取对应的证据
        if flag:
            tots[l] += 1

    selected = dict()

    # 进行采样选择数据
    for example in tqdm(train):
        yn = 0                                  # yn 表示类别标签，1：SUPPORT，0：REFUTE
        cid = example['id']
        label = example['label']

        if label == 'NOT ENOUGH INFO':
            continue

        all_evidence = [evi for evi_set in example['evidence'] for evi in evi_set]
        docs = set()
        for evi in all_evidence:
            evi_doc = evi[2]
            if evi_doc != None:
                docs.add(evi_doc)

        t2phrases = id2titles[cid]              # 通过实例的 cid 来直接获取对应的 title 字典
        for title in t2phrases:
            if title in docs:
                yn = 1

        prob = (samp_size - sofar[label]) / (tots[label])

        if yn == 1 and random() < prob:
            titles = list(t2phrases.keys())
            shuffle(titles)
            flagy = False
            flagn = False

            for t in titles:
                if not flagy and t in docs:
                    ty = t
                    flagy = True
                if not flagn and t not in docs:
                    tn = t
                    flagn = True
                if flagy and flagn:
                    selected[cid] = dict()
                    for t, y_n in [(ty, 1), (tn, 0)]:
                        ps = t2phrases[t]
                        shuffle(ps)
                        p, s = ps[0]
                        selected[cid][y_n] = [t, p, s]
                    sofar[label] += 1
                    break

        if yn == 1:
            tots[label] -= 1
    
    # 将采样结果写入文件
    with open('./data/doc_ir_docs', 'w') as w:
        for cid in selected:
            for yn in selected[cid]:
                [t, p, s] = selected[cid][yn]
                w.write(str(cid) + '\t' + str(yn) + '\t' + t + '\t' + p + '\t' + str(s) + '\n')

    for label in sofar:
        print(label, sofar[label])

    return selected


def load_selected(fname='./data/doc_ir_docs'):
    '''
    加载采样数据
    '''
    selected = dict()

    with open(fname) as f:
        for line in tqdm(f):
            fields = line.rstrip('\n').split('\t')

            cid = int(fields[0])
            yn = int(fields[1])
            t = fields[2]
            p = fields[3]
            s = int(fields[4])

            if cid not in selected:
                selected[cid] = dict()
            selected[cid][yn] = [t, p, s]

    return selected
    
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser('perform ir for document')
    parser.add_argument('--best', type=int, default=5, help='how many documents to retrieve')
    args = parser.parse_args()
    print(args)

    train, dev = load_paper_dataset()
    # train, dev = load_split_trainset(9999)

    try:
        with open('./data/doc_ir_model.bin', 'rb') as rb:
            model = pickle.load(rb)
    except:
        try:
            selected = load_selected()
        except:
            selected = select_docs(train)

        # 建立模型
        model = doc_ir_model()
        # 对训练数据进行预处理
        X, y = model.process_train(selected, train)
        # 训练模型
        model.fit(X,y)
        # 存储训练好的模型
        with open('./data/doc_ir_model.bin', 'wb') as wb:
            if os.path.getsize('./data/edocs.bin') < 100:
                raise RuntimeError('Size of edocs.bin is too small. It may be an empty file.')
            pickle.dump(model, wb)
    try:
        with open('./data/edocs.bin', 'rb') as rb:
            edocs = pickle.load(rb)
    except:
        t2jnum = titles_to_jsonl_num()
        edocs = title_edict(t2jnum)
        with open('./data/edocs.bin', 'wb') as wb:
            pickle.dump(edocs, wb)

    print(len(model.f2v))
    # 使用训练好的模型对验证集进行文档检索
    docs = doc_ir(dev, edocs, best=args.best, model=model)
    # 对检索结果进行评估
    title_hits(dev, docs)
