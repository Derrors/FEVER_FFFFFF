# _*_ coding: utf-8 _*_

import argparse
import pickle
from collections import Counter
from random import random, shuffle

import numpy as np
from nltk import word_tokenize
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from util import load_stoplist, normalize_title

from document_retrieval import doc_ret, doc_ret_model, title_edict
from fever_io import (load_doc_lines, load_paper_dataset, load_split_trainset,
                      titles_to_jsonl_num)

stop = load_stoplist()


def div(x, y):
    if y == 0:
        return 1.0
    else:
        return x / y


def line_features(c_toks=set(), title='', t_toks=set(), line='', l_toks=set(), lid=0, score=0):
    '''
    构建句子的特征字典：

    参数：
    c_toks: claim 分词后词语的集合
    title：标题
    t_toks: title 分词后词语的集合
    line: 句子
    l_toks: line 分词后词语的集合

    返回值：
    features: 句子的特征字典
    '''
    features = dict()
    features['lenl'] = len(l_toks)                                  # line 长度
    features['tinl'] = (title in line)                              # 标题是否在 line 中出现
    features['lid'] = lid                                           # line 的 id
    features['lid0'] = (lid == 0)                                   # line 的 id 是否为 0
    features['score'] = score                                       # 文档检索的分数

    cns_toks = c_toks - stop                                        # claim 除去停用词
    cnt_toks = c_toks - t_toks                                      # claim 除去标题词
    cntns_toks = cns_toks - t_toks                                  # claim 除去停用词和标题词

    lnt_toks = l_toks - t_toks                                      # line 除去标题词
    lns_toks = l_toks - stop                                        # line 除去停用词
    lntns_toks = lns_toks - t_toks                                  # line 除去标题词和停用词

    cl_toks = c_toks & l_toks                                       # claim 和 line 都有的词
    clnt_toks = cnt_toks & lnt_toks                                 # 除标题词外, claim 和 line 都有的词
    clns_toks = cns_toks & lns_toks                                 # 除停用词外, claim 和 line 都有的词
    clntns_toks = cntns_toks & lntns_toks                           # 除标题词和停用词外, claim 和 line 都有的词

    features['pc'] = div(len(cl_toks), len(c_toks))                 # cl_toks / c_toks
    features['pl'] = div(len(cl_toks), len(l_toks))                 # cl_toks / l_toks
    features['pcns'] = div(len(clns_toks), len(cns_toks))           # clns_toks / cns_toks
    features['plns'] = div(len(clns_toks), len(lns_toks))           # clns_toks / lns_toks
    features['pcnt'] = div(len(clnt_toks), len(cnt_toks))           # clnt_toks / cnt_toks
    features['plnt'] = div(len(clnt_toks), len(lnt_toks))           # clnt_toks / lnt_toks
    features['pcntns'] = div(len(clntns_toks), len(cntns_toks))     # clntns_toks / cntns_toks
    features['plntns'] = div(len(clntns_toks), len(lntns_toks))     # clntns_toks / lntns_toks

    return features


def best_lines(claim='', tscores=list(), lines=dict(), best=5, model=None):
    '''
    计算在得分最高（前 best 个）的 line
    '''

    lscores = list()
    c_toks = set(word_tokenize(claim.lower()))

    for title, tscore in tscores:
        t_toks = normalize_title(title)                         # 对 title 进行处理
        t = ' '.join(t_toks)
        t_toks = set(t_toks)
        for lid in lines[title]:                                # 获取标题对应句子的行号
            line = lines[title][lid]
            l_toks = set(word_tokenize(line.lower()))
            if len(l_toks) > 0:
                lscores.append((title, lid, model.score_instance(c_toks, t, t_toks, line, l_toks, lid, tscore)))
    lscores = sorted(lscores, key=lambda x: -1 * x[2])[: best]
    return lscores


def line_hits(data=list(), evidence=dict()):
    '''
    评估检索结果
    '''
    hits = Counter()
    returned = Counter()
    full = Counter()

    for example in data:
        cid = example['id']
        label = example['label']

        if label == 'NOT ENOUGH INFO':
            continue

        all_evidence = [evi for evi_set in example['evidence'] for evi in evi_set]
        lines = dict()                                      # doc -> evi_line
        for evi in all_evidence:
            evi_d = evi[2]
            evi_line = evi[3]
            if evi_d is not None:
                if evi_d not in lines:
                    lines[evi_d] = set()
                lines[evi_d].add(evi_line)

        e2s = dict()                                        # evi -> sid
        evsets = dict()                                     # sid -> eviset
        sid = 0
        for s in example['evidence']:
            evsets[sid] = set()
            for e in s:
                evsets[sid].add((e[2], e[3]))
                if (e[2], e[3]) not in e2s:
                    e2s[(e[2], e[3])] = set()
                e2s[(e[2], e[3])].add(sid)
            sid = sid + 1

        for i, (d, l, s) in enumerate(evidence[cid]):       # d: doc, l: lid
            hits[i] = hits[i] + 1 * (d in lines and l in lines[d])
            returned[i] = returned[i] + 1
            flag = 0
            if (d, l) in e2s:
                for sid in e2s[(d, l)]:
                    s = evsets[sid]
                    if (d, l) in s:
                        if len(s) == 1:
                            flag = 1
                        s.remove((d, l))
            full[i] += flag
            if flag == 1:
                break
    print()

    denom = returned[0]
    for i in range(0, len(hits)):
        print(i, hits[i], returned[i], full[i] / denom)
        full[i + 1] += full[i]


def sent_ret(data=list(), docs=dict(), lines=dict(), best=5, model=None):
    '''
    根据 claim 返回 best 个最适合的句子
    '''
    evidence = dict()

    for example in tqdm(data):
        cid = example['id']
        evidence[cid] = list()
        tscores = docs[cid]
        claim = example['claim']
        evidence[cid] = best_lines(claim, tscores, lines, best, model)

    return evidence


class sent_ret_model():
    def __init__(self, line_features=line_features):
        # sent_ret 使用逻辑回归 (Logistic Regression) 模型
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


def select_lines(docs, t2jnum, train, save_file):
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

    with open(save_file, 'w') as w:
        for cid in selected:
            for yn in selected[cid]:
                [title, l_id, l_txt, score] = selected[cid][yn]
                w.write(str(cid) + '\t' + str(yn) + '\t' + title + '\t' + str(l_id) + '\t' + str(l_txt) + '\t' + str(score) + '\n')

    for l in sofar:
        print(l, sofar[l])

    return selected


def load_selected(fname):
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


def run_sent_ret(config):
    train, dev = load_paper_dataset()
    # train, dev = load_split_trainset(9999)

    with open('data/preprocessed_data/edocs.bin', 'rb') as rb:
        edocs = pickle.load(rb)

    with open(config['doc_ret_model'], 'rb') as rb:
        dmodel = pickle.load(rb)

    t2jnum = titles_to_jsonl_num()

    try:
        with open(config['sent_ret_model'], 'rb') as rb:
            model = pickle.load(rb)                             # 加载模型参数
    except BaseException:
        try:
            selected = load_selected(config['sent_ret_line'])   # 加载采样数据
        except BaseException:
            docs = doc_ret(train, edocs, model=dmodel)
            selected = select_lines(docs, t2jnum, train, config['sent_ret_line'])

        model = sent_ret_model()
        X, y = model.process_train(selected, train)             # 训练模型
        model.fit(X, y)

        with open(config['sent_ret_model'], 'wb') as wb:
            pickle.dump(model, wb)

    docs = doc_ret(dev, edocs, model=dmodel)                     # 进行文档检索
    lines = load_doc_lines(docs, t2jnum)
    evidence = sent_ret(dev, docs, lines, best=config['n_best'], model=model)       # 进行句子检索
    line_hits(dev, evidence)                                    # 评估结果


if __name__ == '__main__':
    config = {}
    run_sent_ret(config)
