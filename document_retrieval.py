# _*_ coding: utf-8 _*_

import os
import pickle
from collections import Counter
from random import random, shuffle

import numpy as np
from nltk import word_tokenize
from nltk.corpus import gazetteers, names
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from util import edict, load_stoplist, normalize_title, pdict
from fever_io import load_paper_dataset, titles_to_jsonl_num

places = set(gazetteers.words())                # 地名
people = set(names.words())                     # 人名
stop = load_stoplist()                          # 停用词


def title_edict(t2jnum={}):
    '''
    建立文档标题的字典
    '''
    edocs = edict()
    for title in t2jnum:
        _title = normalize_title(title)
        if len(_title) > 0:
            if edocs[_title][0] is None:
                edocs[_title] = []
            edocs[_title][0].append(title)
    return edocs


def find_titles_in_claim(claim='', edocs=edict()):
    '''
    在 claim 中寻找标题短语及其在 claim 中的位置
    '''
    find = pdict(edocs)
    docset = {}
    # 对 claim 进行分词
    ctoks = word_tokenize(claim)

    for word in ctoks:
        for dlist, phrase, start in find[word]:
            for d in dlist:
                if d not in docset:
                    docset[d] = []
                docset[d].append((phrase, start))
    return docset


def phrase_features(phrase='', start=0, title='', claim=''):
    '''
    构建句子的特征字典：

    参数：
    title: 文档标题
    phrase: claim 中的短语
    claim: 对比的声明
    start: phrase 在 claim 中的位置

    返回值：
    features: 句子的特征字典
    '''
    features = dict()                                   # 特征字典
    stoks = phrase.split()                              # 分词
    _, rmndr = normalize_title(title, rflag=True)       # 标准化并分割标题

    features['rmndr'] = (rmndr == '')                           # True: 不存在潜在信息：(xxx)
    features['rinc'] = ((rmndr != '') and (rmndr in claim))     # True: 存在潜在信息：(xxx)且 xxx 在 claim 存在
    features['start'] = start                                   # 在 claim 中标题的位置
    features['start0'] = (start == 0)                           # 在 claim 首部
    features['lend'] = len(stoks)                               # 词数
    features['lend1'] = (features['lend'] == 1)                 # True: 只有一个单词
    features['cap1'] = stoks[0][0].isupper()                    # True: 第一个单词是首字母是大写
    features['stop1'] = (stoks[0].lower() in stop)              # True：第一个单词是停用词
    features['people1'] = (stoks[0] in people)                  # True：第一个单词是人名
    features['places1'] = (stoks[0] in places)                  # True：第一个单词是地名
    features['capany'] = False                                  # True：包含首字母大写的单词
    features['capall'] = True                                   # True：每个单词的首字母都是大写
    features['stopany'] = False                                 # True：存在停用词
    features['stopall'] = True                                  # True：所有词都为停用词
    features['peopleany'] = False                               # True：存在人名
    features['peopleall'] = True                                # True：所有词都为人名
    features['placesany'] = False                               # True：存在地名
    features['placesall'] = True                                # True：所有词都为地名

    for tok in stoks:
        features['capany'] = (features['capany'] or tok[0].isupper())
        features['capall'] = (features['capall'] and tok[0].isupper())
        features['stopany'] = (features['stopany'] or tok.lower() in stop)
        features['stopall'] = (features['stopall'] and tok.lower() in stop)
        features['peopleany'] = (features['peopleany'] or tok in people)
        features['peopleall'] = (features['peopleall'] and tok in people)
        features['placesany'] = (features['placesany'] or tok in places)
        features['placesall'] = (features['placesall'] and tok in places)

    return features


def score_title(ps_list=[], title='dummy', claim='dummy', model=None):
    '''
    对文档标题进行评分：取句子得分最高的作为文档的得分

    参数：
    ps_list: 和标题相关的一系列句子
    title: 在 claim 中出现的文档标题

    返回值：
    maxscore：取句子中得分最高的作为文档的得分
    '''
    maxscore = -1000000

    for phrase, start in ps_list:
        score = model.score_instance(phrase, start, title, claim)
        maxscore = max(maxscore, score)
    return maxscore


def best_titles(claim='', edocs=edict(), best=5, model=None):
    '''
    计算在 claim 中出现的得分最高（前 best 个）的文档
    '''
    # 在 claim 中寻找文档标题和相关句子
    t2phrases = find_titles_in_claim(claim, edocs)
    tscores = list()

    # 对每个标题打分
    for title in t2phrases:
        tscores.append((title, score_title(t2phrases[title], title, claim, model)))

    # 对得分进行排序并取前 best 个
    tscores = sorted(tscores, key=lambda x: -1 * x[1])[:best]

    return tscores


def title_hits(data=list(), tscores=dict()):
    '''
    对文档检索阶段的结果进行评估
    '''
    hits = Counter()
    returned = Counter()
    full = Counter()

    for example in data:
        cid = example['id']
        _ = example['claim']
        label = example['label']

        # 对 NEI 类不做处理
        if label == 'NOT ENOUGH INFO':
            continue

        all_evidence = [evi for evi_set in example['evidence'] for evi in evi_set]

        # 建立证据相关的文档的集合
        docs = set()
        for evi in all_evidence:
            evi_doc = evi[2]                        # evi: [31205, 37902, 'Peggy_Sue_Got_Married', 0]
            if evi_doc is not None:
                docs.add(evi_doc)

        # 构建证据文档 -> sid, sid -> 证据文档的集合
        e2s = dict()
        evi_sets = dict()
        sid = 0

        for s in example['evidence']:                       # s 包含多条 evi
            evi_sets[sid] = set()                           # 证据的集合，每个证据集对应一个 sid 编号，可根据 sid 查找一个证据集
            for evi in s:
                evi_sets[sid].add(evi[2])

                if evi[2] not in e2s:
                    e2s[evi[2]] = set()                     # 证据集合编号的集合，可根据证据来查找证据所在的证据集编号

                e2s[evi[2]].add(sid)
            sid = sid + 1

        for i, (pre_doc,
                _) in enumerate(tscores[cid]):              # prd_doc 为检索到的文档， score 为对应的评分
            hits[i] = hits[i] + 1 * (pre_doc in docs)       # hits 记录正确定位到证据所在文档的个数
            returned[i] = returned[i] + 1                   # returned 记录返回的文档数
            flag = 0

            if pre_doc in e2s:                              # 检索到的文档是否在证据集里
                for sid in e2s[pre_doc]:                    # 根据 pre_doc 查找对应证据集的 sid
                    s = evi_sets[sid]                       # 再根据 sid 返回对应的证据集
                    if pre_doc in s:
                        if len(s) == 1:
                            flag = 1                        # flag=1 标记着证据全部被检索到
                        s.remove(pre_doc)
            full[i] += flag                                 # full 记录检索到完整证据的文档数

            if flag == 1:
                break
    print()

    denom = returned[0]

    # 输出文档检索的结果
    for i in range(0, len(hits)):
        print(i, hits[i], returned[i], full[i] / denom)
        full[i + 1] += full[i]


def doc_ret(data=list(), edocs=edict(), best=5, model=None):
    '''
    对每个 claim 返回 best 个得分最高的文档标题
    '''
    docs = dict()
    for example in tqdm(data):
        tscores = best_titles(example['claim'], edocs, best, model)
        docs[example['id']] = tscores
    return docs


class doc_ret_model():
    def __init__(self, phrase_features=phrase_features):
        # doc_ret 使用逻辑回归 (Logistic Regression) 模型
        self.model = LogisticRegression(C=100000000, solver='sag', max_iter=100000)
        feature_list = sorted(
            list(phrase_features('dummy', 0, 'dummy',
                                 'dummy').keys()))              # 所有特征的列表
        self.f2v = {f: i for i, f in enumerate(feature_list)}   # 对每个特征进行编号

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
        features = phrase_features(phrase, start, title, claim)  # 获取语句的特征
        for f in features:
            array[obsnum, self.f2v[f]] = float(features[f])  # 将语句的特征转换为特征向量

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
        ndim = len(self.f2v)  # 特征变量维度
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


def sample_docs(train, save_file):
    '''
    在训练数据中进行采样

    返回值：
    seleted[cid][yn] = [title, phrase, start]
    '''
    samp_size = 25000
    tots = {'SUPPORTS': 0, 'REFUTES': 0}
    sofar = {'SUPPORTS': 0, 'REFUTES': 0}

    # 读取文档标题字典
    if os.path.exists('data/preprocessed_data/edocs.bin'):
        with open('data/preprocessed_data/edocs.bin', 'rb') as rb:
            edocs = pickle.load(rb)
    else:
        t2jnum = titles_to_jsonl_num()
        edocs = title_edict(t2jnum)
        with open('data/preprocessed_data/edocs.bin', 'wb') as wb:
            pickle.dump(edocs, wb)

    id2titles = dict()

    for example in train:
        cid = example['id']
        label = example['label']

        if label == 'NOT ENOUGH INFO':
            continue

        # 构建训练集中的证据对应的文档集
        all_evidence = [evi for evi_set in example['evidence'] for evi in evi_set]
        docs = set()
        for evi in all_evidence:
            evi_doc = evi[2]
            if evi_doc is not None:
                docs.add(evi_doc)

        # 将 claim 中存在的 title 转换为对应的标题短语
        t2phrases = find_titles_in_claim(example['claim'], edocs)

        id2titles[cid] = t2phrases
        flag = False
        for title in t2phrases:
            if title in docs:
                flag = True

        # 如果 claim 中出现的 title 存在于证据对应的文档集中，即 claim 可通过其中出现的 title 来获取对应的证据
        if flag:
            tots[label] += 1

    selected = dict()

    # 进行采样选择数据
    for example in tqdm(train):
        yn = 0                      # yn 表示类型，1：正样本，0：负样本
        cid = example['id']
        label = example['label']

        if label == 'NOT ENOUGH INFO':
            continue

        all_evidence = [evi for evi_set in example['evidence'] for evi in evi_set]
        docs = set()
        for evi in all_evidence:
            evi_doc = evi[2]
            if evi_doc is not None:
                docs.add(evi_doc)

        t2phrases = id2titles[cid]  # 通过实例的 cid 来直接获取对应的 title 字典
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
    with open(save_file, 'w') as w:
        for cid in selected:
            for yn in selected[cid]:
                [t, p, s] = selected[cid][yn]
                w.write(str(cid) + '\t' + str(yn) + '\t' + t + '\t' + p + '\t' + str(s) + '\n')

    for label in sofar:
        print(label, sofar[label])

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
            t = fields[2]
            p = fields[3]
            s = int(fields[4])

            if cid not in selected:
                selected[cid] = dict()
            selected[cid][yn] = [t, p, s]

    return selected


def run_doc_ret(config):
    train, dev = load_paper_dataset()

    if os.path.exists(config['doc_ret_model']):
        with open(config['doc_ret_model'], 'rb') as rb:
            model = pickle.load(rb)
    else:
        if os.path.exists(config['doc_ret_docs']):
            selected = load_selected(config['doc_ret_docs'])
        else:
            selected = sample_docs(train, config['doc_ret_docs'])

        # 建立模型
        model = doc_ret_model()
        # 对训练数据进行预处理
        X, y = model.process_train(selected, train)
        # 训练模型
        model.fit(X, y)
        # 存储训练好的模型
        with open(config['doc_ret_model'], 'wb') as wb:
            pickle.dump(model, wb)

    if os.path.exists('data/preprocessed_data/edocs.bin'):
        with open('data/preprocessed_data/edocs.bin', 'rb') as rb:
            edocs = pickle.load(rb)
    else:
        t2jnum = titles_to_jsonl_num()
        edocs = title_edict(t2jnum)
        with open('data/preprocessed_data/edocs.bin', 'wb') as wb:
            pickle.dump(edocs, wb)

    print(len(model.f2v))
    # 使用训练好的模型对验证集进行文档检索
    docs = doc_ret(dev, edocs, best=config['n_best'], model=model)
    # 对检索结果进行评估
    title_hits(dev, docs)


if __name__ == '__main__':
    config = {
        'n_best': 5,
        'edocs_path': '/home/qhli/FEVER/FEVER_FFFFF/fever/data/preprocessed_data/edocs.bin',
        'doc_ret_docs': '/home/qhli/FEVER/FEVER_FFFFF/fever/results/doc_ret/doc_ret_docs1',
        'doc_ret_model': '/home/qhli/FEVER/FEVER_FFFFF/fever/results/doc_ret/doc_ret_model1.bin'
    }
    run_doc_ret(config)
