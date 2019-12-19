# _*_ coding: utf-8 _*_

import pickle
import numpy as np
from tqdm import tqdm
from collections import Counter
from nltk.corpus import gazetteers, names
from nltk import word_tokenize, sent_tokenize

from util import edict, pdict, normalize_title, load_stoplist
from fever_io import titles_to_jsonl_num, load_split_trainset


places = set(gazetteers.words())        # 地名
people = set(names.words())             # 人名
stop = load_stoplist()                  # 停用词


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
    在 claim 中寻找标题及其在 claim 中的位置
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
    phrase: 语句
    claim: 对比的声明
    start: 标题在声明中的位置

    返回值：
    features: 句子的特征字典
    '''
    features = dict()                                               # 特征字典
    stoks = phrase.split()                                          # 句子分词
    t_toks, rmndr = normalize_title(title, rflag=True)              # 标准化并分割标题

    features['rmndr']     = (rmndr == '')                           # True: 不存在潜在信息：(xxx)
    features['rinc']      = ((rmndr != '') and (rmndr in claim))    # True: 存在潜在信息：(xxx)且 xxx 在 claim 存在
    features['start']     = start                                   # 在 claim 中标题的位置
    features['start0']    = (start == 0)                            # 标题在 claim 首部
    features['lend']      = len(stoks)                              # 第一条句子的词数
    features['lend1']     = (features['lend'] == 1)                 # True: 第一条句子只有一个单词
    features['cap1']      = stoks[0][0].isupper()                   # True: 第一个单词是首字母是大写
    features['stop1']     = (stoks[0].lower() in stop)              # True：第一个单词是停用词
    features['people1']   = (stoks[0] in people)                    # True：第一个单词是人名
    features['places1']   = (stoks[0] in places)                    # True：第一个单词是地名
    features['capany']    = False                                   # True：句子中包含首字母大写的单词
    features['capall']    = True                                    # True：句子中每个单词的首字母都是大写
    features['stopany']   = False                                   # True：句子中存在停用词
    features['stopall']   = True                                    # True：句子中所有词都为停用词
    features['peopleany'] = False                                   # True：句子中存在人名
    features['peopleall'] = True                                    # True：句子中所有词都为人名
    features['placesany'] = False                                   # True：句子中存在地名
    features['placesall'] = True                                    # True：句子中所有词都为地名

    for tok in stoks:
        features['capany']    = (features['capany'] or tok[0].isupper())
        features['capall']    = (features['capall'] and tok[0].isupper())
        features['stopany']   = (features['stopany'] or tok.lower() in stop)
        features['stopall']   = (features['stopall'] and tok.lower() in stop)
        features['peopleany'] = (features['peopleany'] or tok in people)
        features['peopleall'] = (features['peopleall'] and tok in people)
        features['placesany'] = (features['placesany'] or tok in places)
        features['placesall'] = (features['placesall'] and tok in places)

    return features


def score_phrase(features=dict()):
    '''
    依据句子的特征对句子进行评分
    '''
    weights = {
        'lend':          0.928, 
        'lend1':        -2.619, 
        'cap1':          0.585, 
        'capany':        0.408, 
        'capall':        0.685, 
        'stop1':        -1.029, 
        'stopany':      -1.419, 
        'stopall':      -1.061, 
        'places1':       0.305, 
        'placesany':    -0.179, 
        'placesall':     0.763, 
        'people1':       0.172, 
        'peopleany':    -0.278, 
        'peopleall':    -1.554, 
        'start':        -0.071, 
        'start0':        2.103
        }

    # 计算句子得分
    score = 0
    for w in weights:
        score = score + features[w] * weights[w]
    return score


def score_title(ps_list=[], title='dummy', claim='dummy', model=None):
    '''
    对文档标题进行评分：取句子得分最高的作为文档的得分

    参数：
    ps_list: 和标题相关的一系列句子
    title: 在 claim 中出现的文档标题
    claim: 
     
    返回值：
    maxscore：取句子中得分最高的作为文档的得分
    '''
    maxscore = -1000000

    for phrase, start in ps_list:
        if model is None:
            score = score_phrase(phrase_features(phrase, start, title, claim))
        else:
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
    tscores = sorted(tscores, key=lambda x: -1 * x[1])[: best]

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
        claim = example['claim']
        label = example['label']

        # 对 NEI 类不做处理
        if label == 'NOT ENOUGH INFO':
            continue

        all_evidence = [evi for evi_set in example['evidence'] for evi in evi_set]

        # 建立证据相关的文档的集合
        docs = set()
        for evi in all_evidence:
            evi_doc = evi[2]                # evi: [31205, 37902, 'Peggy_Sue_Got_Married', 0]
            if evi_doc != None:
                docs.add(evi_doc)

        # 构建证据文档 -> sid, sid -> 证据文档的集合
        e2s = dict()
        evi_sets = dict()
        sid = 0

        for s in example['evidence']:       # s 包含多条 evi
            evi_sets[sid] = set()           # 证据的集合，每个证据集对应一个 sid 编号，可根据 sid 查找一个证据集
            for evi in s:
                evi_sets[sid].add(evi[2])

                if evi[2] not in e2s:
                    e2s[evi[2]] = set()     # 证据集合编号的集合，可根据证据来查找证据所在的证据集编号

                e2s[evi[2]].add(sid)
            sid = sid + 1

        for i, (pre_doc, score) in enumerate(tscores[cid]):     # prd_doc 为检索到的文档， score 为对应的评分
            hits[i] = hits[i] + 1 * (pre_doc in docs)           # hits 记录正确定位到证据所在文档的个数
            returned[i] = returned[i] + 1                       # returned 记录返回的文档数
            flag = 0

            if pre_doc in e2s:                                  # 检索到的文档是否在证据集里
                for sid in e2s[pre_doc]:                        # 根据 pre_doc 查找对应证据集的 sid
                    s = evi_sets[sid]                           # 再根据 sid 返回对应的证据集
                    if pre_doc in s:
                        if len(s) == 1:
                            flag = 1                            # flag=1 标记着证据全部被检索到
                        s.remove(pre_doc)
            full[i] += flag                                     # full 记录检索到完整证据的文档数

            if flag == 1:
                break
    print()

    denom = returned[0]

    # 输出文档检索的结果
    for i in range(0, len(hits)):
        print(i, hits[i], returned[i], full[i] / denom)
        full[i + 1] += full[i]


def doc_ir(data=list(), edocs=edict(), best=5, model=None):
    '''
    对每个 claim 返回 best 个得分最高的文档标题
    '''
    docs = dict()
    for example in tqdm(data):
        tscores = best_titles(example['claim'], edocs, best, model)
        docs[example['id']] = tscores
    return docs


if __name__ == '__main__':
    try:
        with open('data/edocs.bin','rb') as rb:
            edocs = pickle.load(rb)
    except:
        t2jnum = titles_to_jsonl_num()
        edocs = title_edict(t2jnum)
        with open('data/edocs.bin', 'wb') as wb:
            pickle.dump(edocs, wb)

    train, dev = load_split_trainset(9999)
    docs = doc_ir(dev, edocs)
    title_hits(dev, docs)
    docs = doc_ir(train, edocs)
    title_hits(train, docs)

