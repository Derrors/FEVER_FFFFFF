# _*_ coding: utf-8 _*_

import pickle
from collections import Counter

from nltk import word_tokenize
from tqdm import tqdm

from doc_ir import doc_ir, title_edict
from fever_io import load_doc_lines, load_split_trainset, titles_to_jsonl_num
from util import load_stoplist, normalize_title

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


def score_line(features=dict()):
    '''
    特征权重
    '''
    vlist = {
        'lenl': 0.032,
        'tinl': -0.597,
        'lid': -0.054,
        'lid0': 1.826,
        'pc': -3.620,
        'pl': 3.774,
        'pcns': 3.145,
        'plns': -6.423,
        'pcnt': 4.195,
        'pcntns': 2.795,
        'plntns': 5.133
    }

    score = 0

    for v in vlist:
        score = score + features[v] * vlist[v]
    return score


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
                if model is None:
                    lscores.append((title, lid, score_line(line_features(c_toks, t, t_toks, line, l_toks, lid, tscore))))
                else:
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


def line_ir(data=list(), docs=dict(), lines=dict(), best=5, model=None):
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


if __name__ == '__main__':
    t2jnum = titles_to_jsonl_num()
    try:
        with open('./data/edocs.bin', 'rb') as rb:
            edocs = pickle.load(rb)
    except BaseException:
        edocs = title_edict(t2jnum)
        with open('./data/edocs.bin', 'wb') as wb:
            pickle.dump(edocs, wb)

    train, dev = load_split_trainset(9999)
    docs = doc_ir(dev, edocs)
    print(len(docs))

    lines = load_doc_lines(docs, t2jnum)
    print(len(lines))

    evidence = line_ir(dev, docs, lines)
    line_hits(dev, evidence)

    docs = doc_ir(train, edocs)
    print(len(docs))

    lines = load_doc_lines(docs, t2jnum)
    print(len(lines))

    evidence = line_ir(train, docs, lines)
    line_hits(train, evidence)
