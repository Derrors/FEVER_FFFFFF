
import re
import os
import sys
import json
import random

from util import abs_path
from tqdm import tqdm


def save_jsonl(dictionaries, path, print_message=True, skip_if_exists=False):
    '''
    将保存字典的列表输出为 jsonl 文件
    '''
    # 待保存文件存在时处理方式
    if os.path.exists(path):
        if not skip_if_exists:
            raise OSError('file {} already exists'.format(path))
        else:
            print('CAUTION: skip saving (file {} already exists)'.format(path))
            return

    if print_message:
        print('saving at {}'.format(path))

    with open(path, 'a') as out_file:
        for instance in dictionaries:
            out_file.write(json.dumps(instance) + '\n')


def read_jsonl(path):
    '''
    读取 jsonl 文件
    '''
    with open(path, 'r') as in_file:
        out = [json.loads(line) for line in in_file]
    return out


def load_doc_lines(docs=dict(), t2jnum=dict(), wikipedia_dir='./data/wiki-pages/'):
    '''
    建立由 title 查找对应的 line_id 和 line_text 的字典
    参数：
    docs: claim ids 对应 titles 的字典   # {cid: [(title, score),  ...], ...}
    t2jnum: title 对应 jsonl num 的字典
    '''

    doclines = dict()               # title -> jnum -> line_id, line_text
    jnums = dict()                  # jnum -> point1, point2, ...
    titles = set()

    # 获取 title 和 title 对应的 jnum, point
    for cid in docs:
        for title, _ in docs[cid]:
            doclines[title] = dict()
            titles.add(title)

            jnum, point = t2jnum[title]
            if jnum not in jnums:
                jnums[jnum] = set()
            jnums[jnum].add(point)

    for jnum in tqdm(jnums):
        points = sorted(list(jnums[jnum]))               # 对 point 排序
        fname = wikipedia_dir + 'wiki-' + jnum + '.jsonl'
        with open(fname, 'r') as f:
            for point in points:
                f.seek(point, 0)
                line = f.readline()
                # 读取为字典结构
                data = json.loads(line.strip())
                title = data['id']
                lines = data['lines']
                # 将 title 对应 line num, line_text 加入字典
                if title in titles and lines != '':
                    for l in lines.split('\n'):
                        fields = l.split('\t')
                        if fields[0].isnumeric():
                            l_id = int(fields[0])
                            l_text = fields[1]
                            doclines[title][l_id] = l_text
    return doclines

         
def load_doclines(titles, t2jnum, filtering=True):
    '''
    加载 title 对应的所有 lines

    参数：
    titles: list of titles

    '''
    # 过滤掉不在 t2jnum 里的标题
    if filtering:
        filtered_titles = [title for title in titles if title in t2jnum]
        print('mismatch: {} / {}'.format(len(titles) - len(filtered_titles), len(titles)))
        titles = filtered_titles

    return load_doc_lines({'dummy_id' : [(title, 'dummy_linum') for title in titles]}, t2jnum, wikipedia_dir=abs_path('data/wiki-pages/'))


def titles_to_jsonl_num(wikipedia_dir='./data/wiki-pages/', doctitles='./data/doctitles'):
    '''
    建立从文档标题到 jsonl 编号文件的查找字典, 并保存到 ./data/doctitles 以加速读取
    t2jnum = {'title': (jnum, point), ...}
    '''
    t2jnum = dict()
    try:
        with open(doctitles, 'r') as f:
            for line in f:
                fields = line.strip().split('\t')
                title = fields[0]                   # 标题
                jnum = fields[1]                    # 对应的 jsonl 文件编号
                point = int(fields[2])
                t2jnum[title] = (jnum, point)
            if len(t2jnum) == 0:
                raise RuntimeError('doctitles file ({}) might be empty.'.format(doctitles))
    except:
        with open(doctitles,'w') as w:
            for i in tqdm(range(1, 110)):
                jnum = '{:03d}'.format(i)
                fname = wikipedia_dir + 'wiki-' + jnum + '.jsonl'
                with open(fname) as f:
                    point = f.tell()
                    line = f.readline()
                    while line:
                        data = json.loads(line.strip())
                        title = data['id']
                        lines = data['lines']
                        w.write(title + '\t' + jnum + '\t' + str(point) + '\n')
                        t2jnum[title] = (jnum, point)
                        point = f.tell()
                        line = f.readline()
    return t2jnum


def get_evidence_sentence_list(evidences, t2l2s):
    '''
    根据 evidences 里的 title, linum 在 t2l2s 找到对应的 sentences

    参数：
    evidences: [(title, linum), ...]
    t2l2s: title -> linum -> sentence

    返回值：
    evidence sentences list

    '''

    SEP = '#'

    def maybe_prepend(title, linum):
        prep = list()
        prep.append(title)

        content = ' {} '.format(SEP).join(prep)
        if prep:
            return '{0} {1} {0}'.format(SEP, content)
        else:
            return content

    titles = [title for title, _ in evidences]
    linums = [linum for _, linum in evidences]

    evidences_sentences = []
    for title, linum in zip(titles, linums):
        _title = re.sub('_', ' ', title)                # 'hoge_fuga_hoo' -> 'hoge fuga hoo'
        sentence = maybe_prepend(_title, linum) + ' ' + t2l2s[title][linum]
        sentence.strip()
        evidences_sentences.append(sentence)

    return evidences_sentences


def load_wikipedia(wikipedia_dir='./data/wiki-pages/', n_files=1000):
    '''
    加载 wiki-pages 数据, 每个 file 包含 50k articles.
    返回包含所有 wikipedia article texts 的列表
    '''
    all_texts = []
    print('loading wikipedia...')
    
    # 获取需要加载的文件名并排序
    file_names = sorted(os.listdir(wikipedia_dir))[: n_files]
    for file_name in tqdm(file_names):
        with open(wikipedia_dir + file_name, 'r') as openfile:
            texts = [json.loads(line)['text'] for line in openfile.readlines()]
        all_texts.extend(texts)
    print('Loaded', len(all_texts), 'articles. Size (MB):', round(sys.getsizeof(all_texts) / 1024 / 1024, 3))

    return all_texts

'''
def get_label_set():
    label_set = {'SUPPORTS','REFUTES','NOT ENOUGH INFO'}
    return label_set
'''

def load_fever_train(path='data/train.jsonl', n_instances=999999):
    '''
    加载训练集
    '''
    data = []
    with open(path, 'r') as openfile:
        for i, line in enumerate(openfile.readlines()):
            data.append(json.loads(line))
            if i + 1 >= n_instances:
                break
    return data


def load_split_trainset(dev_size):
    '''
    加载全部训练数据, 并初步划分为训练集和验证集
    '''
    # load fever training data
    full_train = load_fever_train()

    positives = []              # SUPPORT 类数据
    negatives = []              # REFUTES 类数据
    neutrals = []               # NEI 类数据

    # 根据标签对数据分类
    for example in full_train:
        label = example['label']
        if label == 'SUPPORTS':
            positives.append(example)
        elif label == 'REFUTES':
            negatives.append(example)
        elif label == 'NOT ENOUGH INFO':
            neutrals.append(example)

    # 随机乱序
    random.seed(2020)
    random.shuffle(positives)
    random.shuffle(negatives)
    random.shuffle(neutrals)

    # 划分验证集, 三类平衡
    size = int(dev_size / 3)
    dev_set = positives[: size] + negatives[: size] + neutrals[: size]
    # 剩余数据作为训练集
    train_set = positives[size: ] + negatives[size: ] + neutrals[size: ]

    random.shuffle(dev_set)
    random.shuffle(train_set)

    return train_set, dev_set


def load_paper_dataset(train=abs_path('./data/train.jsonl'), dev=abs_path('./data/dev.jsonl')):
    '''
    加载论文对应数据
    '''
    train_ds = load_fever_train(path=train, n_instances=9999999999)
    dev_ds = load_fever_train(path=dev, n_instances=9999999999)
    return train_ds, dev_ds


if __name__ == '__main__':
    # load fever training data
    fever_data = load_fever_train(n_instances=20)
    print(len(fever_data))

    # load train and split-off dev set of size 9999.
    train, dev = load_split_trainset(9999)
    print(len(train))
    print(len(dev))

    for sample in train[: 3]:
        print(sample)

