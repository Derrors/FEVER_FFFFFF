# _*_ coding: utf-8 _*_

import os
import json
from nltk import word_tokenize, sent_tokenize


def load_stoplist(stop_file='stoplist'):
    '''
    加载停用词表
    '''
    stop_words = set()
    with open(stop_file) as f:
        for line in f:
            word = line.strip()
            stop_words.add(word)
    return stop_words


def normalize_title(title, rflag=False):
    '''
    对标题进行标准化（字符转换、分割，字母小写）
    '''
    text = title.replace('_',' ').replace('-COLON-',':')            # '_' -> ' ', '-COLON-' -> ':'
    rmndr = ''
    # 标题包含（）,则将标题分成前部和（）两部分
    if text.find('-LRB-') > -1:                                     # '-LRB-' -> '(', '-RRB-' -> ')'
        rmndr = text[text.find('-LRB-'): ]
        rmndr = rmndr.replace('-LRB-','(').replace('-RRB-',')')
        text = text[: text.find('-LRB-')].rstrip(' ')
    text = word_tokenize(text.lower())
    # 是否返回（）里的内容
    if rflag:
        return text, rmndr                                          # title(adas) -> title、(adas)
    else:
        return text


def abs_path(relative_path_to_file):
    '''
    路径转换：相对路径 -> 绝对路径
    '''
    # os.path.abspath(__file__)返回的是.py文件的绝对路径（完整路径）
    current_dir = os.path.dirname(os.path.abspath(__file__)) 
    return os.path.join(current_dir, relative_path_to_file)


class edict():
    def __init__(self):
        self.d = dict()
    
    def __getitem__(self, key):
        if key[0] in self.d:
            if len(key)==1:
                return self.d[key[0]]
            else:
                return self.d[key[0]][1][key[1:]]
        else:
            return (None, None)
    
    def __setitem__(self, key, value):
        if len(key) == 1:
            self.d[key[0]] = (value, self.d.get(key[0], (None, edict()))[1])
        else:
            val, sube = self.d.get(key[0], (None, edict()))
            sube[key[1:]] = value
            self.d[key[0]] = (val, sube)
    
    def __contains__(self, key):
        if len(key) == 1:
            return key[0] in self.d
        else:
            return key[0] in self.d and key[1: ] in self.d[key[0]][1]
    
    def __len__(self):
        return len(self.d)


class pdict():
    def __init__(self, ed):
        self.ed = ed
        self.pos = 0
        self.d = {'': (self.ed, self.pos)}

    def __getitem__(self, key):
        self.pos += 1
        
        newd = {'': (self.ed, self.pos)}
        rlist = []

        for prefix in self.d:
            start = self.d[prefix][1]

            if [key.lower()] in self.d[prefix][0]:
                tf, ped = self.d[prefix][0][[key.lower()]]
                new_prefix = prefix + ' ' + key

                if len(ped) > 0:
                    newd[new_prefix] = (ped, start)

                if tf is not None:
                    rlist.append((tf, new_prefix, start))
        self.d = newd
        return rlist
