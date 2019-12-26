# _*_ coding: utf-8 _*_

import os
from configs import config
import subprocess

from doc_ir_model import doc_ir_model
from line_ir_model import line_ir_model
from get_evidence import run_ir
from converter import run_convert
from jack_reader import run_rte
from neural_aggregator import run_aggregator
from rerank import run_rerank
from score import run_score


def train_rte(config):
    os.chdir('../jack')
    options = list()
    options.append('with')
    options.append('config={}'.format(config['jack_config_file']))
    options.append('save_dir={}'.format(config['save_dir']))
    options.append('train={}'.format(config['train_file']))
    options.append('dev={}'.format(config['dev_file']))
    options.append('test={}'.format(config['dev_file']))
    if 'load_dir' in config and config['load_dir'] != '':
        options.append('load_dir={}'.format(config['load_dir']))

    script = ['python3'] + ['bin/jack-train.py'] + options
    _ = subprocess.run(script)
    os.chdir('../fever')


if __name__ == "__main__":
    # perform IR if file doesn't exist
    if not (os.path.exists(config['ir']['train_output']) and os.path.exists(config['ir']['dev_output'])):
        run_ir(config['ir'])                    # python get_evidence.py

    # convert format if file does not exist
    if not (os.path.exists(config['convert']['train_output']) and os.path.exists(config['convert']['dev_output'])):
        run_convert(config['convert'])               # python convert.py

    # train rte model if file does not exist
    if not os.path.isdir('./results/rte_output/reader'):
        train_rte(config['train_rte'])           # python ../jack/bin/jack-train.py

    # rte inference if file does not exist
    if not os.path.exists(config['rte']['train_predicted_labels_and_scores']) or not os.path.exists(
            config['rte']['dev_predicted_labels_and_scores']):
        run_rte(config['rte'])

    # aggregation if file not exists
    if not os.path.exists(config['aggregator']['dev_predicted_labels']):
        run_aggregator(config['aggregator'])

    if 'rerank' in config and not os.path.exists(config['rerank']['dev_reranked_evidence']):
        run_rerank(config['rerank'])

    # scoring
    if not os.path.exists(config['score']['dev']['score_file']):
        run_score(config['score']['train'])
        run_score(config['score']['dev'])
