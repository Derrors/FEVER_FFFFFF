# _*_ coding: utf-8 _*_

import argparse
import os
import config
import subprocess

from get_evidence import run_ir
from converter import run_convert
from jack_reader import run_rte
from neural_aggregator import run_aggregator
from rerank import run_rerank


def environ(env):
    original_environ_dict = os.environ.copy()
    os.environ.update(env)
    yield
    os.environ.clear()
    os.environ.update(original_environ_dict)


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


def run_score(config):
    os.chdir('../fever-baselines')
    options = []
    options.extend(['--predicted_labels', config['dev_predicted_labels']])
    options.extend(['--predicted_evidence', config['dev_predicted_evidence']])
    options.extend(['--actual', config['dev_actual_file']])
    options.extend(['--score_file', config['dev_score']])
    options.extend(['--submission_file', config['dev_submission']])

    with environ({'PYTHONPATH': 'src:../fever'}):
        script = ['python3'] + ['src/scripts/score.py'] + options
        _ = subprocess.run(script)

    options = []
    options.extend(['--predicted_labels', config['test_predicted_labels']])
    options.extend(['--predicted_evidence', config['test_predicted_evidence']])
    options.extend(['--actual', config['test_actual_file']])
    options.extend(['--score_file', config['test_score']])
    options.extend(['--submission_file', config['test_submission']])

    with environ({'PYTHONPATH': 'src:../fever'}):
        script = ['python3'] + ['src/scripts/score.py'] + options + ['--test']
        _ = subprocess.run(script)

    os.chdir('../fever')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', required=True, help='Please input a name for the directory of this task.')
    args = parser.parse_args()

    task_file = os.path.join('./result/', args.task_name)
    if not os.path.exists(task_file):
        os.mkdir(task_file)

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
            config['rte']['dev_predicted_labels_and_scores']) or not os.path.exists(config['rte']['test_predicted_labels_and_scores']):
        run_rte(config['rte'])

    # aggregation if file not exists
    if not os.path.exists(config['aggregator']['dev_predicted_labels_file']):
        run_aggregator(config['aggregator'])

    if 'rerank' in config and not os.path.exists(config['rerank']['dev_reranked_evidence']):
        run_rerank(config['rerank'])

    # scoring
    os.chdir('../fever-baselines')
    if not os.path.exists(config['score']['dev_score_']):
        run_score(config['score'])

    os.chdir('../fever')
