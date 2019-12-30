# _*_ coding: utf-8 _*_

import os
from configs import config
import subprocess

from document_retrieval import run_doc_ret
from sentence_retrieval import run_sent_ret
from evidence_retriieval import run_evi_ret
from natural_language_inference import run_nli
from neural_aggregator import run_aggregator
from rerank import run_rerank
from score import run_score


if __name__ == "__main__":
    if not os.path.exists('./results/doc_ret'):
        os.mkdir('./results/doc_ret')
        run_doc_ret(config['doc_ret'])

    if not os.path.exists('./results/sent_ret'):
        os.mkdir('./results/sent_ret')
        run_sent_ret(config['sent_ret'])

    # perform IR if file doesn't exist
    if not os.path.exists('./results/evi_ret'):
        os.mkdir('./results/evi_ret')
        run_evi_ret(config['evi_ret'])

    # nli inference if file does not exist
    if not os.path.exists('./results/nli'):
        os.mkdir('./results/nli')
        run_nli(config['nli'])

    # aggregation if file not exists
    if not os.path.exists('./results/aggregator'):
        os.mkdir('./results/aggregator')
        run_aggregator(config['aggregator'])

    if not os.path.exists('./results/rerank'):
        os.mkdir('./results/rerank')
        run_rerank(config['rerank'])

    # scoring
    if not os.path.exists('./results/score'):
        os.mkdir('./results/score')
        run_score(config['score']['train'])
        run_score(config['score']['dev'])
