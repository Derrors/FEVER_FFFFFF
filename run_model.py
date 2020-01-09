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
    print('Step 1: Document Retrieval .....')
    if not os.path.exists('./results/doc_ret'):
        os.mkdir('./results/doc_ret')
        run_doc_ret(config['doc_ret'])
    print('Document Retrieval has conpleted.')

    print('Step 2: Sentence Retrieval .....')
    if not os.path.exists('./results/sent_ret'):
        os.mkdir('./results/sent_ret')
        run_sent_ret(config['sent_ret'])
    print('Sentence Retrieval has completed.')

    # perform IR if file doesn't exist
    if not os.path.exists('./results/evi_ret'):
        os.mkdir('./results/evi_ret')
        run_evi_ret(config['evi_ret'])

    # nli inference if file does not exist
    print('Step 3: Natural Language Inference .....')
    if not os.path.exists('./results/nli'):
        os.mkdir('./results/nli')
        run_nli(config['nli'])
    print('Natural Language Inference has completed.')

    # aggregation if file not exists
    print('Step 4: Aggregation .....')
    if not os.path.exists('./results/aggregator'):
        os.mkdir('./results/aggregator')
        run_aggregator(config['aggregator'])
    print('Aggregation has completed.')

    print('Step 5: Rerank the results .....')
    if not os.path.exists('./results/rerank'):
        os.mkdir('./results/rerank')
        run_rerank(config['rerank'])

    # scoring
    print('Step 6: Score the results .....')
    if not os.path.exists('./results/score'):
        os.mkdir('./results/score')
        run_score(config['score']['dev'])
