
config = {
    'doc_ret': {
        'n_best': 5,
        'edocs_path': '/home/qhli/FEVER/FEVER_FFFFF/fever/data/preprocessed_data/edocs.bin',
        'doc_ret_docs': '/home/qhli/FEVER/FEVER_FFFFF/fever/results/doc_ret/doc_ret_docs',
        'doc_ret_model': '/home/qhli/FEVER/FEVER_FFFFF/fever/results/doc_ret/doc_ret_model.bin'
    },

    'sent_ret': {
        'n_best': 5,
        'edocs_path': '/home/qhli/FEVER/FEVER_FFFFF/fever/data/preprocessed_data/edocs.bin',
        'sent_ret_line': '/home/qhli/FEVER/FEVER_FFFFF/fever/results/sent_ret/sent_ret_lines',
        'doc_ret_model': '/home/qhli/FEVER/FEVER_FFFFF/fever/results/doc_ret/doc_ret_model.bin',
        'sent_ret_model': '/home/qhli/FEVER/FEVER_FFFFF/fever/results/sent_ret/sent_ret_model.bin'
    },

    'evi_ret': {
        'n_docs': 5,
        'n_sents': 5,
        'train_input': '/home/qhli/FEVER/FEVER_FFFFF/fever/data/dataset/train.jsonl',
        'dev_input': '/home/qhli/FEVER/FEVER_FFFFF/fever/data/dataset/dev.jsonl',
        'train_output': '/home/qhli/FEVER/FEVER_FFFFF/fever/results/evi_ret/evi_ret_train.jsonl',
        'dev_output': '/home/qhli/FEVER/FEVER_FFFFF/fever/results/evi_ret/evi_ret_dev.jsonl'
    },

    'nli': {
        'n_sentences': 15,
        'batch_size': 32,
        'train_input_file': '/home/qhli/FEVER/FEVER_FFFFF/fever/results/evi_ret/evi_ret_train.jsonl',
        'dev_input_file': '/home/qhli/FEVER/FEVER_FFFFF/fever/results/evi_ret/evi_ret_dev.jsonl',
        'saved_reader': '/home/qhli/FEVER/FEVER_FFFFF/fever/data/reader',
        'train_predicted_labels_and_scores': '/home/qhli/FEVER/FEVER_FFFFF/fever/results/nli/train.predictions.jsonl',
        'dev_predicted_labels_and_scores': '/home/qhli/FEVER/FEVER_FFFFF/fever/results/nli/dev.predictions.jsonl'
    },

    'aggregator': {
        'layers': [28, 100, 100],
        'epochs': 5,
        'n_sentences': 7,
        'sampling': False,
        'evi_scores': True,
        'train_file': '/home/qhli/FEVER/FEVER_FFFFF/fever/results/nli/train.predictions.jsonl',
        'dev_file': '/home/qhli/FEVER/FEVER_FFFFF/fever/results/nli/dev.predictions.jsonl',
        'train_predicted_labels': '/home/qhli/FEVER/FEVER_FFFFF/fever/results/aggregator/train_aggregated_labels.jsonl',
        'dev_predicted_labels': '/home/qhli/FEVER/FEVER_FFFFF/fever/results/aggregator/dev_aggregated_labels.jsonl'
    },

    'rerank': {
        'n_sentences': 15,
        'train_reranked_evidence': '/home/qhli/FEVER/FEVER_FFFFF/fever/results/rerank/train_reranked_evidences.jsonl',
        'train_rte_predictions': '/home/qhli/FEVER/FEVER_FFFFF/fever/results/nli/train.predictions.jsonl',
        'train_aggregated_labels': '/home/qhli/FEVER/FEVER_FFFFF/fever/results/aggregator/train_aggregated_labels.jsonl',
        'train_predicted_evidence': '/home/qhli/FEVER/FEVER_FFFFF/fever/results/evi_ret/evi_ret_train.jsonl',
        'dev_reranked_evidence': '/home/qhli/FEVER/FEVER_FFFFF/fever/results/rerank/dev_reranked_evidences.jsonl',
        'dev_rte_predictions': '/home/qhli/FEVER/FEVER_FFFFF/fever/results/nli/dev.predictions.jsonl',
        'dev_aggregated_labels': '/home/qhli/FEVER/FEVER_FFFFF/fever/results/aggregator/dev_aggregated_labels.jsonl',
        'dev_predicted_evidence': '/home/qhli/FEVER/FEVER_FFFFF/fever/results/evi_ret/evi_ret_dev.jsonl'
    },

    'score': {
        'train': {
            'score_file': '/home/qhli/FEVER/FEVER_FFFFF/fever/results/score/train_score.jsonl',
            'actual_file': '/home/qhli/FEVER/FEVER_FFFFF/fever/results/evi_ret/evi_ret_train.jsonl',
            'submission': '/home/qhli/FEVER/FEVER_FFFFF/fever/results/score/train_submission.jsonl',
            'predicted_labels': '/home/qhli/FEVER/FEVER_FFFFF/fever/results/aggregator/train_aggregated_labels.jsonl',
            'predicted_evidence': '/home/qhli/FEVER/FEVER_FFFFF/fever/results/rerank/train_reranked_evidences.jsonl'
        },

        'dev': {
            'score_file': '/home/qhli/FEVER/FEVER_FFFFF/fever/results/score/dev_score.jsonl',
            'actual_file': '/home/qhli/FEVER/FEVER_FFFFF/fever/results/evi_ret/evi_ret_dev.jsonl',
            'submission': '/home/qhli/FEVER/FEVER_FFFFF/fever/results/score/submission.jsonl',
            'predicted_labels': '/home/qhli/FEVER/FEVER_FFFFF/fever/results/aggregator/dev_aggregated_labels.jsonl',
            'predicted_evidence': '/home/qhli/FEVER/FEVER_FFFFF/fever/results/rerank/dev_reranked_evidences.jsonl'
        }
    }
}
