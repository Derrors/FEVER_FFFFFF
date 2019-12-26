
config = {
    'ir': {
        'n_docs': 5,
        'n_sents': 5,
        'train_input': '/home/qhli/FEVER/FEVER_FFFFF/fever/data/train.jsonl',
        'dev_input': '/home/qhli/FEVER/FEVER_FFFFF/fever/data/dev.jsonl',
        'train_output': '/home/qhli/FEVER/FEVER_FFFFF/fever/results/ir_output/ir_train.jsonl',
        'dev_output': '/home/qhli/FEVER/FEVER_FFFFF/fever/results/ir_output/ir_dev.jsonl'
    },

    'convert': {
        'n_sentences': 5,
        'use_ir_pred': False,
        'train_input': '/home/qhli/FEVER/FEVER_FFFFF/fever/results/ir_output/ir_train.jsonl',
        'train_output': '/home/qhli/FEVER/FEVER_FFFFF/fever/results/converted/converted_train.jsonl',
        'dev_input': '/home/qhli/FEVER/FEVER_FFFFF/fever/results/ir_output/ir_dev.jsonl',
        'dev_output': '/home/qhli/FEVER/FEVER_FFFFF/fever/results/converted/converted_dev.jsonl'
    },

    'train_rte': {
        'jack_config_file': '/home/qhli/FEVER/FEVER_FFFFF/jack/conf/nli/fever/esim.yaml',
        'load_dir': 'esim_snli',
        'save_dir': '/home/qhli/FEVER/FEVER_FFFFF/fever/home/qhli/FEVER/FEVER_FFFFF/fever/fever/results/train_rte/reader',
        'train_file': '/home/qhli/FEVER/FEVER_FFFFF/fever/home/qhli/FEVER/FEVER_FFFFF/fever/fever/results/converted/converted_train.jsonl',
        'dev_file': '/home/qhli/FEVER/FEVER_FFFFF/fever/home/qhli/FEVER/FEVER_FFFFF/fever/fever/results/converted/converted_dev.jsonl'
    },

    'rte': {
        'n_sentences': 15,
        'batch_size': 32,
        'train_input_file': '/home/qhli/FEVER/FEVER_FFFFF/fever/results/ir_output/ir_train.jsonl',
        'dev_input_file': '/home/qhli/FEVER/FEVER_FFFFF/fever/results/ir_output/ir_dev.jsonl',
        'saved_reader': '/home/qhli/FEVER/FEVER_FFFFF/fever/results/train_rte/reader',
        'train_predicted_labels_and_scores': '/home/qhli/FEVER/FEVER_FFFFF/fever/results/rte_output/train.predictions.jsonl',
        'dev_predicted_labels_and_scores': '/home/qhli/FEVER/FEVER_FFFFF/fever/results/rte_output/dev.predictions.jsonl'
    },

    'aggregator': {
        'layers': [28, 100, 100],
        'epochs': 5,
        'n_sentences': 7,
        'sampling': False,
        'evi_scores': True,
        'train_file': '/home/qhli/FEVER/FEVER_FFFFF/fever/results/rte_output/train.predictions.jsonl',
        'dev_file': '/home/qhli/FEVER/FEVER_FFFFF/fever/results/rte_output/dev.predictions.jsonl',
        'train_predicted_labels': '/home/qhli/FEVER/FEVER_FFFFF/fever/results/aggregator_output/train_aggregated_labels.jsonl',
        'dev_predicted_labels': '/home/qhli/FEVER/FEVER_FFFFF/fever/results/aggregator_output/dev_aggregated_labels.jsonl'
    },

    'rerank': {
        'n_sentences': 15,
        'train_reranked_evidence': '/home/qhli/FEVER/FEVER_FFFFF/fever/results/rerank_output/train_reranked_evidences.jsonl',
        'train_rte_predictions': '/home/qhli/FEVER/FEVER_FFFFF/fever/results/rte_output/train.predictions.jsonl',
        'train_aggregated_labels': '/home/qhli/FEVER/FEVER_FFFFF/fever/results/aggregator_output/train_aggregated_labels.jsonl',
        'train_predicted_evidence': '/home/qhli/FEVER/FEVER_FFFFF/fever/results/ir_output/ir_train.jsonl',
        'dev_reranked_evidence': '/home/qhli/FEVER/FEVER_FFFFF/fever/results/rerank_output/dev_reranked_evidences.jsonl',
        'dev_rte_predictions': '/home/qhli/FEVER/FEVER_FFFFF/fever/results/rte_output/dev.predictions.jsonl',
        'dev_aggregated_labels': '/home/qhli/FEVER/FEVER_FFFFF/fever/results/aggregator_output/dev_aggregated_labels.jsonl',
        'dev_predicted_evidence': '/home/qhli/FEVER/FEVER_FFFFF/fever/results/ir_output/ir_dev.jsonl'
    },

    'score': {
        'train': {
            'score_file': '/home/qhli/FEVER/FEVER_FFFFF/fever/results/score/train_score.jsonl',
            'actual_file': '/home/qhli/FEVER/FEVER_FFFFF/fever/results/ir_output/ir_train.jsonl',
            'submission': '/home/qhli/FEVER/FEVER_FFFFF/fever/results/score/train_submission.jsonl',
            'predicted_labels': '/home/qhli/FEVER/FEVER_FFFFF/fever/results/aggregator_output/train_aggregated_labels.jsonl',
            'predicted_evidence': '/home/qhli/FEVER/FEVER_FFFFF/fever/results/rerank_output/train_reranked_evidences.jsonl'
        },

        'dev': {
            'score_file': '/home/qhli/FEVER/FEVER_FFFFF/fever/results/score/dev_score.jsonl',
            'actual_file': '/home/qhli/FEVER/FEVER_FFFFF/fever/results/ir_output/ir_dev.jsonl',
            'submission': '/home/qhli/FEVER/FEVER_FFFFF/fever/results/score/submission.jsonl',
            'predicted_labels': '/home/qhli/FEVER/FEVER_FFFFF/fever/results/aggregator_output/dev_aggregated_labels.jsonl',
            'predicted_evidence': '/home/qhli/FEVER/FEVER_FFFFF/fever/results/rerank_output/dev_reranked_evidences.jsonl'
        }
    }
}
