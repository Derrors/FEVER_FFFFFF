
config = {
    'ir': {
        'n_docs': 5,
        'n_sents': 5,
        'train_input': './data/train.jsonl',
        'dev_input': './data/test.jsonl',
        'test_input': './data/shared_task_test.jsonl',
        'train_output': './data/ir_output/ir_train.jsonl',
        'dev_output': './data/ir_output/ir_dev.jsonl',
        'test_output': './data/ir_output/ir_test.jsonl'
    },

    'convert': {
        'n_sentences': 5,
        'use_ir_pred': False,
        'train_input': './data/ir_output/ir_train.jsonl',
        'train_output': './data/converted/converted_train.jsonl',
        'dev_input': './data/ir_output/ir_dev.jsonl',
        'dev_output': './data/converted/converted_test.jsonl'
    },

    'train_rte': {
        "jack_config_file": "./conf/nli/fever/esim.yaml",
        "load_dir": "esim_snli",
        "save_dir": "../fever/results/trian_rte/reader",
        "train_file": '../fever/data/converted/converted_train.jsonl',
        "dev_file": '../fever/data/converted/converted_test.jsonl'
    },

    'rte': {
        'n_sentences': 15,
        'batch_size': 32,
        "train_input_file": './data/ir_output/ir_train.jsonl',
        "dev_input_file": './data/ir_output/ir_dev.jsonl',
        "test_input_file": './data/ir_output/ir_test.jsonl',
        "saved_reader": "./results/trian_rte/reader",
        "train_predicted_labels_and_scores": "./results/rte_output/train.predictions.jsonl",
        "dev_predicted_labels_and_scores": "./results/rte_output/dev.predictions.jsonl",
        "test_predicted_labels_and_scores": "./results/rte_output/test.predictions.jsonl"
    },

    'aggregator': {
        'layers': [28, 100, 100],
        "epochs": 5,
        'n_sentences': 7,
        'sampling': False,
        'evi_scores': True,
        "train_file": "./results/rte_output/train.predictions.jsonl",
        "dev_file": "./results/rte_output/dev.predictions.jsonl",
        "test_file": "./results/rte_output/test.predictions.jsonl",
        "dev_predicted_labels": "./results/aggregator_output/dev_aggregated_labels.jsonl",
        "test_predicted_labels": "./results/aggregator_output/test_aggregated_labels.jsonl"
    },

    'rerank': {
        'n_sentences': 15,
        'dev_reranked_evidence': './results/rerank_output/dev_reranked_evidences.jsonl',
        'dev_rte_predictions': "./results/rte_output/dev.predictions.jsonl",
        'dev_aggregated_labels': "./results/aggregator_output/dev_aggregated_labels.jsonl",
        'dev_predicted_evidence': './data/ir_output/ir_dev.jsonl',
        'test_reranked_evidence': './results/rerank_output/test_reranked_evidences.jsonl',
        'test_rte_predictions': "./results/rte_output/test.predictions.jsonl",
        'test_aggregated_labels': "./results/aggregator_output/test_aggregated_labels.jsonl",
        'test_predicted_evidence': './data/ir_output/ir_test.jsonl'
    },

    'score': {
        "dev_score": "./results/score/dev_score.jsonl",
        "dev_actual_file": './data/ir_output/ir_dev.jsonl',
        "dev_submission_file": "./results/score/submission.jsonl",
        "dev_predicted_labels": "./results/aggregator_output/dev_aggregated_labels.jsonl",
        'dev_predicted_evidence': './results/rerank_output/dev_reranked_evidences.jsonl',
        "test_score": "./results/score/test_score.jsonl",
        "test_actual_file": './data/shared_task_test.jsonl',
        "test_submission": "./results/score/test_submission.jsonl",
        "test_predicted_labels": "./results/aggregator_output/test_aggregated_labels.jsonl",
        "test_predicted_evidence": './results/rerank_output/test_reranked_evidences.jsonl'
    }
}
