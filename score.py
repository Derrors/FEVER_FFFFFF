
import json
import sys
from fever.scorer import fever_score
from prettytable import PrettyTable
from analyse import print_confusion_mat, save_simple_result, save_submission_file


def run_score(config):
    ids = []
    predicted_labels = []
    predicted_evidence = []
    actual = []

    with open(config['predicted_labels'], "r") as predictions_file:
        for line in predictions_file:
            predicted_labels.append(json.loads(line)["predicted"])

    with open(config['predicted_evidence'], "r") as predictions_file:
        for line in predictions_file:
            predicted_evidence.append(json.loads(line)["predicted_sentences"][:5])
            ids.append(json.loads(line)["id"])

    predictions = []
    for id, ev, label in zip(ids, predicted_evidence, predicted_labels):
        predictions.append({"id": id, "predicted_evidence": ev, "predicted_label": label})

    save_submission_file(predictions, config['submission'])

    with open(config['actual_file'], "r") as actual_file:
        for line in actual_file:
            actual.append(json.loads(line))

    score, acc, precision, recall, f1 = fever_score(predictions, actual)
    save_simple_result(config['score_file'], score, acc, precision, recall)
    print_confusion_mat(predictions, actual)

    tab = PrettyTable()
    tab.field_names = ["FEVER Score", "Label Accuracy", "Evidence Precision", "Evidence Recall", "Evidence F1"]
    tab.add_row((round(score, 4), round(acc, 4), round(precision, 4), round(recall, 4), round(f1, 4)))

    print(tab)
