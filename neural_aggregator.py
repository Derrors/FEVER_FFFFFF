# _*_ coding: utf-8 _*_

import argparse
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from fever_io import read_jsonl, save_jsonl

np.random.seed(1)
torch.backends.cudnn.deterministic = True
torch.manual_seed(1)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1)

label2idx = {'SUPPORTS': 0, 'REFUTES': 1, 'NOT ENOUGH INFO': 2, 'DUMMY LABEL': 3}
idx2label = {idx: label for label, idx in label2idx.items()}
supports = idx2label[0]
refutes = idx2label[1]
nei = idx2label[2]

# 3层全连接网络


class Net(nn.Module):
    def __init__(self, layers=[15, 10, 5]):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(layers[0], layers[1])
        self.fc2 = nn.Linear(layers[1], layers[2])
        self.fc3 = nn.Linear(layers[2], 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Predicted_Labels_Dataset(Dataset):
    def __init__(self, jsonl_file, n_sentences=5, sampling=False, use_ev_scores=False, test=False):
        instances = read_jsonl(jsonl_file)

        if sampling:
            instances = sample(instances)

        self.instances = instances
        self.n_sentences = n_sentences
        self.test = test
        self.use_ev_scores = use_ev_scores

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        if self.test:
            label = label2idx['DUMMY LABEL']
        else:
            label = label2idx[self.instances[idx]['label']]

        if self.use_ev_scores:
            input = create_input3(
                self.instances[idx]['predicted_labels'],
                self.instances[idx]['scores'],
                self.instances[idx]['ev_scores'],
                n_sentences=self.n_sentences
            )
        else:
            input = create_input(
                self.instances[idx]['predicted_labels'],
                self.instances[idx]['scores'],
                n_sentences=self.n_sentences
            )

        return (label, input)


def sample(train_set):
    '''
    采样数据
    '''
    print('performing sampling...')
    sampled_instances = list()
    label2freq = Counter((instance['label'] for instance in train_set))         # 统计数据集中各类标签的数量
    print('label2freq:', label2freq)

    min_freq = min(label2freq.values())
    counter_dict = dict()
    for instance in train_set:                                                  # 根据数量最少的标签来采样数据
        label = instance['label']
        if label not in counter_dict:
            counter_dict[label] = 1
        elif counter_dict[label] < min_freq:
            counter_dict[label] += 1
            sampled_instances.append(instance)

    return sampled_instances


def create_input(predicted_labels, scores, n_sentences):
    '''
    '''
    zero_plus_eye = np.vstack([np.eye(3), np.zeros((1, 3))])
    zero_pad_idx = 3

    pred_labels = [label2idx[pred_label] for pred_label in predicted_labels]
    scores = scores.copy()

    # 如果句子数小于 n_sentences, 则标签补 3, 得分补 0
    if len(pred_labels) < n_sentences:
        n_fillup = n_sentences - len(pred_labels)
        pred_labels += [zero_pad_idx] * n_fillup
        scores += [0.] * n_fillup

    one_hot = zero_plus_eye[pred_labels, :]            # 4种标签的 One-hot 形式

    np_out = np.reshape(np.multiply(one_hot, np.expand_dims(scores, axis=1)), (-1))

    return np_out


def create_input3(predicted_labels, scores, sentence_scores, n_sentences):
    '''
    4 features for each predicted evidence: the 3 RTE class probabilities, and the IR score.
    All 3 class probabilities are given.
    '''
    assert len(predicted_labels[0]) == len(scores)

    if len(sentence_scores) == 0:
        return np.zeros([4 * n_sentences])

    # 构造输入特征
    features_per_predicted_evidence = []
    for per_evidence_labels, per_evidence_scores, ir_evidence_scores in list(zip(predicted_labels[0], scores, sentence_scores))[:n_sentences]:

        if not per_evidence_scores:
            new_features = [0.0, 0.0, 0.0, 0.0]
        else:
            new_features = per_evidence_scores + [ir_evidence_scores[2]]

        features_per_predicted_evidence.extend(new_features)

    # 句子数小于 n_sentences 时, 补 0
    for missing_evidence in range(n_sentences - len(sentence_scores)):
        features_per_predicted_evidence.extend([0.0, 0.0, 0.0, 0.0])

    np_out = np.array(features_per_predicted_evidence)
    assert np_out.shape == (4 * n_sentences, ), (np_out, len(predicted_labels[0]), len(scores), len(sentence_scores))

    return np_out


def simple_test(dev_dataloader):
    '''
    测试在开发集上的预测准确率
    '''
    neural_hit = 0

    with torch.no_grad():
        for i, (target, input) in enumerate(dev_dataloader):
            neural_pred = model(input.float())
            _, pred_labels = torch.max(neural_pred, 1)
            neural_hit += torch.sum(pred_labels == target)

    performance = int(neural_hit) / len(dev_dataloader.dataset)
    print('neural:', performance)
    return performance


def predict(test_dataloader):
    '''
    获取标签的预测结果
    '''
    results = list()

    with torch.no_grad():
        for i, (labels, input) in enumerate(test_dataloader):
            neural_preds = model(input.float())
            _, pred_labels = torch.max(neural_preds, 1)

            for label, neural_pred in zip(labels, pred_labels):
                results.append({
                    'actual': idx2label[int(label)],
                    'predicted': idx2label[int(neural_pred)]
                })

    return results


def run_aggregator(config):

    train_set = Predicted_Labels_Dataset(config['train_file'], config['n_sentences'], sampling=config['sampling'], use_ev_scores=config['evi_scores'])
    dev_set = Predicted_Labels_Dataset(config['dev_file'], config['n_sentences'], use_ev_scores=config['evi_scores'])
    test_set = Predicted_Labels_Dataset(config['test_file'], config['n_sentences'], use_ev_scores=config['evi_scores'], test=True)

    train_dataloader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=0)
    dev_dataloader = DataLoader(dev_set, batch_size=64, shuffle=False, num_workers=0)
    test_dataloader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=0)

    model = Net(layers=[int(width) for width in config['layers']])

    class_weights = [1.0, 1.0, 1.0]
    label2freq = Counter((instance['label'] for instance in train_set.instances))
    total = sum(label2freq.values())
    for label in label2freq:
        class_weights[label2idx[label]] = 1.0 / (label2freq[label]) * total

    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights))
    optimizer = optim.Adam(model.parameters())

    dev_results = []

    for epoch in range(config['epochs']):
        running_loss = 0.0

        for i, (labels, inputs) in enumerate(train_dataloader):
            optimizer.zero_grad()

            outputs = model(inputs.float())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 1000 == 999:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 1000))
                running_loss = 0.0
        dev_results.append(simple_test(dev_dataloader))

    print('Finished Training.')
    performance = max(dev_results)
    print('dev set:', performance)

    dev_results = predict(dev_dataloader)
    test_results = predict(test_dataloader)
    save_jsonl(dev_results, config['dev_predicted_labels'])
    save_jsonl(test_results, config['test_predicted_labels'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True)
    parser.add_argument('--dev', required=True)
    parser.add_argument('--test', required=True)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--n_sentences', default=5, type=int)
    parser.add_argument('--predicted_labels', required=True)
    parser.add_argument('--test_predicted_labels', required=True)
    parser.add_argument('--sampling', action='store_true')
    parser.add_argument('--ev_scores', action='store_true')
    parser.add_argument('--l2', default=0.0, type=float, required=False)
    parser.add_argument('--layers', nargs='+', required=True)
    args = parser.parse_args()

    print(args)
    hyperparameter2performance = dict()

    for n_sentences in [9]:
        print('=========== n_sentences {}============'.format(str(n_sentences)))
        args.n_sentences = n_sentences
        # number of inputs will be 4 times number of evidence sentences.
        args.layers[0] = args.n_sentences * 4

        train_set = Predicted_Labels_Dataset(args.train, args.n_sentences, sampling=args.sampling, use_ev_scores=args.ev_scores)
        dev_set = Predicted_Labels_Dataset(args.dev, args.n_sentences, use_ev_scores=args.ev_scores)
        test_set = Predicted_Labels_Dataset(args.test, args.n_sentences, use_ev_scores=args.ev_scores, test=True)

        train_dataloader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=0)
        dev_dataloader = DataLoader(dev_set, batch_size=64, shuffle=False, num_workers=0)
        test_dataloader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=0)

        model = Net(layers=[int(width) for width in args.layers])
        print('----Neural Aggregator Architecture----')
        print(model)

        class_weights = [1.0, 1.0, 1.0]
        label2freq = Counter((instance['label'] for instance in train_set.instances))
        total = sum(label2freq.values())
        for label in label2freq:
            class_weights[label2idx[label]] = 1.0 / (label2freq[label]) * total
        print(label2freq)
        print('Class Weights:', class_weights)

        criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights))
        optimizer = optim.Adam(model.parameters())

        dev_results_throughout_training = []

        for epoch in range(args.epochs):
            print('epoch:', epoch)
            running_loss = 0.0

            for i, (labels, inputs) in enumerate(train_dataloader):
                optimizer.zero_grad()

                outputs = model(inputs.float())
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if i % 1000 == 999:
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 1000))
                    running_loss = 0.0
            dev_results_throughout_training.append(simple_test(dev_dataloader))

        print('Finished Training')
        print('dev set:')
        performance = simple_test(dev_dataloader)
        hyperparameter2performance[n_sentences] = max(dev_results_throughout_training)

    for k, v in sorted(hyperparameter2performance.items()):
        print(v)

    dev_results = predict(dev_dataloader)
    test_results = predict(test_dataloader)
    save_jsonl(dev_results, args.predicted_labels)
    save_jsonl(test_results, args.test_predicted_labels)
