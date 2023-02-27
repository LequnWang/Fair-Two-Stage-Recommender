"""
Binarize Relevance: the new relevance is 1 if the original relevance is 2, 3, or 4, and 0 otherwise.
Transform feature: sklearn Standard Scaler on the whole dataset.
Transform to pickle: for each query, we have features of webpages, labels of webpages, groups of webpages (by
whether the click count of an url is greater than zero or not).
"""

import argparse
import pickle
from sklearn.preprocessing import StandardScaler
import numpy as np


def read_original_file(path):
    labels = []
    groups = []
    qids = []
    features = []
    current_qid = None
    with open(path, 'r') as f:
        for line in f:
            items = line.strip().split()
            label = int(items[0])
            if label > 1:
                labels.append(1)
            else:
                labels.append(0)
            qid = items[1]
            if current_qid != qid:
                current_qid = qid
            qids.append(current_qid)
            feature = np.zeros(137)
            feature[0] = 1.
            for item in items[2:]:
                index, value = item.split(":")
                index = int(index)
                value = float(value)
                feature[index] = value
            features.append(feature)
            # feature 135 is url click count
            if feature[135] > 0:
                groups.append(1)
            else:
                groups.append(0)
    labels = np.array(labels)
    groups = np.array(groups)
    qids = np.array(qids)
    features = np.array(features)
    return labels, groups, qids, features


def create_data_dict(labels, groups, qids, features):
    data_dict = []
    current_qid_index = -1
    current_qid = None
    for i in range(labels.size):
        if qids[i] != current_qid:
            current_qid = qids[i]
            current_qid_index += 1
            data_dict.append({})
            data_dict[current_qid_index]["features"] = []
            data_dict[current_qid_index]["labels"] = []
            data_dict[current_qid_index]["groups"] = []
        data_dict[current_qid_index]["features"].append(features[i])
        data_dict[current_qid_index]["labels"].append(labels[i])
        data_dict[current_qid_index]["groups"].append(groups[i])
    for i in range(current_qid_index + 1):
        data_dict[i]["features"] = np.array(data_dict[i]["features"])
        data_dict[i]["labels"] = np.array(data_dict[i]["labels"])
        data_dict[i]["groups"] = np.array(data_dict[i]["groups"])
    return data_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="./data/train.txt",
                        help="the training data in the original dataset, which we use to simulate user feedback")
    parser.add_argument("--valid_path", type=str, default="./data/vali.txt",
                        help="the validation data in the original dataset, which we use to train logging policy")
    parser.add_argument("--test_path", type=str, default="./data/test.txt",
                        help="the testing data in the original dataset, which we use for evaluation")
    parser.add_argument("--data_path", type=str, default ="./data/data.pkl", help="processed data")
    # parser.add_argument("--data_user_feedback_simulation_path", type=str,
    #                     default="./data/data_user_feedback_simulation.pkl", help="data to simulate user feedback")
    # parser.add_argument("--data_logging_policy_path", type=str, default="./data/data_logging_policy.pkl",
    #                     help="data to train the logging policy")
    # parser.add_argument("--data_test_path", type=str, default="./data/data_test.pkl",
    #                     help="data to evaluate algorithms")

    args = parser.parse_args()
    train_labels, train_groups, train_qids, train_features = read_original_file(args.train_path)
    valid_labels, valid_groups, valid_qids, valid_features = read_original_file(args.valid_path)
    test_labels, test_groups, test_qids, test_features = read_original_file(args.test_path)

    # print the average number of items and relevant items from different groups in the training set
    print("the average number of items each query: {}".format(str(
        (train_labels.size + valid_labels.size + test_labels.size)
        / 30000)))
    print("the average number of relevant items each query: {}".format(str(
        (np.sum(train_labels) + np.sum(valid_labels) + np.sum(test_labels))
        / 30000)))
    print("the average number of relevant items from the clicked group each query: {}".format(str(
        (np.dot(train_labels, train_groups) + np.dot(valid_labels, valid_groups) + np.dot(test_labels, test_groups))
        / 30000)))
    print("the average number of relevant items from the non-clicked group each query: {}".format(str(
        (np.dot(train_labels, 1 - train_groups) + np.dot(valid_labels, 1 - valid_groups) +
         np.dot(test_labels, 1 - test_groups)) / 30000)))


    # transform features
    # train_size = train_labels.size
    # valid_size = valid_labels.size
    # test_size = test_labels.size
    scaler = StandardScaler()
    features = scaler.fit_transform(np.concatenate((train_features, valid_features, test_features)))
    labels = np.concatenate((train_labels, valid_labels, test_labels))
    groups = np.concatenate((train_groups, valid_groups, test_groups))
    qids = np.concatenate((train_qids, valid_qids, test_qids))

    # transform to dict
    data_dict = create_data_dict(labels, groups, qids, features)
    with open(args.data_path, "wb") as f:
        pickle.dump(data_dict, f)
