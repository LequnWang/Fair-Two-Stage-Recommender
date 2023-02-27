"""
Select top items from each group so that the sum of the Platt scaling calibrated prediction scores is greater than
the target number of qualified items on average over the calibration set
"""

import argparse
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from train_LR import NoisyLR


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_feedback_path", type=str, help="the user feedback data")
    parser.add_argument("--data_test_path", type=str, help="data to evaluate algorithms")
    parser.add_argument("--logging_policy_path", type=str, help="the logging policy logistic regression model")
    parser.add_argument("--result_path", type=str, help="evaluation results on the test data")
    parser.add_argument("--k_clicked", type=float, help="the target expected number of relevant items in the clicked "
                                                        "group")
    parser.add_argument("--k_not_clicked", type=float, help="the target expected number of relevant items in the not"
                                                            "clicked group")
    args = parser.parse_args()
    with open(args.user_feedback_path, "rb") as f:
        user_feedback = pickle.load(f)
        n = len(user_feedback)

    # feature, label, weight for calibration
    X_cal, y_cal, weight_cal = [], [], []
    for query in user_feedback:
        for j in range(query["scores"].size):
            score = max(min(query["scores"][j], 1e-6), 1 - 1e-6)
            logit = np.log(score / (1 - score))
            if query["clicks"][j]:
                X_cal.append([logit])
                y_cal.append(1)
                weight_cal.append(query["Ws"][j])
                X_cal.append([logit])
                y_cal.append(0)
                weight_cal.append(1 - query["Ws"][j])
            else:
                X_cal.append([logit])
                y_cal.append(0)
                weight_cal.append(1)
    X_cal = np.array(X_cal)
    y_cal = np.array(y_cal)
    weight_cal = np.array(weight_cal)
    calibration_model = LogisticRegression().fit(X_cal, y_cal, sample_weight=weight_cal)

    # change scores to calibrated scores
    for query in user_feedback:
        scores = query["scores"]
        scores = np.array([max(min(score, 1e-6), 1 - 1e-6) for score in scores])
        logits = np.array([[np.log(scores[j] / (1 - scores[j]))] for j in range(scores.size)])
        query["scores"] = calibration_model.predict_proba(logits)[:, 1]

    # find the largest ts for each group
    t_max_clicked = 0
    t_max_not_clicked = 0
    for query in user_feedback:
        if np.sum(query["groups"]) > t_max_clicked:
            t_max_clicked = np.sum(query["groups"])
        if np.sum(1 - query["groups"]) > t_max_not_clicked:
            t_max_not_clicked = np.sum(1 - query["groups"])
    print(t_max_clicked)
    print(t_max_not_clicked)

    # select thresholds for each group by the average sum of scores on the calibration data
    t_hat_clicked = None
    for t in range(t_max_clicked + 1):
        sum_of_scores = 0
        for (i, query) in enumerate(user_feedback):
            num_selected = 0
            for j in range(query["scores"].size):
                if num_selected >= t:
                    break
                if query["groups"][j]:
                    num_selected += 1
                    sum_of_scores += query["scores"][j]
        average_sum_of_scores = sum_of_scores / n
        if average_sum_of_scores >= args.k_clicked:
            t_hat_clicked = t
            break

    if t_hat_clicked is None:
        print("no threshold for the clicked such that the sum of scores on the calibration data is greater the target"
              "number of relevant items")
        t_hat_clicked = 50

    t_hat_not_clicked = None
    for t in range(t_max_not_clicked + 1):
        sum_of_scores = 0
        for (i, query) in enumerate(user_feedback):
            num_selected = 0
            for j in range(query["scores"].size):
                if num_selected >= t:
                    break
                if 1 - query["groups"][j]:
                    num_selected += 1
                    sum_of_scores += query["scores"][j]
        average_sum_of_scores = sum_of_scores / n
        if average_sum_of_scores >= args.k_not_clicked:
            t_hat_not_clicked = t
            break

    if t_hat_not_clicked is None:
        print("no threshold for the not clicked such that the sum of scores on the calibration data is greater "
              "the target number of relevant items")
        t_hat_not_clicked = 50

    # test the selected thresholds
    with open(args.data_test_path, "rb") as f:
        data_test = pickle.load(f)
    with open(args.logging_policy_path, "rb") as f:
        logging_policy = pickle.load(f)

    # rank items by the logging policy
    data_test_size = len(data_test)
    ranked_data_test = []
    for i in range(data_test_size):
        ranked_data_test.append({})
        scores = logging_policy.predict_noisy_proba(data_test[i]["features"], data_test[i]["groups"])[:, 1]
        scores, labels, groups = zip(*sorted(zip(scores, data_test[i]["labels"], data_test[i]["groups"]),
                                             key=lambda tup: tup[0], reverse=True))
        scores, labels, groups = np.array(scores), np.array(labels), np.array(groups)
        ranked_data_test[i]["scores"] = []
        ranked_data_test[i]["labels"] = []
        ranked_data_test[i]["groups"] = []
        num_clicked = 0
        num_not_clicked = 0
        for (j, group) in enumerate(groups):
            if num_clicked >= t_hat_clicked and num_not_clicked >= t_hat_not_clicked:
                break
            if group:
                if num_clicked >= t_hat_clicked:
                    continue
                ranked_data_test[i]["scores"].append(scores[j])
                ranked_data_test[i]["labels"].append(labels[j])
                ranked_data_test[i]["groups"].append(groups[j])
                num_clicked += 1
            else:
                if num_not_clicked >= t_hat_not_clicked:
                    continue
                ranked_data_test[i]["scores"].append(scores[j])
                ranked_data_test[i]["labels"].append(labels[j])
                ranked_data_test[i]["groups"].append(groups[j])
                num_not_clicked += 1
        ranked_data_test[i]["scores"] = np.array(ranked_data_test[i]["scores"])
        ranked_data_test[i]["labels"] = np.array(ranked_data_test[i]["labels"])
        ranked_data_test[i]["groups"] = np.array(ranked_data_test[i]["groups"])

    # calculate metrics
    performance_metrics = {}
    performance_metrics["num_relevant_clicked"] = 0
    performance_metrics["num_relevant_not_clicked"] = 0
    performance_metrics["num_selected_clicked"] = 0
    performance_metrics["num_selected_not_clicked"] = 0
    for query in ranked_data_test:
        performance_metrics["num_relevant_clicked"] += np.sum(query["labels"] * query["groups"])
        performance_metrics["num_relevant_not_clicked"] += np.sum(query["labels"] * (
                                                                         1 - query["groups"]))
        performance_metrics["num_selected_clicked"] += np.sum(query["groups"])
        performance_metrics["num_selected_not_clicked"] += np.sum(1 - query["groups"])
    performance_metrics["num_relevant_clicked"] /= (1. * len(ranked_data_test))
    performance_metrics["num_relevant_not_clicked"] /= (1. * len(ranked_data_test))
    performance_metrics["num_selected_clicked"] /= (1. * len(ranked_data_test))
    performance_metrics["num_selected_not_clicked"] /= (1. * len(ranked_data_test))
    performance_metrics["constraint_satisfied_clicked"] = performance_metrics["num_relevant_clicked"] >= args.k_clicked
    performance_metrics["constraint_satisfied_not_clicked"] = performance_metrics["num_relevant_not_clicked"] >= \
                                                          args.k_not_clicked
    performance_metrics["num_relevant"] = performance_metrics["num_relevant_clicked"] + \
                                          performance_metrics["num_relevant_not_clicked"]
    performance_metrics["num_selected"] = performance_metrics["num_selected_clicked"] + \
                                          performance_metrics["num_selected_not_clicked"]
    print(t_hat_clicked, t_hat_not_clicked)
    print(performance_metrics)
    # save results
    with open(args.result_path, "wb") as f:
        pickle.dump(performance_metrics, f)
