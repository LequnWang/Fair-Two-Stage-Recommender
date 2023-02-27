"""
Select top items from each group so that the sum of prediction scores is greater than the target number of
qualified items
"""

import argparse
import pickle
import numpy as np
from train_LR import NoisyLR


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_test_path", type=str, help="data to evaluate algorithms")
    parser.add_argument("--logging_policy_path", type=str, help="the logging policy logistic regression model")
    parser.add_argument("--result_path", type=str, help="evaluation results on the test data")
    parser.add_argument("--k_clicked", type=float, help="the target expected number of relevant items in the clicked "
                                                        "group")
    parser.add_argument("--k_not_clicked", type=float, help="the target expected number of relevant items in the not"
                                                            "clicked group")

    args = parser.parse_args()

    with open(args.data_test_path, "rb") as f:
        data_test = pickle.load(f)
    with open(args.logging_policy_path, "rb") as f:
        logging_policy = pickle.load(f)

    # select items
    data_test_size = len(data_test)
    ranked_data_test = []
    for i in range(data_test_size):
        ranked_data_test.append({})
        scores = logging_policy.predict_noisy_proba(data_test[i]["features"], data_test[i]["groups"])[:, 1]
        scores, labels, groups = zip(*sorted(zip(scores, data_test[i]["labels"], data_test[i]["groups"]),
                                             key=lambda tup: tup[0], reverse=True))
        scores, labels, groups = np.array(scores), np.array(labels), np.array(groups)
        sum_scores_clicked = 0
        sum_scores_not_clicked = 0
        ranked_data_test[i]["scores"] = []
        ranked_data_test[i]["labels"] = []
        ranked_data_test[i]["groups"] = []
        num_clicked = 0
        num_not_clicked = 0
        for (j, group) in enumerate(groups):
            if sum_scores_clicked >= args.k_clicked and sum_scores_not_clicked >= args.k_not_clicked:
                break
            if group:
                if sum_scores_clicked >= args.k_clicked:
                    continue
                ranked_data_test[i]["scores"].append(scores[j])
                ranked_data_test[i]["labels"].append(labels[j])
                ranked_data_test[i]["groups"].append(groups[j])
                sum_scores_clicked += scores[j]
            else:
                if sum_scores_not_clicked >= args.k_not_clicked:
                    continue
                ranked_data_test[i]["scores"].append(scores[j])
                ranked_data_test[i]["labels"].append(labels[j])
                ranked_data_test[i]["groups"].append(groups[j])
                sum_scores_not_clicked += scores[j]
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
    print(performance_metrics)
    # save results
    with open(args.result_path, "wb") as f:
        pickle.dump(performance_metrics, f)
