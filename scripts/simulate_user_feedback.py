"""
Simulate user feedback according to position based clicked model
"""

import argparse
import pickle
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "src"))
from train_LR import NoisyLR

def simulate_data_for_a_query(query, exposure_steepness):
    feedback_data = {}
    feedback_data["Ws"] = np.array([(i + 1) ** exposure_steepness for i in range(query["labels"].size)])
    feedback_data["groups"] = query["groups"]
    feedback_data["scores"] = query["scores"]
    feedback_data["clicks"] = np.zeros(query["labels"].size)
    for i in range(query["labels"].size):
        observe = np.random.binomial(1, 1. / (i + 1) ** exposure_steepness)
        if observe and query["labels"][i]:
            feedback_data["clicks"][i] = 1
    return feedback_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="the data to simulate user feedback")
    parser.add_argument("--logging_policy_path", type=str, help="the logging policy logistic regression model")
    parser.add_argument("--n_queries", type=int, help="the number of simulated queries and user feedback")
    parser.add_argument("--exposure_steepness", type=float, default=1., help="exposure steepness")
    parser.add_argument("--simulated_data_path", type=str, help="output simulation data path")
    parser.add_argument("--t_max", type=int, default=50, help="maximum number of candidates from each group")
    args = parser.parse_args()

    with open(args.data_path, "rb") as f:
        data = pickle.load(f)
    data_size = len(data)
    print("data size: {}".format(str(data_size)))
    with open(args.logging_policy_path, "rb") as f:
        logging_policy = pickle.load(f)
    print("noise ratio: ", logging_policy.noise_ratio)

    # rank items by the logging policy, only select t_max from each group
    ranked_data = []
    for i in range(data_size):
        ranked_data.append({})
        scores = logging_policy.predict_noisy_proba(data[i]["features"], data[i]["groups"])[:, 1]
        # scores_clicked_group = []
        # scores_not_clicked_group = []
        # labels_clicked_group = []
        # labels_not_clicked_group = []
        # groups_clicked_group = []
        # groups_not_clicked_group = []
        # for (j, group) in enumerate(data[i]["groups"]):
        #     if group:
        #         scores_clicked_group.append(data[i]["scores"][j])
        #         labels_clicked_group.append(data[i]["labels"][j])
        #         groups_clicked_group.append(data[i]["groups"][j])
        #     else:
        #         scores_not_clicked_group.append(data[i]["scores"][j])
        #         labels_not_clicked_group.append(data[i]["labels"][j])
        #         groups_not_clicked_group.append(data[i]["groups"][j])
        scores, labels, groups = zip(*sorted(zip(scores, data[i]["labels"], data[i]["groups"]),
                                             key=lambda tup: tup[0], reverse=True))
        scores, labels, groups = np.array(scores), np.array(labels), np.array(groups)
        ranked_data[i]["scores"] = []
        ranked_data[i]["labels"] = []
        ranked_data[i]["groups"] = []
        num_clicked = 0
        num_not_clicked = 0
        for (j, group) in enumerate(groups):
            if num_clicked >= args.t_max and num_not_clicked >= args.t_max:
                break
            if group:
                if num_clicked >= args.t_max:
                    continue
                ranked_data[i]["scores"].append(scores[j])
                ranked_data[i]["labels"].append(labels[j])
                ranked_data[i]["groups"].append(groups[j])
                num_clicked += 1
            else:
                if num_not_clicked >= args.t_max:
                    continue
                ranked_data[i]["scores"].append(scores[j])
                ranked_data[i]["labels"].append(labels[j])
                ranked_data[i]["groups"].append(groups[j])
                num_not_clicked += 1
        ranked_data[i]["scores"] = np.array(ranked_data[i]["scores"])
        ranked_data[i]["labels"] = np.array(ranked_data[i]["labels"])
        ranked_data[i]["groups"] = np.array(ranked_data[i]["groups"])
        # ranked_data[i]["scores"], ranked_data[i]["labels"], ranked_data[i]["groups"] = zip(*sorted(
        #     zip(scores, data[i]["labels"], data[i]["groups"]), key=lambda tup: tup[0], reverse=True))
        # ranked_data[i]["scores"], ranked_data[i]["labels"], ranked_data[i]["groups"] = \
        #     np.array(ranked_data[i]["scores"]), np.array(ranked_data[i]["labels"]), np.array(ranked_data[i]["groups"])

    # simulate user feedback according to position based click model
    sweeps = args.n_queries // data_size
    rest_n_queries = args.n_queries % data_size
    simulated_data = []
    for sweep in range(sweeps):
        for i in range(data_size):
            simulated_data.append(simulate_data_for_a_query(ranked_data[i], args.exposure_steepness))
    random_perm = np.random.permutation(data_size)
    for i in range(rest_n_queries):
        simulated_data.append(simulate_data_for_a_query(ranked_data[random_perm[i]], args.exposure_steepness))

    with open(args.simulated_data_path, "wb") as f:
        pickle.dump(simulated_data, f)



