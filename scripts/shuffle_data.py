"""
shuffle the data for logging policy training, user feeedback generation, and evaluation
"""

import argparse
import pickle
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data/data.pkl", help="processed data")
    parser.add_argument("--data_user_feedback_simulation_path", type=str, help="data to simulate user feedback")
    parser.add_argument("--data_logging_policy_path", type=str, help="data to train the logging policy")
    parser.add_argument("--data_test_path", type=str, help="data to evaluate algorithms")
    parser.add_argument("--data_logging_policy_proportion", type=float, default=0.01, help="proportion of data to "
                                                                                           "train the logging policy")
    parser.add_argument("--data_test_proportion", type=float, default=0.3)

    args = parser.parse_args()
    with open(args.data_path, 'rb') as f:
        data = pickle.load(f)
    n = len(data)
    data_logging_policy_size = int(n * args.data_logging_policy_proportion)
    data_test_size = int(n * args.data_test_proportion)
    data_user_feedback_simulation_size = n - data_logging_policy_size - data_test_size

    np.random.shuffle(data)
    data_logging_policy = data[: data_logging_policy_size]
    data_user_feedback_simulation = data[data_logging_policy_size: n - data_test_size]
    data_test = data[n - data_test_size:]

    X_logging_policy = []
    y_logging_policy = []
    for i in range(data_logging_policy_size):
        for j in range(data_logging_policy[i]["labels"].size):
            X_logging_policy.append(data_logging_policy[i]["features"][j])
            y_logging_policy.append(data_logging_policy[i]["labels"][j])
    X_logging_policy = np.array(X_logging_policy)
    y_logging_policy = np.array(y_logging_policy)
    with open(args.data_user_feedback_simulation_path, "wb") as f:
        pickle.dump(data_user_feedback_simulation, f)
    with open(args.data_test_path, "wb") as f:
        pickle.dump(data_test, f)
    with open(args.data_logging_policy_path, "wb") as f:
        pickle.dump([X_logging_policy, y_logging_policy], f)
