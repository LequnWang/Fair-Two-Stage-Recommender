"""
Train a Noisy Logistic Regression Classifier from Training Data
"""

import argparse
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score


class NoisyLR(LogisticRegression):
    def set_noise_ratio(self, noise_ratio=None):
        self.noise_ratio = noise_ratio

    def predict_noisy_proba(self, X, groups=None):
        if self.noise_ratio is None:
            return super().predict_proba(X)

        proba = super().predict_proba(X)
        for i in range(proba.shape[0]):
            if groups is None or groups[i]:
                noise_or_not = np.random.binomial(1, self.noise_ratio["clicked"])
            else:
                noise_or_not = np.random.binomial(1, self.noise_ratio["not_clicked"])
            if noise_or_not:
                noise = np.random.beta(1, 10)
                proba[i, 0] = 1. - noise
                proba[i, 1] = noise
        return proba


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", type=str, help="the input training data path")
    # parser.add_argument("--lbd", type=float, help="L2 regularization parameter")
    parser.add_argument("--C", type=float, default=1., help="L2 regularization parameter")
    parser.add_argument("--classifier_path", type=str, help="the output classifier path")
    parser.add_argument('--noise_ratio_clicked', type=float, default=0., help="noise ratio of clicked")
    parser.add_argument('--noise_ratio_not_clicked', type=float, default=-1., help="noise ratio of not clicked group")

    args = parser.parse_args()

    with open(args.train_data_path, "rb") as f:
        X, y = pickle.load(f)
        # print(np.sum(y))
        # print(y.size)
        # n = y.size
        # C = 1 / (args.lbd * n)

    classifier = NoisyLR(C=args.C, max_iter=1000).fit(X, y)
    if args.noise_ratio_not_clicked < 0.:
        classifier.set_noise_ratio(None)
    else:
        noise_ratio = {}
        noise_ratio["clicked"] = args.noise_ratio_clicked
        noise_ratio["not_clicked"] = args.noise_ratio_not_clicked
        classifier.set_noise_ratio(noise_ratio)
    y_pred = classifier.predict(X)
    y_score = classifier.predict_proba(X)[:, 1]
    print("training accuracy: {}".format(str(accuracy_score(y, y_pred))))
    print("training roc_auc score: {}".format(str(roc_auc_score(y, y_score))))
    print("training average_precision score: {}".format(str(average_precision_score(y, y_score))))
    with open(args.classifier_path, "wb") as f:
        pickle.dump(classifier, f)
