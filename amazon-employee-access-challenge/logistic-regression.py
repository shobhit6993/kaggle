import pandas as pd
import numpy as np
import itertools
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.cross_validation import cross_val_score
from random import shuffle


class AmazonAccess(object):
    def __init__(self, train_file, test_file, train_out_file, test_out_file):
        self.train_file = train_file
        self.test_file = test_file
        self.train_out_file = train_out_file
        self.test_out_file = test_out_file

        self.data = self.__read_data(train_file)
        self.X_train_raw = None
        self.X_train_sparse = None
        self.y = None

        self.keymap = None
        self.model = None

        self.test_data = self.__read_data(test_file)
        self.X_test_raw = None
        self.X_test_sparse = None

        self.all_feats = ['RESOURCE', 'MGR_ID', 'ROLE_ROLLUP_1',
                          'ROLE_ROLLUP_2', 'ROLE_DEPTNAME',
                          'ROLE_TITLE', 'ROLE_FAMILY_DESC',
                          'ROLE_FAMILY', 'ROLE_CODE'
                          ]

    def __read_data(self, filename):
        return pd.read_csv(filename)

    def __replace_by_noise(self, c):
        """Replace a column with noise to assess the strength of contribution
        of the column to label prediction. The noise is emulated by shuffling
        the entries for that column among the training examples.

        Args:
            c (int): Column index
        """
        n = self.X_train_raw.shape[0]
        temp = []
        for r in xrange(0, n):
            temp.append(self.X_train_raw[r][c])
        shuffle(temp)
        for r in xrange(0, n):
            self.X_train_raw[r][c] = temp[r]

    def __extract_xy_train(self, colmns):
        """Extracts feature matrix (X_train) and label vector (y) from csv
        parsed labelled training data. Only the specified columns are used.

        Args:
            colmns (list): List of columns to be used in feature matrix.
        """
        self.keymap = None
        self.X_train_raw = self.data[colmns].as_matrix()
        # self.__replace_by_noise(c=1)
        n = self.X_train_raw.shape[0]
        self.X_train_raw = np.hstack((self.X_train_raw,
                                      np.ones(shape=(n, 1))))
        self.X_train_sparse, self.keymap = self.__one_hot_encoder(
            self.X_train_raw, True, False)
        self.y = self.data['ACTION'].as_matrix().reshape(-1, 1).ravel()

    def __extract_xy_test(self, colmns):
        """Extracts feature matrix (X_test) and label vector (y) from csv
        parsed unlabelled test data. Only the specified columns are used.

        Args:
            colmns (list): List of columns to be used in feature matrix.
        """
        self.X_test_raw = self.test_data[colmns].as_matrix()
        n = self.X_test_raw.shape[0]
        self.X_test_raw = np.hstack((self.X_test_raw,
                                     np.ones(shape=(n, 1))))
        self.X_test_sparse, _ = self.__one_hot_encoder(
            self.X_test_raw, False, False, self.keymap)

    def __one_hot_encoder(self, raw_data, train, oov, keymap=None):
        """Given a matrix with categorical columns, returns a sparse matrix
        with one-hot encoded columns. Also sets the keymap for categorical to
        index mapping. The keymap is set only during training.

        To handle OOV categories during testing, the first occurrence of each
        category in a column during training is treated as OOV (all features
        for that category in that particular training eg are set to 0).

        Args:
            raw_data (matrix): Matrix of raw categorical data.
            train (bool): Set to True during training.
            oov (bool): Set to True to add smoothing.

        Returns:
            sparse csr: sparse one-hot encoded matrix
        """
        n = raw_data.shape[0]

        if keymap is None:
            keymap = []
            for col in raw_data.T:
                unique = set(col)
                keymap.append(dict((v, i) for i, v in enumerate(unique)))

        sparse_data = []
        seen = []   # keep track of first occurrence of each ID in a column
        for c, col in enumerate(raw_data.T):
            col_matrix = sparse.lil_matrix((n, len(keymap[c])))
            if oov and train:   # treat first occ. as OOV only during training
                seen.append(set())

            for r, v in enumerate(col):
                if oov and train and (v not in seen[-1]):  # first occ. of v
                    seen[-1].add(v)  # add v to seen set. Keep the feat vec 0s.

                if v in keymap[c]:
                    col_matrix[r, keymap[c][v]] = 1

            sparse_data.append(col_matrix)

        return sparse.hstack(sparse_data).tocsr(), keymap

    def __cross_validation(self, X, y):
        """Performs 5-fold cross validation on the supplied feature matrix
        and labels.

        Args:
            X (2d): Input feature matrix
            y (1d): Labels

        Returns:
            float: Mean ROC AUC for the 5-fold CV
        """
        scores = cross_val_score(self.model,
                                 X, y, scoring=make_scorer(roc_auc_score),
                                 cv=10, n_jobs=1)
        print scores
        print "auc mean = ", scores.mean()
        print "auc sd = ", np.std(scores)
        return scores.mean()

    def __train_and_predict(self, X_train, y, X_test):
        """Trains the model on supplied training matrix and labels, writes
        prediction probabilities on the supplied test matrix to csv, and the
        prediction probabilities on the supplied train matrix to csv.

        Args:
            X_train (2d): Training feature matrix.
            y (1d): Label vector.
            X_test (2d): Test feature matrix.
        """
        self.model.fit(X_train, y)
        print "Training = %f" % self.model.score(X_train, y)

        prediction_probs = self.model.predict_proba(X_test)
        self.__write_csv(prediction_probs,
                         self.X_test_sparse.shape[0], self.test_out_file)

        prediction_probs = self.model.predict_proba(X_train)
        self.__write_csv(prediction_probs,
                         self.X_train_sparse.shape[0], self.train_out_file)

    def __write_csv(self, prediction_probs, n, filename):
        """Write ID, Action to csv file. The Action field contains probability
        of being assigned label 1 (resource allocated).

        Args:
            prediction_probs (list): Vector of prediction probabilites.
            n (int): Number of data points
            filename (string): csv file to be writen.
        """
        d = {'Id': pd.Series([i for i in xrange(1, n + 1)]),
             'Action': pd.Series(prediction_probs[:, 1])}
        df = pd.DataFrame(d)
        df = df[['Id', 'Action']]
        df.to_csv(filename, sep=',', encoding='utf-8',
                  index=False)

    def logistic_regression_cv(self, feat_list):
        """Trains and tests a LogisticRegression model with GridSearch for
        the regularization parameter. Uses only the features provided in the
        feat_list

        Args:
            feat_list (list): List of features (df colm names) to be used.
        """
        self.__extract_xy_train(feat_list)
        self.__extract_xy_test(feat_list)

        self.model = LogisticRegressionCV(Cs=20,
                                          class_weight='balanced',
                                          cv=5,
                                          penalty='l2',
                                          scoring=make_scorer(roc_auc_score),
                                          n_jobs=1)
        self.__cross_validation(self.X_train_sparse, self.y)
        self.__train_and_predict(self.X_train_sparse,
                                 self.y, self.X_test_sparse)

    def logistic_regression(self, feat_list):
        """Trains and tests a LogisticRegression model. Uses only the feature
        provided in the feat_list

        Args:
            feat_list (list): List of features (df colm names) to be used.
        """
        self.__extract_xy_train(feat_list)
        self.__extract_xy_test(feat_list)

        self.model = LogisticRegression(class_weight='balanced')
        return self.__cross_validation(self.X_train_sparse, self.y)
        # self.__train_and_predict(self.X_train_sparse,
        # self.y, self.X_test_sparse)

    def feature_selection(self, max_features):
        subset = set(itertools.combinations(self.all_feats, 1))
        feat_list = []
        curr_list = []
        k = 1
        prev_max_score = -1
        max_score = 0
        while k <= max_features and prev_max_score < max_score:
            print "Trying number of features = ", k
            prev_max_score = max_score
            max_score = -1
            best_s = None
            for s in subset:
                curr_list = list(feat_list)
                curr_list.append(s[0])
                print curr_list
                score = self.logistic_regression(curr_list)
                if score > max_score:
                    best_s = s
                    max_score = score
                print ""
            feat_list.append(best_s[0])
            subset.remove(best_s)
            print "Best for %d is %f" % (k, max_score), feat_list
            k = k + 1

        # remove last feature added
        if k != max_features:
            del(feat_list[-1])

        print "\n\n*****Feature selection done.******"
        print "Best greedy feature subset: ", feat_list, "\n\n"
        return feat_list


def main(train_file, test_file, train_out_file, test_out_file):
    lr = AmazonAccess(train_file, test_file, train_out_file, test_out_file)
    feat_list = lr.feature_selection(10)
    lr.logistic_regression_cv(feat_list)

if __name__ == '__main__':
    train_file = "./data/train.csv"
    test_file = "./data/test.csv"
    test_out_file = "./results/test.csv"
    train_out_file = "./results/train.csv"
    main(train_file, test_file, train_out_file, test_out_file)
