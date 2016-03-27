import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from xgboost.sklearn import XGBClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import roc_auc_score
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

        self.all_feats = ['RESOURCE', 'MGR_ID',
                          'ROLE_DEPTNAME',
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

    def __train_and_predict(self, X_train, y, X_test):
        """Trains the model on supplied training matrix and labels, writes
        prediction probabilities on the supplied test matrix to csv, and the
        prediction probabilities on the supplied train matrix to csv.

        Args:
            X_train (2d): Training feature matrix.
            y (1d): Label vector.
            X_test (2d): Test feature matrix.
        """
        self.model.fit(X_train, y, eval_metric='auc')
        prediction_probs = self.model.predict_proba(X_train)[:, 1]
        print "Training auc = %f" % roc_auc_score(y, prediction_probs)
        self.__write_csv(prediction_probs,
                         X_train.shape[0], self.train_out_file)

        prediction_probs = self.model.predict_proba(X_test)[:, 1]
        self.__write_csv(prediction_probs,
                         X_test.shape[0], self.test_out_file)

        self.feature_imp()

    def __write_csv(self, prediction_probs, n, filename):
        """Write ID, Action to csv file. The Action field contains probability
        of being assigned label 1 (resource allocated).

        Args:
            prediction_probs (list): Vector of prediction probabilites.
            n (int): Number of data points
            filename (string): csv file to be writen.
        """
        d = {'Id': pd.Series([i for i in xrange(1, n + 1)]),
             'Action': pd.Series(prediction_probs)}
        df = pd.DataFrame(d)
        df = df[['Id', 'Action']]
        df.to_csv(filename, sep=',', encoding='utf-8',
                  index=False)

    def feature_imp(self):
        feat_imp = pd.Series(
            self.model.booster().get_fscore()).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')
        plt.show()

    def xgboost_cv(self, X, y):
        xgb_param = self.model.get_xgb_params()
        xg_train = xgb.DMatrix(X, label=y)
        cv_result = xgb.cv(params=xgb_param,
                           dtrain=xg_train,
                           num_boost_round=self.model.get_params()[
                               'n_estimators'],
                           nfold=5,
                           stratified=True,
                           metrics='auc',
                           early_stopping_rounds=50)
        print cv_result
        self.model.set_params(n_estimators=cv_result.shape[0])

    def calc_num_estimators(self):
        self.model = XGBClassifier(
            learning_rate=0.1,
            n_estimators=2000,
            max_depth=7,
            min_child_weight=2,
            gamma=0.0,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            nthread=4,
            scale_pos_weight=0.05,
            seed=27)
        self.xgboost_cv(self.X_train_raw, self.y)
        # 1442 estimators required initially
        # 639 estimators required after tuning

    def tune_depth_child_weight_coarse(self):
        param_test1 = {'max_depth': range(3, 10, 2),
                       'min_child_weight': range(1, 6, 2)}
        gsearch1 = GridSearchCV(
            estimator=XGBClassifier(
                learning_rate=0.1, n_estimators=1442,
                max_depth=4, min_child_weight=1,
                gamma=0, subsample=0.8,
                colsample_bytree=0.8, objective='binary:logistic',
                nthread=4, scale_pos_weight=0.05, seed=27),
            param_grid=param_test1,
            scoring='roc_auc', n_jobs=4, iid=False, cv=5)

        gsearch1.fit(self.X_train_raw, self.y)
        print gsearch1.grid_scores_, gsearch1.best_params_,
        print gsearch1.best_score_
        # best = {'max_depth': 7, 'min_child_weight': 1}

    def tune_depth_child_weight_fine(self):
        param_test2 = {'max_depth': [6, 7, 8],
                       'min_child_weight': [0, 1, 2]}
        gsearch2 = GridSearchCV(
            estimator=XGBClassifier(
                learning_rate=0.1, n_estimators=1442,
                max_depth=7, min_child_weight=1,
                gamma=0, subsample=0.8,
                colsample_bytree=0.8, objective='binary:logistic',
                nthread=4, scale_pos_weight=0.05, seed=27),
            param_grid=param_test2,
            scoring='roc_auc', n_jobs=4, iid=False, cv=5)

        gsearch2.fit(self.X_train_raw, self.y)
        print gsearch2.grid_scores_, gsearch2.best_params_,
        print gsearch2.best_score_
        # best = {'max_depth': 7, 'min_child_weight': 2}

    def tune_gamma(self):
        param_test3 = {'gamma': [i / 10.0 for i in range(0, 5)]}
        gsearch3 = GridSearchCV(
            estimator=XGBClassifier(
                learning_rate=0.1, n_estimators=1442,
                max_depth=7, min_child_weight=2,
                gamma=0, subsample=0.8,
                colsample_bytree=0.8, objective='binary:logistic',
                nthread=4, scale_pos_weight=0.05, seed=27),
            param_grid=param_test3,
            scoring='roc_auc', n_jobs=4, iid=False, cv=5)

        gsearch3.fit(self.X_train_raw, self.y)
        print gsearch3.grid_scores_, gsearch3.best_params_,
        print gsearch3.best_score_
        # best = {'gamma': 0.0}

    def tune_subsample_colsample(self):
        param_test4 = {'subsample': [i / 10.0 for i in range(6, 10)],
                       'colsample_bytree': [i / 10.0 for i in range(6, 10)]}
        gsearch4 = GridSearchCV(
            estimator=XGBClassifier(
                learning_rate=0.1, n_estimators=639,
                max_depth=7, min_child_weight=2,
                gamma=0, subsample=0.8,
                colsample_bytree=0.8, objective='binary:logistic',
                nthread=4, scale_pos_weight=0.05, seed=27),
            param_grid=param_test4,
            scoring='roc_auc', n_jobs=4, iid=False, cv=5)

        gsearch4.fit(self.X_train_raw, self.y)
        print gsearch4.grid_scores_, gsearch4.best_params_,
        print gsearch4.best_score_
        # best = {'subsample': 0.7, 'colsample_bytree': 0.6}

    def tune_scale(self):
        param_test5 = {'scale_pos_weight': list(np.linspace(0.16, 0.26, 9))}
        gsearch5 = GridSearchCV(
            estimator=XGBClassifier(
                learning_rate=0.1, n_estimators=639,
                max_depth=7, min_child_weight=2,
                gamma=0, subsample=0.7,
                colsample_bytree=0.6, objective='binary:logistic',
                nthread=4, scale_pos_weight=0.05, seed=27),
            param_grid=param_test5,
            scoring='roc_auc', n_jobs=4, iid=False, cv=5)

        gsearch5.fit(self.X_train_raw, self.y)
        print gsearch5.grid_scores_, gsearch5.best_params_,
        print gsearch5.best_score_
        # {'scale_pos_weight': 0.19} 0.855971333488

    def xgboost_grid_cv_stepwise(self, feat_list):
        self.__extract_xy_train(feat_list)
        self.__extract_xy_test(feat_list)

        # self.calc_num_estimators()
        # 1442 estimators required
        # self.tune_depth_child_weight_coarse()
        # best = {'max_depth': 7, 'min_child_weight': 1}
        # self.tune_depth_child_weight_fine()
        # best = {'max_depth': 7, 'min_child_weight': 2}
        # self.tune_gamma()
        # best = {'gamma': 0.0}
        # recaliberate boosting rounds with updated param
        # self.calc_num_estimators()
        # 639 estimators required
        # self.tune_subsample_colsample()
        # best = {'subsample': 0.7, 'colsample_bytree': 0.6}
        self.tune_scale()
        # best = {'scale_pos_weight': 0.19}

    def xgboost_train_predict(self, feat_list):
        self.__extract_xy_train(feat_list)
        self.__extract_xy_test(feat_list)

        self.model = XGBClassifier(
            learning_rate=0.05,
            n_estimators=1000,
            max_depth=7,
            min_child_weight=2,
            gamma=0.0,
            subsample=0.7,
            colsample_bytree=0.6,
            objective='binary:logistic',
            nthread=4,
            scale_pos_weight=0.19)
        self.xgboost_cv(self.X_train_raw, self.y)
        self.__train_and_predict(self.X_train_raw, self.y, self.X_test_raw)


def main(train_file, test_file, train_out_file, test_out_file):
    lr = AmazonAccess(train_file, test_file, train_out_file, test_out_file)
    # lr.xgboost_grid_cv_stepwise(lr.all_feats)
    lr.xgboost_train_predict(lr.all_feats)

if __name__ == '__main__':
    train_file = "./data/train_with_feat.csv"
    test_file = "./data/test_with_feat.csv"
    test_out_file = "./results/test.csv"
    train_out_file = "./results/train.csv"
    main(train_file, test_file, train_out_file, test_out_file)
