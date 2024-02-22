import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score

from .model_base import Model
from calf import logger


class LogModel(Model):
    """
    Learns a separate ridge regression model for each target dimension and automatically
    chooses a regularization parameter for each target dimension.
    """

    def __init__(self):

        # weight matrix, shape (num_target_dimensions, num_brain_dimensions + 1),
        # includes intercept

        # 1D array of the regularization parameter used for each target dimension
        self.regularization_params = None
        self.models = None
        self.weights = None
        self.intercepts = None

    @property
    def model_type(self):
        return "logistic_reg"

    @property
    def params(self):
        return {
            "weights": self.weights,
            "intercepts": self.intercepts,
            "regularization_params": self.regularization_params
        }

    def train(self, train_examples, train_targets):

        # define range of possible regularization params
        params = [1, 0.1, 10, 0.01, 100, 0.001, 1000, 0.0001, 10000,]

        # find the parameter that works best for each dimension
        min_err_idx = self._choose_reg_params(train_examples,
                                              train_targets,
                                              params)

        self.regularization_params, self.models, self.weights, self.intercepts = \
            self._calculate_weights(train_examples, train_targets, min_err_idx, params)

    @staticmethod
    def _calculate_weights(
            train_examples, train_targets, min_err_idx, params
    ):
        """
        Train a separate regression for each dimension of the target space

        Parameters:
            min_err_idx: indices of the best regularization parameters
            params: range of possible regularization parameters
        """
        num_target_dimensions = train_targets.shape[1]
        num_example_dimensions = train_examples.shape[1]
        r = np.zeros(num_target_dimensions)
        models = []
        weight_matrix = np.zeros((num_example_dimensions, num_target_dimensions))
        intercepts = np.zeros(num_target_dimensions)

        for cur_target in range(num_target_dimensions):
            r[cur_target] = params[min_err_idx[cur_target]]
            model = LogisticRegression(
                penalty="l2", C=r[cur_target], # solver="liblinear"
            ).fit(train_examples, train_targets[:, cur_target])
            models.append(model)
            weight_matrix[:, cur_target] = model.coef_.squeeze()
            intercepts[cur_target] = model.intercept_.item()

        return r, models, weight_matrix, intercepts

    @staticmethod
    def _choose_reg_params(train_examples, train_targets, params):
        """
        Find the best regularization parameter for each dimension

        Parameters:
            train_examples: array of shape (num_examples, num_voxels)
            train_targets: array of shape (num_examples, num_dimensions)
            params: range of possible training parameters

        Returns: array of shape (num_target_dimensions)
        """
        logger.info("choose reg_params")
        num_target_dimensions = train_targets.shape[1]
        # cross-validation error
        cv_err = np.zeros((len(params), num_target_dimensions))
        for i in range(len(params)):
            regularization_param = params[i]

            model = MultiOutputClassifier(LogisticRegression(
                penalty="l2", C=regularization_param, solver="liblinear"
            )).fit(train_examples, train_targets)
            probs = model.predict_proba(train_examples)

            for j in range(num_target_dimensions):
                loss = metrics.log_loss(train_targets[:, j], probs[j], labels=[0, 1])
                cv_err[i, j] = loss

        min_err_idx = np.argmin(cv_err, axis=0)
        return min_err_idx

    def test(self, test_examples):
        return np.dot(test_examples, self.weights) + self.intercepts
