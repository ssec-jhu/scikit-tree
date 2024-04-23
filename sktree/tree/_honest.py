
import copy
import numpy as np
from sklearn.base import ClassifierMixin, MetaEstimatorMixin, _fit_context, clone, is_classifier
from sklearn.utils.validation import check_is_fitted, check_X_y
from scipy.sparse import issparse

import sklearn.tree._criterion as _criterion
import sklearn.tree._splitter as _splitter
from sklearn.tree._criterion import BaseCriterion
from .._lib.sklearn.tree._classes import BaseDecisionTree

CRITERIA_CLF = {
    "gini": _criterion.Gini,
    "log_loss": _criterion.Entropy,
    "entropy": _criterion.Entropy,
}
CRITERIA_REG = {
    "squared_error": _criterion.MSE,
    "friedman_mse": _criterion.FriedmanMSE,
    "absolute_error": _criterion.MAE,
    "poisson": _criterion.Poisson,
}

DENSE_SPLITTERS = {"best": _splitter.BestSplitter, "random": _splitter.RandomSplitter}

SPARSE_SPLITTERS = {
    "best": _splitter.BestSparseSplitter,
    "random": _splitter.RandomSparseSplitter,
}


class HonestDecisionTree(BaseDecisionTree):



    def __init__(
        self,
        criterion,
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        class_weight=None,
        ccp_alpha=0.0,
        monotonic_cst=None,
        tree_estimator=None,
        honest_fraction=0.5,
        honest_prior="empirical",
    ):
        self.tree_estimator = tree_estimator
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.class_weight = class_weight
        self.random_state = random_state
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.honest_fraction = honest_fraction
        self.honest_prior = honest_prior
        self.monotonic_cst = monotonic_cst

        # XXX: to enable this, we need to also reset the leaf node samples during `_set_leaf_nodes`
        self.store_leaf_values = False


    @_fit_context(prefer_skip_nested_validation=True)
    def fit(
        self,
        X,
        y,
        sample_weight=None,
        check_input=True,
        classes=None,
    ):
        """Build a decision tree classifier from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csc_matrix``.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels) as integers or strings.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. Splits are also
            ignored if they would result in any single class carrying a
            negative weight in either child node.

        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you're doing.

        classes : array-like of shape (n_classes,), default=None
            List of all the classes that can possibly appear in the y vector.
            Must be provided at the first call to partial_fit, can be omitted
            in subsequent calls.

        Returns
        -------
        self : HonestTreeClassifier
            Fitted estimator.
        """
        self._fit(
            X,
            y,
            sample_weight=sample_weight,
            check_input=check_input,
            classes=classes,
        )
        return self



    def _fit(
        self,
        X,
        y,
        sample_weight=None,
        check_input=True,
        missing_values_in_feature_mask=None,
        classes=None
    ):
        """Build an honest tree classifier from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csc_matrix``.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels) as integers or strings.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. Splits are also
            ignored if they would result in any single class carrying a
            negative weight in either child node.

        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        classes : array-like of shape (n_classes,), default=None
            List of all the classes that can possibly appear in the y vector.

        Returns
        -------
        self : HonestTreeClassifier
            Fitted tree estimator.
        """
        rng = np.random.default_rng(self.random_state)
        if check_input:
            X, y = check_X_y(X, y, multi_output=True)

        # Account for bootstrapping too
        if sample_weight is None:
            _sample_weight = np.ones((X.shape[0],), dtype=np.float64)
        else:
            _sample_weight = np.array(sample_weight)

        nonzero_indices = np.where(_sample_weight != 0)[0]





    def _build_tree(
        self,
        X,
        y,
        sample_weight,
        missing_values_in_feature_mask,
        min_samples_leaf,
        min_weight_leaf,
        max_leaf_nodes,
        min_samples_split,
        max_depth,
        random_state,
        honest_samples,
    ):
        """Build the actual tree.

        Parameters
        ----------
        X : Array-like
            X dataset.
        y : Array-like
            Y targets.
        sample_weight : Array-like
            Sample weights
        min_samples_leaf : float
            Number of samples required to be a leaf.
        min_weight_leaf : float
            Weight of samples required to be a leaf.
        max_leaf_nodes : float
            Maximum number of leaf nodes allowed in tree.
        min_samples_split : float
            Minimum number of samples to split on.
        max_depth : int
            The maximum depth of any tree.
        random_state : int
            Random seed.
        honest_samples : Array-like
            Honest sample indicator.
        """

        n_samples = X.shape[0]

        criterion = self.criterion
        if not isinstance(criterion, _criterion.BaseCriterion):
            if is_classifier(self):
                criterion = CRITERIA_CLF[self.criterion](
                    self.n_outputs_, self.n_classes_
                )
            else:
                criterion = CRITERIA_REG[self.criterion](self.n_outputs_, n_samples)
        else:
            # Make a deepcopy in case the criterion has mutable attributes that
            # might be shared and modified concurrently during parallel fitting
            criterion = copy.deepcopy(criterion)

        SPLITTERS = SPARSE_SPLITTERS if issparse(X) else DENSE_SPLITTERS

        if self.monotonic_cst is None:
            monotonic_cst = None
        else:
            if self.n_outputs_ > 1:
                raise ValueError(
                    "Monotonicity constraints are not supported with multiple outputs."
                )
            # Check to correct monotonicity constraint' specification,
            # by applying element-wise logical conjunction
            # Note: we do not cast `np.asarray(self.monotonic_cst, dtype=np.int8)`
            # straight away here so as to generate error messages for invalid
            # values using the original values prior to any dtype related conversion.
            monotonic_cst = np.asarray(self.monotonic_cst)
            if monotonic_cst.shape[0] != X.shape[1]:
                raise ValueError(
                    "monotonic_cst has shape {} but the input data "
                    "X has {} features.".format(monotonic_cst.shape[0], X.shape[1])
                )
            valid_constraints = np.isin(monotonic_cst, (-1, 0, 1))
            if not np.all(valid_constraints):
                unique_constaints_value = np.unique(monotonic_cst)
                raise ValueError(
                    "monotonic_cst must be None or an array-like of -1, 0 or 1, but"
                    f" got {unique_constaints_value}"
                )
            monotonic_cst = np.asarray(monotonic_cst, dtype=np.int8)
            if is_classifier(self):
                if self.n_classes_[0] > 2:
                    raise ValueError(
                        "Monotonicity constraints are not supported with multiclass "
                        "classification"
                    )
                # Binary classification trees are built by constraining probabilities
                # of the *negative class* in order to make the implementation similar
                # to regression trees.
                # Since self.monotonic_cst encodes constraints on probabilities of the
                # *positive class*, all signs must be flipped.
                monotonic_cst *= -1

