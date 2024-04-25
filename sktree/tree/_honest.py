
import copy
import numbers
from math import ceil
from numbers import Integral, Real

import numpy as np
from sklearn.base import ClassifierMixin, MetaEstimatorMixin, _fit_context, clone, is_classifier
from sklearn.utils.validation import check_is_fitted, check_X_y, check_random_state
from sklearn.utils import compute_sample_weight
from scipy.sparse import issparse
from sklearn.utils._param_validation import Hidden, Interval, RealNotInt, StrOptions

import sklearn.tree._criterion as _criterion
import sklearn.tree._splitter as _splitter
from sklearn.tree._tree import DTYPE, DOUBLE
from sklearn.tree._criterion import BaseCriterion
from sklearn.tree._splitter import BaseSplitter
from .._lib.sklearn.tree._classes import BaseDecisionTree
from sklearn.utils.multiclass import (
    _check_partial_fit_first_call,
    check_classification_targets,
)

from sklearn.utils.validation import (
    _assert_all_finite_element_wise,
    _check_sample_weight,
    assert_all_finite,
    check_is_fitted,
)

from sklearn.tree._tree import Tree, DepthFirstTreeBuilder, BestFirstTreeBuilder


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

    _parameter_constraints = (
        BaseDecisionTree._parameter_constraints.copy()
        |  {"honest_fraction": [Interval(Real, 0.0, 1.0, closed="neither")]}
    )



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
        # honest_prior="empirical",
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
        # self.honest_prior = honest_prior
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
        random_state = check_random_state(self.random_state)

        if check_input:
            # Need to validate separately here.
            # We can't pass multi_output=True because that would allow y to be
            # csr.

            # _compute_missing_values_in_feature_mask will check for finite values and
            # compute the missing mask if the tree supports missing values
            check_X_params = dict(
                dtype=DTYPE, accept_sparse="csc", force_all_finite=False
            )
            check_y_params = dict(ensure_2d=False, dtype=None)
            if y is not None or self._get_tags()["requires_y"]:
                X, y = self._validate_data(
                    X, y, validate_separately=(check_X_params, check_y_params)
                )
            else:
                X = self._validate_data(X, **check_X_params)

            missing_values_in_feature_mask = (
                self._compute_missing_values_in_feature_mask(X)
            )
            if issparse(X):
                X.sort_indices()

                if X.indices.dtype != np.intc or X.indptr.dtype != np.intc:
                    raise ValueError(
                        "No support for np.int64 index based sparse matrices"
                    )

            if y is not None and self.criterion == "poisson":
                if np.any(y < 0):
                    raise ValueError(
                        "Some value(s) of y are negative which is"
                        " not allowed for Poisson regression."
                    )
                if np.sum(y) <= 0:
                    raise ValueError(
                        "Sum of y is not positive which is "
                        "necessary for Poisson regression."
                    )

        # Determine output settings
        n_samples, self.n_features_ = X.shape

        # Do preprocessing if 'y' is passed
        is_classification = False
        if y is not None:
            is_classification = is_classifier(self)
            y = np.atleast_1d(y)
            expanded_class_weight = None

            if y.ndim == 1:
                # reshape is necessary to preserve the data contiguity against vs
                # [:, np.newaxis] that does not.
                y = np.reshape(y, (-1, 1))

            self.n_outputs_ = y.shape[1]

            if is_classification:
                check_classification_targets(y)
                y = np.copy(y)

                self.classes_ = []
                self.n_classes_ = []

                if self.class_weight is not None:
                    y_original = np.copy(y)

                y_encoded = np.zeros(y.shape, dtype=int)
                if classes is not None:
                    classes = np.atleast_1d(classes)
                    if classes.ndim == 1:
                        classes = np.array([classes])

                    for k in classes:
                        self.classes_.append(np.array(k))
                        self.n_classes_.append(np.array(k).shape[0])

                    for i in range(n_samples):
                        for j in range(self.n_outputs_):
                            y_encoded[i, j] = np.where(self.classes_[j] == y[i, j])[0][
                                0
                            ]
                else:
                    for k in range(self.n_outputs_):
                        classes_k, y_encoded[:, k] = np.unique(
                            y[:, k], return_inverse=True
                        )
                        self.classes_.append(classes_k)
                        self.n_classes_.append(classes_k.shape[0])

                y = y_encoded

                if self.class_weight is not None:
                    expanded_class_weight = compute_sample_weight(
                        self.class_weight, y_original
                    )

                self.n_classes_ = np.array(self.n_classes_, dtype=np.intp)

            if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
                y = np.ascontiguousarray(y, dtype=DOUBLE)

            if len(y) != n_samples:
                raise ValueError(
                    "Number of labels=%d does not match number of samples=%d"
                    % (len(y), n_samples)
                )

        # set decision-tree model parameters
        max_depth = np.iinfo(np.int32).max if self.max_depth is None else self.max_depth

        if isinstance(self.min_samples_leaf, numbers.Integral):
            min_samples_leaf = self.min_samples_leaf
        else:  # float
            min_samples_leaf = int(ceil(self.min_samples_leaf * n_samples))

        if isinstance(self.min_samples_split, str):
            if self.min_samples_split == "sqrt":
                min_samples_split = max(1, int(np.sqrt(self.n_features_in_)))
            elif self.min_samples_split == "log2":
                min_samples_split = max(1, int(np.log2(self.n_features_in_)))
        elif isinstance(self.min_samples_split, numbers.Integral):
            min_samples_split = self.min_samples_split
        else:  # float
            min_samples_split = int(ceil(self.min_samples_split * n_samples))
            min_samples_split = max(2, min_samples_split)
        min_samples_split = max(min_samples_split, 2 * min_samples_leaf)
        self.min_samples_split_ = min_samples_split

        if isinstance(self.max_features, str):
            if self.max_features == "sqrt":
                max_features = max(1, int(np.sqrt(self.n_features_in_)))
            elif self.max_features == "log2":
                max_features = max(1, int(np.log2(self.n_features_in_)))
        elif self.max_features is None:
            max_features = self.n_features_in_
        elif isinstance(self.max_features, numbers.Integral):
            max_features = self.max_features
        else:  # float
            if self.max_features > 0.0:
                max_features = max(1, int(self.max_features * self.n_features_in_))
            else:
                max_features = 0

        self.max_features_ = max_features

        max_leaf_nodes = -1 if self.max_leaf_nodes is None else self.max_leaf_nodes

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, DOUBLE)

        # Added for honest trees ===============================================
        honest_samples = np.zeros(n_samples, dtype=bool)
        honest_idxs = np.random.choice(
            n_samples, int(self.honest_fraction * n_samples), replace=False
        )
        honest_samples[honest_idxs] = True
        # ======================================================================

        if y is not None and expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight

        # Set min_weight_leaf from min_weight_fraction_leaf
        if sample_weight is None:
            min_weight_leaf = self.min_weight_fraction_leaf * n_samples
        else:
            min_weight_leaf = self.min_weight_fraction_leaf * np.sum(sample_weight)

        # build the actual tree now with the parameters
        self._build_tree(
            X=X,
            y=y,
            sample_weight=sample_weight,
            missing_values_in_feature_mask=missing_values_in_feature_mask,
            min_samples_leaf=min_samples_leaf,
            min_weight_leaf=min_weight_leaf,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_split=min_samples_split,
            max_depth=max_depth,
            random_state=random_state,
            honest_samples=honest_samples,  # Added for honest trees
        )

        return self


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
        honest_samples, # Added for honest trees
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
            Honest sample indicator, dtype is bool
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


        # Update for honesty
        # Need to update splitter instantiation to support the Honest Criterion
        if not isinstance(self.splitter, BaseSplitter):
            splitter = SPLITTERS[self.splitter](
                criterion,
                self.max_features_,
                min_samples_leaf,
                min_weight_leaf,
                random_state,
                monotonic_cst,
            )


        if is_classifier(self):
            self.tree_ = Tree(self.n_features_in_, self.n_classes_, self.n_outputs_)
        else:
            self.tree_ = Tree(
                self.n_features_in_,
                # TODO: tree shouldn't need this in this case
                np.array([1] * self.n_outputs_, dtype=np.intp),
                self.n_outputs_,
            )

        # Use BestFirst if max_leaf_nodes given; use DepthFirst otherwise
        if max_leaf_nodes < 0:
            self.builder_ = DepthFirstTreeBuilder( # need to refactor for honesty
                splitter,
                min_samples_split,
                min_samples_leaf,
                min_weight_leaf,
                max_depth,
                self.min_impurity_decrease,
                self.store_leaf_values,
            )
        else:
            self.builder_ = BestFirstTreeBuilder( # need to refactor for honesty
                splitter,
                min_samples_split,
                min_samples_leaf,
                min_weight_leaf,
                max_depth,
                max_leaf_nodes,
                self.min_impurity_decrease,
                self.store_leaf_values,
            )
        self.builder_.build(
            self.tree_, X, y, sample_weight, missing_values_in_feature_mask
        )

        if self.n_outputs_ == 1 and is_classifier(self):
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

        # Update for honesty. do we prune trees?
        self._prune_tree()