

from .._lib.sklearn.tree._classes import BaseDecisionTree


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


    def _fit(self, X, y, sample_weight=None, check_input=True, missing_values_in_feature_mask=None, classes=None):
        return super()._fit(X, y, sample_weight, check_input, missing_values_in_feature_mask, classes)
