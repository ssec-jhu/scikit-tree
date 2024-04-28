# cython: language_level=3
# cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
from cython.parallel import prange

cimport numpy as cnp

cnp.import_array()

from libc.math cimport isnan
from libcpp.unordered_set cimport unordered_set

from sktree._lib.sklearn.tree._utils cimport RAND_R_MAX, rand_uniform

from ._utils cimport rand_weighted_binary

from numpy import float32 as DTYPE

TREE_LEAF = -1
TREE_UNDEFINED = -2
cdef intp_t _TREE_LEAF = TREE_LEAF
cdef intp_t _TREE_UNDEFINED = TREE_UNDEFINED


cpdef apply_marginal_tree(
    BaseTree tree,
    object X,
    const intp_t[:] marginal_indices,
    intp_t traversal_method,
    unsigned char use_sample_weight,
    object random_state
):
    """Apply a dataset to a marginalized tree.

    Parameters
    ----------
    tree : Tree
        The tree to apply.
    X : ndarray of shape (n_samples, n_features)
        The dataset to apply.
    marginal_indices : ndarray of shape (n_marginals,)
        The indices of the features to marginalize, which
        are columns in ``X``.
    traversal_method : intp_t
        The traversal method to use. 0 for 'random', 1 for
        'weighted'.
    use_sample_weight : unsigned char
        Whether or not to use the weighted number of samples
        in each node.
    random_state : object
        The random number state.

    Returns
    -------
    out : ndarray of shape (n_samples,)
        The indices of the leaf that each sample falls into.
    """
    # Check input
    if not isinstance(X, np.ndarray):
        raise ValueError("X should be in np.ndarray format, got %s" % type(X))

    if X.dtype != DTYPE:
        raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)

    cdef intp_t n_marginals = marginal_indices.shape[0]

    # sklearn_rand_r random number state
    cdef uint32_t rand_r_state = random_state.randint(0, RAND_R_MAX)

    # define a set of all marginal indices
    cdef unordered_set[intp_t] marginal_indices_map

    # check all marginal indices are valid, and also convert to an unordered map
    for i in range(n_marginals):
        if marginal_indices[i] >= X.shape[1]:
            raise ValueError(
                "marginal_indices must be less than X.shape[1]"
            )

        marginal_indices_map.insert(marginal_indices[i])

    # now we will apply the dataset to the tree
    out = _apply_dense_marginal(
        tree,
        X,
        marginal_indices_map,
        traversal_method,
        use_sample_weight,
        &rand_r_state
    )
    return out


cdef void _resample_split_node(
    BaseTree tree,
    Node* node,
    unordered_set[intp_t] marginal_indices_map,
    const float32_t[:, :] X,
    const float64_t[:, ::1] y,
    const float64_t[:] sample_weight,
) noexcept nogil:
    pass


cdef inline cnp.ndarray _apply_dense_marginal(
    BaseTree tree,
    const float32_t[:, :] X,
    unordered_set[intp_t] marginal_indices_map,
    intp_t traversal_method,
    unsigned char use_sample_weight,
    uint32_t* rand_r_state
):
    """Finds the terminal region (=leaf node) for each sample in X.

    Applies dense dataset to the tree and returns the indices of
    the leaf that each sample falls into. This marginalizes out
    features that are not in the marginal indices.

    Parameters
    ----------
    tree : Tree
        The tree to apply.
    X : const ndarray of shape (n_samples, n_features)
        The data matrix.
    marginal_indices_map : unordered_set[intp_t]
        The indices of the features to marginalize, which
        are columns in ``X``.
    traversal_method : intp_t
        The traversal method to use. 0 for 'random', 1 for
        'weighted'.
    use_sample_weight : unsigned char
        Whether or not to use the weighted number of samples
        in each node.
    rand_r_state : uint32_t
        The random number state.
    """
    # Extract input
    cdef const float32_t[:, :] X_ndarray = X
    cdef intp_t n_samples = X.shape[0]
    cdef float32_t X_i_node_feature

    cdef float32_t n_node_samples, n_right_samples, n_left_samples
    cdef float64_t p_left
    cdef intp_t is_left

    # Initialize output
    cdef intp_t[:] out = np.zeros(n_samples, dtype=np.intp)

    # Initialize auxiliary data-structure
    cdef Node* node = NULL
    cdef intp_t i = 0

    with nogil:
        for i in prange(n_samples):
            node = tree.nodes

            # While node not a leaf
            while node.left_child != _TREE_LEAF:
                # XXX: this will only work for axis-aligned features
                if is_element_present(marginal_indices_map, node.feature):
                    if traversal_method == 1:
                        # if the feature is in the marginal indices, then we
                        # will flip a weighted coin to go down the left, or
                        # right child
                        if use_sample_weight:
                            n_node_samples = node.weighted_n_node_samples
                            n_left_samples = tree.nodes[node.left_child].weighted_n_node_samples
                            n_right_samples = tree.nodes[node.right_child].weighted_n_node_samples
                        else:
                            n_node_samples = node.n_node_samples
                            n_left_samples = tree.nodes[node.left_child].n_node_samples
                            n_right_samples = tree.nodes[node.right_child].n_node_samples

                        # compute the probabilities for going left and right
                        p_left = (<float64_t>n_left_samples / n_node_samples)

                        # randomly sample a direction
                        is_left = rand_weighted_binary(p_left, rand_r_state)

                        if is_left:
                            node = &tree.nodes[node.left_child]
                        else:
                            node = &tree.nodes[node.right_child]
                    else:
                        # traversal method is 0, so it is completely random
                        # and defined by a coin-flip
                        p_left = rand_uniform(0, 1, rand_r_state)
                        if p_left <= 0.5:
                            node = &tree.nodes[node.left_child]
                        else:
                            node = &tree.nodes[node.right_child]
                else:
                    X_i_node_feature = tree._compute_feature(X_ndarray, i, node)
                    # ... and node.right_child != _TREE_LEAF:
                    if isnan(X_i_node_feature):
                        if node.missing_go_to_left:
                            node = &tree.nodes[node.left_child]
                        else:
                            node = &tree.nodes[node.right_child]
                    elif X_i_node_feature <= node.threshold:
                        node = &tree.nodes[node.left_child]
                    else:
                        node = &tree.nodes[node.right_child]

            out[i] = <intp_t>(node - tree.nodes)  # node offset

    return np.asarray(out)


cdef inline intp_t is_element_present(unordered_set[intp_t]& my_set, intp_t element) noexcept nogil:
    """Helper function to check presence of element in set."""
    cdef unordered_set[intp_t].iterator it = my_set.find(element)

    if it != my_set.end():
        return 1
    else:
        return 0
