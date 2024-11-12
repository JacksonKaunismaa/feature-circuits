###### utilities for dealing with sparse COO tensors ######
import torch as t


def flatten_index(idxs, shape):
    """
    index : a tensor of shape [n, len(shape)]
    shape : a shape
    return a tensor of shape [n] where each element is the flattened index
    """
    idxs = idxs.t()
    # get strides from shape
    strides = [1]
    for i in range(len(shape)-1, 0, -1):
        strides.append(strides[-1]*shape[i])
    strides = list(reversed(strides))
    strides = t.tensor(strides).to(idxs.device)
    # flatten index
    return (idxs * strides).sum(dim=1).unsqueeze(0)


def prod(l):
    out = 1
    for x in l: out *= x
    return out


def sparse_flatten(x):
    return t.sparse_coo_tensor(
        flatten_index(x.indices(), x.shape),
        x.values(),
        (prod(x.shape),),
        is_coalesced=True
    )


def reshape_index(index, shape):
    """
    index : a tensor of shape [n]
    shape : a shape
    return a tensor of shape [n, len(shape)] where each element is the reshaped index
    """
    multi_index = []
    for dim in reversed(shape):
        multi_index.append(index % dim)
        index //= dim
    multi_index.reverse()
    return t.stack(multi_index, dim=-1)


def sparse_reshape(x, shape):
    """
    x : a sparse COO tensor
    shape : a shape
    return x reshaped to shape
    """
    # first flatten x
    x = sparse_flatten(x)
    new_indices = reshape_index(x.indices()[0], shape)
    return t.sparse_coo_tensor(new_indices.t(), x.values(), shape, is_coalesced=True)


def sparse_mean(x, dim):
    if isinstance(dim, int):
        return x.sum(dim=dim) / x.shape[dim]
    else:
        return x.sum(dim=dim) / prod(x.shape[d] for d in dim)


def sparse_select_last(x, dim):
    # select the last element of the sequence for sparse tensor
    if isinstance(dim, tuple):
        seq_len = x.shape[dim[0]]
        good_mask = x.indices()[dim[0]] == seq_len - 1
        new_shape = [s for i, s in enumerate(x.shape) if i not in dim]
        new_dims = [i for i in range(len(x.shape)) if i not in dim]
        for d in dim[1:]:
            good_mask &= x.indices()[d] == x.shape[d] - 1
    else:
        seq_len = x.shape[dim]
        good_mask = x.indices()[dim] == seq_len - 1
        new_shape = [s for i, s in enumerate(x.shape) if i != dim]
        new_dims = [i for i in range(len(x.shape)) if i != dim]

    values = x.values()[good_mask]
    indices = x.indices()[new_dims][:, good_mask]
    return t.sparse_coo_tensor(indices, values, new_shape, is_coalesced=True)