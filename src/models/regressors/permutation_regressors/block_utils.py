import numpy as np

def expand_to_blocks(permutation_matrix, len_blocks):
    N = permutation_matrix.shape[0]
    permutation_rows = []
    for i in range(N):
        block_rows = []
        # Search for which block is treated the i-th 
        p = 0
        while (p < N) and (permutation_matrix[i][p] == 0):
            p += 1
        B_p = len_blocks[p]
        for i in range(N):
            B_i = len_blocks[i]
            if i != p:
                block_rows.append(np.zeros((B_p, B_i)))
            else:
                block_rows.append(np.identity(B_p))
        permutation_rows.append(np.concatenate(block_rows, axis=1))
    permutation = np.concatenate(permutation_rows, axis=0)
    return permutation

def expand_to_blocks_v2(permutation_matrix, len_block):
    N = permutation_matrix.shape[0]
    permutation_mat = []
    for i in range(N):
        permutation_rows = []
        for j in range(N):
            permutation_rows.append(permutation_matrix[i][j] * np.identity(len_block))
        permutation_mat.append(np.concatenate(permutation_rows, axis=1))
    return np.concatenate(permutation_mat, axis=0)


def relative_block_coords(block):
    return recursive_block(block, 0)[1]

def recursive_block(block, n_el):
    L = []
    print(block)
    for i in range(len(block)):
        if isinstance(block[i], list):
            nblock_el, new_block = recursive_block(block[i], n_el)
            L.append(new_block)
            n_el = nblock_el
        else:
            L.append(n_el)
            n_el += 1
    print(n_el)
    return n_el, L

def sum_block(block, L):
    for i in range(len(block)):
        if isinstance(block[i], list):
            L = sum_block(block[i], L)
        else:
            L.append(block[i])
    return L


if __name__ == '__main__':
    P = np.array([[0, 1],
                  [1, 0]])
    print(expand_to_blocks_v2(P, 2))
