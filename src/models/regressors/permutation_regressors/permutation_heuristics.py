import itertools
import math
import numpy as np


def permutation_by_block(current_block, block):
    current_block = np.array(current_block).astype(int)
    block = np.array(block).astype(int)

    def sample_permutations(n_sample, n):
        block_permutation = permutation_matrix(block)
        block_permutation_choice = np.random.choice(np.arange(block_permutation.shape[0]),
                                                    size=(n_sample,),
                                                    replace=False)
        permutation = block_permutation[block_permutation_choice, :]
        tiled_current_block = np.tile(current_block[np.newaxis, :], reps=(permutation.shape[0], 1))
        return np.concatenate([tiled_current_block, permutation], axis=1)

    return sample_permutations

def permutation_by_blockv2(current_block, block):
    current_block = np.array(current_block).astype(int)
    block = np.array(block).astype(int)

    def sample_permutations(n_sample, n):
        # samples 2n permutations
        block_permutation = permutation_matrix(block)
        block_permutation_choice = np.random.choice(np.arange(block_permutation.shape[0]),
                                                    size=(n_sample,),
                                                    replace=False)
        permutation = block_permutation[block_permutation_choice, :]
        tiled_current_block = np.tile(current_block[np.newaxis, :], reps=(permutation.shape[0], 1))
        first = np.concatenate([tiled_current_block, permutation], axis=1)
        second = np.concatenate([permutation, tiled_current_block], axis=1)
        return np.concatenate([first, second], axis=0)

    return sample_permutations

 
def permutation_matrix(elements):
    perm_its = itertools.permutations(elements)
    n = len(elements)
    n_factorial = math.factorial(n)
    perm_ar = np.zeros((n_factorial, n), dtype=int)

    for i in range(n_factorial):
        perm_ar[i, :] = next(perm_its)
    return perm_ar.astype(int)

def permutation_with_exclusion(elements, excluded_elements):
    perm_its = itertools.permutations(elements)
    n = len(elements)
    n_factorial = math.factorial(n)
    n_excluded = excluded_elements.shape[0] 
    perm_ar = np.zeros((n_factorial - n_excluded, n), dtype=int)

    i = 0
    while i < n_factorial - n_excluded:
        candidate = np.array(next(perm_its)).reshape(1, n)
        tiled_candidate = np.tile(candidate, reps=(n_excluded, 1))
        if np.prod(np.linalg.norm(excluded_elements - tiled_candidate, axis=1)) != 0:
            perm_ar[i] = candidate
            i += 1

    return perm_ar



def k_random_permutations(k, head_seed=None, tail_seed=None):
    def sample_permutations(n_sample, n):
        a = np.arange(n)
        
        permutations = np.zeros((n_sample, n),
                                dtype=int)
        # First sample the tail of the permutation i.e n - k random elements
        np.random.seed(seed=tail_seed)
        tail = np.random.choice(a=a,
                                size=(n-k,),
                                replace=False)
        permutations[:, k:] = np.tile(tail[np.newaxis, :],
                                      reps=(n_sample, 1))
        head = np.array([i for i in a if i not in tail])
        head_permutations = permutation_matrix(head)

        np.random.seed(seed=head_seed)
        choice = np.random.choice(np.arange(head_permutations.shape[0]),
                                  size=(n_sample,),
                                  replace=False)
        permutations[:, :k] = head_permutations[choice, :]
        return permutations

    return sample_permutations

def only_answer(answer):
    def sample_permutations(n_sample, n):
        return np.reshape(SUPPORTED_ANSWERS[answer](n), (1, n))
    return sample_permutations

def random_permutations():
    def sample_permutations(n_sample, n):
        permutations = permutation_matrix(np.arange(n))
        choice = np.random.choice(np.arange(permutations.shape[0]),
                                  size=(n_sample,),
                                  replace=False)
        return permutations[choice, :]
    return sample_permutations

def random_permutationsv2():
    def sample_permutations(n_sample, n):
        permutation_matrix = np.zeros((0, n), dtype=int)
        while permutation_matrix.shape[0] < n_sample:
            permutation = np.random.permutation(n).reshape((1, n))
            if not(isinmatrix(permutation, permutation_matrix)) or permutation_matrix.shape[0] == 0:
                permutation_matrix = np.concatenate([permutation_matrix, permutation], axis=0)
        return permutation_matrix.astype(int)
    return sample_permutations

def random_with_exclusion(excluded_elements):
    def sample_permutations(n_sample, n):
        permutation_matrix = np.zeros((0, n), dtype=int)
        while permutation_matrix.shape[0] < n_sample:
            permutation = np.random.permutation(n).reshape((1, n))
            new_permutation = ((not(isinmatrix(permutation,
                                               permutation_matrix))
                                and  
                                not(isinmatrix(permutation,
                                               excluded_elements)))
                               or
                               permutation_matrix.shape[0] == 0)
            if new_permutation:
                permutation_matrix = np.concatenate([permutation_matrix, permutation], axis=0)
        return permutation_matrix.astype(int)
    return sample_permutations

def random_with_answerv2(answer):
    def sample_permutations(n_sample, n):
        answer_instance = SUPPORTED_ANSWERS[answer](n)
        res = np.zeros((n_sample, n))
        res[0, :] = answer_instance

        remaining_permutations = random_with_exclusion(excluded_elements=answer_instance.reshape(1, n))(n_sample - 1,
                                                                                                        n)
        res[1:, :] = remaining_permutations
        return res.astype(int)
    return sample_permutations

        

def random_with_answer(answer):
    def sample_permutations(n_sample, n):
        answer_instance = SUPPORTED_ANSWERS[answer](n)

        res = np.zeros((n_sample, n))
        res[0, :] = answer_instance

        permutations = permutation_with_exclusion(np.arange(n),
                                                  answer_instance.reshape(1, n))

        choice = np.random.choice(np.arange(permutations.shape[0]),
                                  size=(n_sample - 1,),
                                  replace=False)
        res[1:, :] = permutations[choice, :]

        return res.astype(int)

    return sample_permutations

def random_with_identity():
    return random_with_answer('identity')

def sample_with_heuristic(permutation_heuristic):
    heuristic_type = permutation_heuristic.pop('type')
    return SUPPORTED_HEURISTICS[heuristic_type](**permutation_heuristic)

def identity_answer(n):
    return np.arange(n)

def lastf_identity_answer(n):
    return np.concatenate([[n-1], np.arange(n-1)])

def isinmatrix(x, matrix):
    return np.any(np.all(matrix - x == 0, axis=1))
    



SUPPORTED_ANSWERS = {"identity": identity_answer,
                     "lastf": lastf_identity_answer}


SUPPORTED_HEURISTICS = {"random_old": random_permutations,
                        "random": random_permutationsv2,
                        "with_identity": random_with_identity,
                        "k-random": k_random_permutations,
                        "with_answer": random_with_answerv2,
                        "block": permutation_by_block,
                        "blockv2": permutation_by_blockv2,
                        "only_answer": only_answer}

if __name__ == '__main__':
    n = 40
    n_sample = 120
    excluded_elements = np.arange(n).reshape((1, n))
    random_permutations = random_permutationsv2()(n_sample, n)
    print(random_permutations)
