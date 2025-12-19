# Implementation of Fuzzy Logic Operations

import numpy as np

# return a new array where each element is the max of the two sets
# i.e. [max(a[0], b[0]), max(a[1], b[1]), ..., max(a[n], b[n])]
def fuzzy_union_or(a, b, operator='max'):
    if len(a) != len(b):
        raise ValueError('Fuzzy sets must have the same length (same universe of discourse).')
    if operator == 'max':
        return np.maximum(a, b)
    else:
        raise NotImplementedError(f'Operator "{operator}" not supported for Fuzzy Union.')

# return a new array where each element is the min of the two sets
# i.e. [min(a[0], b[0]), min(a[1], b[1]), ..., min(a[n], b[n])]
def fuzzy_intersection_and(a, b, operator='min'):
    if len(a) != len(b):
        raise ValueError('Fuzzy sets must have the same length (same universe of discourse).')
    if operator == 'min':
        return np.minimum(a, b)
    else:
        raise NotImplementedError(f'Operator "{operator}" not supported for Fuzzy Intersection.')

# return a new array where each element is the [1 - element] of the two sets
# i.e. [(1 - a[0]), (1 - a[1]), ..., (1 - a[n])]
def fuzzy_complement_not(a):
    return 1 - a

def main():
    u = np.array([1, 2, 3, 4, 5])
    print(f'universe of discourse (U): {u}')
    a = np.array([1.0, 0.8, 0.4, 0.1, 0.0])
    b = np.array([0.0, 0.1, 0.3, 0.7, 1.0])
    print('original sets')
    print(f'fuzzy set A: {a}')
    print(f'fuzzy set B: {b}')
    a_or_b = fuzzy_union_or(a, b)
    print('fuzzy union (A OR B)')
    print(f'operation: max(mu_a(x), mu_b(x)) = {a_or_b}')
    a_and_b = fuzzy_intersection_and(a, b)
    print('fuzzy intersection (A AND B)')
    print(f'operation: min(mu_a(x), mu_b(x)) = {a_and_b}')
    not_a = fuzzy_complement_not(a)
    print('fuzzy complement (NOT A)')
    print(f'operation: 1 - mu_a(x): {not_a}')
    not_b = fuzzy_complement_not(b)
    print('fuzzy complement (NOT B)')
    print(f'operation: 1 - mu_b(x): {not_b}')
    return None

if __name__ == '__main__':
    main()

# OUTPUT

# universe of discourse (U): [1 2 3 4 5]
# original sets
# fuzzy set A: [1.  0.8 0.4 0.1 0. ]
# fuzzy set B: [0.  0.1 0.3 0.7 1. ]
# fuzzy union (A OR B)
# operation: max(mu_a(x), mu_b(x)) = [1.  0.8 0.4 0.7 1. ]
# fuzzy intersection (A AND B)
# operation: min(mu_a(x), mu_b(x)) = [0.  0.1 0.3 0.1 0. ]
# fuzzy complement (NOT A)
# operation: 1 - mu_a(x): [0.  0.2 0.6 0.9 1. ]
# fuzzy complement (NOT B)
# operation: 1 - mu_b(x): [1.  0.9 0.7 0.3 0. ]