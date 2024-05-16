import numpy as np
from copy import deepcopy

def find_max_sub_array_sum_only(a):
    if a == None: 
        return None
    l = len(a)
    if l < 2: 
        return sum(l)
    cur_sum = 0
    max_sum = 0
    for idx,val in enumerate(a):
        if cur_sum <= 0:
            cur_sum = val
        else:
            cur_sum += val
        
        if cur_sum > max_sum:
            max_sum = cur_sum
        
    return max_sum

def find_max_sub_array(a):
    if a == None: 
        return None
    l = len(a)
    if l < 2: 
        return sum(l)
    cur_sum, max_sum = (0, 0)
    cur_array, max_array = ([], [])
    for idx,val in enumerate(a):
        if cur_sum <= 0:
            cur_sum = val
            cur_array = [idx]
        else:
            cur_sum += val
            cur_array.append(idx)
        
        if cur_sum > max_sum:
            max_sum = deepcopy(cur_sum)
            max_array = deepcopy(cur_array)
    
    return [a[i] for i in max_array]


a = [1, -2, 3, 10, -4, 7, 2, -5]
max_sub_array = [3, 10, -4, 7, 2]

print find_max_sub_array(a) == max_sub_array
print find_max_sub_array(a)

# run


def count_path_rec(mat, m, n, k):
    # base case
    if (m < 0) or (n < 0):
        return 0
    elif (m==n==0) and (k >= mat[m][n]): 
        return 1
    else:
        return count_path_rec(mat, m-1, n, k - mat[m][n]) + count_path_rec(mat, m, n-1, k - mat[m][n])

def path_count(mat, k):
    m = len(mat) - 1
    n = len(mat[0]) - 1
    if m == n == 0:
        return 1 if mat[0][0] == k else 0

    return count_path_rec(mat, m, n, k)

mat = [[1, 2, 3],
       [4, 6, 5],
       [3, 2, 1]]

tests = [
    path_count(mat,10) == 0,
    path_count(mat,11) == 1,
    path_count(mat,12) == 3,
    path_count(mat,13) == 3,
    path_count(mat,14) == 4,
    path_count(mat,15) == 5,
    path_count(mat,16) == 5,
    path_count(mat,17) == 6 ]

print all(tests)


def find_max_pos_sub_mat(M):
    R = len(mat)
    C = len(mat[0])
    S = deepcopy(mat)
    
    maximum = 0
    for r in range(1,R):
        for c in range(1,C):
            # increase the counter if current is one, 
            # left, above, and left_above coutners are all non-zero
            if M[r][c]:
                S[r][c] = min(S[r][c-1], S[r-1][c], S[r-1][c-1]) + 1
            else:
                S[r][c] = 0
            
            if maximum < S[r][c]:
                maximum = S[r][c]
                max_pos = (r,c)

    top_left = 9
    return (S, maximum, max_pos)


mat =  [[0,1,1,0,1],
        [1,1,0,1,0],
        [0,1,1,1,0],
        [1,1,1,1,0],
        [1,1,1,1,1],
        [0,0,0,0,0]]

# solution: (2,1) to (4,3)
S, maximum, max_pos = find_max_pos_sub_mat(mat)   
print np.array(S)
print maximum
print max_pos


def build_power_set(in_set):
    out = [[]] # start with empty set
    for i in in_set:
        new_sets = []
        for previous_set in out:
            new_sets.append(deepcopy(previous_set) + [i])
        out.extend(new_sets)
    return out

s = list(range(3))
ps = build_power_set(s)
answer = [[], [0], [1], [0, 1], [2], [0, 2], [1, 2], [0, 1, 2]]
print ps == answer



