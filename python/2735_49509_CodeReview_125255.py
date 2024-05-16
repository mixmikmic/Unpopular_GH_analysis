from copy import copy

def merge(arr, left_lo, left_hi, right_lo, right_hi, dct):
    startL = left_lo
    startR = right_lo
    N = left_hi-left_lo + 1 + right_hi - right_lo + 1
    aux = [0] * N
    res = []
    for i in range(N):

        if startL > left_hi: 
            aux[i] = arr[startR]
            startR += 1
        elif startR > right_hi:
            aux[i] = arr[startL]
            startL += 1
        elif arr[startL] <= arr[startR]:
            aux[i] = arr[startL]
            startL += 1
            # print aux
        else:
            aux[i] = arr[startR]
            res.append(startL)
            startR += 1
            # print aux

    for index in res:
        for x in range(index, left_hi+1):
            dct[arr[x]] += 1

    for i in range(left_lo, right_hi+1):
        arr[i] = aux[i - left_lo]
    return


def merge_sort(arr, lo, hi, dct):
    mid = (lo+hi) // 2
    if lo <= mid < hi:
        merge_sort(arr, lo, mid, dct)
        merge_sort(arr, mid+1, hi, dct)
        merge(arr, lo, mid, mid+1, hi, dct)
    return

def count_inversion(arr, N):
    lo = 0
    hi = N-1
    dct = {i:0 for i in arr}
    arr2 = copy(arr)
    merge_sort(arr, lo, hi, dct)
    return ' '.join([str(dct[num]) for num in arr2])

from copy import copy
from operator import add

def new_merge(arr, left_lo, left_hi, right_lo, right_hi, out):
    # docstring goes here
    startL = left_lo
    startR = right_lo
    N = left_hi-left_lo + 1 + right_hi - right_lo + 1
    aux = [0] * N
    res = []
    for i in xrange(N):

        if startL > left_hi: 
            aux[i] = arr[startR]
            startR += 1
        elif startR > right_hi:
            aux[i] = arr[startL]
            startL += 1
        elif arr[startL] <= arr[startR]:
            aux[i] = arr[startL]
            startL += 1
            # print aux
        else:
            aux[i] = arr[startR]
            res.append(startL)
            startR += 1
            # print aux

    for index in res:
        sublist_length = left_hi+1 - index
        ones = [1]*sublist_length
        out[index:left_hi+1] = map(add, out[index:left_hi+1], ones)

    for i in xrange(left_lo, right_hi+1):
        arr[i] = aux[i - left_lo]
    return


def new_merge_sort(arr, lo, hi, out):
    # docstring goes here
    mid = (lo+hi) / 2
    if lo <= mid < hi:
        new_merge_sort(arr, lo, mid, out)
        new_merge_sort(arr, mid+1, hi, out)
        new_merge(arr, lo, mid, mid+1, hi, out)
    return

def new_count_inversion(arr):
    N = len(arr)
    lo = 0
    hi = N-1
    out = [0] * N
    arr2 = copy(arr)
    new_merge_sort(arr, lo, hi, out)
    return ' '.join([str(num) for num in out])

from copy import copy
import numpy as np

def d_merge(arr, left_lo, left_hi, right_lo, right_hi, out):
    # docstring goes here
    startL = left_lo
    startR = right_lo
    N = left_hi-left_lo + 1 + right_hi - right_lo + 1
    aux = [0] * N
    res = []
    for i in xrange(N):

        if startL > left_hi: 
            aux[i] = arr[startR]
            startR += 1
        elif startR > right_hi:
            aux[i] = arr[startL]
            startL += 1
        elif arr[startL] <= arr[startR]:
            aux[i] = arr[startL]
            startL += 1
            # print aux
        else:
            aux[i] = arr[startR]
            res.append(startL)
            startR += 1
            # print aux

    for index in res:
            sublist_length = left_hi+1 - index
            out[index:left_hi+1] += np.ones(sublist_length, dtype = int)

    for i in xrange(left_lo, right_hi+1):
        arr[i] = aux[i - left_lo]
    return


def d_merge_sort(arr, lo, hi, out):
    # docstring goes here
    mid = (lo+hi) / 2
    if lo <= mid < hi:
        d_merge_sort(arr, lo, mid, out)
        d_merge_sort(arr, mid+1, hi, out)
        d_merge(arr, lo, mid, mid+1, hi, out)
    return

def d_count_inversion(arr):
    N = len(arr)
    lo = 0
    hi = N-1
    out = np.array(([0] * N))
    arr2 = copy(arr)
    d_merge_sort(arr, lo, hi, out)
    return ' '.join([str(num) for num in out])

arr = [2, 3, 1, 4]
arr2 = [2, 1, 4, 3]
arr3 = [20]
arr4 = [1, 2, 3, 4, 5, 6]
arr5 = [87, 78, 16, 94]
arr6 = [5, 4, 3, 2, 5, 6, 7]

arrs_to_test = [arr, arr2, arr3, arr4, arr5, arr6]

print [d_count_inversion(copy(test)) for test in arrs_to_test]
print [new_count_inversion(copy(test)) for test in arrs_to_test]
print [count_inversion(copy(test), len(test)) for test in arrs_to_test]

get_ipython().magic('timeit [d_count_inversion(copy(test)) for test in arrs_to_test]')
get_ipython().magic('timeit [new_count_inversion(copy(test)) for test in arrs_to_test]')
get_ipython().magic('timeit [count_inversion(copy(test), len(test)) for test in arrs_to_test]')

from random import randint
big_test = [randint(0, 100) for _ in range(10000)]

get_ipython().magic('timeit x = d_count_inversion(copy(big_test))')
get_ipython().magic('timeit x = new_count_inversion(copy(big_test))')
get_ipython().magic('timeit x = count_inversion(copy(big_test), len(big_test))')

get_ipython().magic('load_ext line_profiler')

assert False

get_ipython().magic('lprun -f d_merge d_count_inversion(copy(big_test))')

get_ipython().magic('lprun -f new_merge new_count_inversion(copy(big_test))')

get_ipython().magic('lprun -f merge count_inversion(copy(big_test), len(big_test))')

