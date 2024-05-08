from multiprocessing import Pool, TimeoutError
import time
import os
from functools import partial

def f(x):
    return x*x

def g(x, y):
    return x*y

if __name__ == "__main__":
    inputs = [(2,3), (3, 4), (4, 5)]
    with Pool(processes=4) as pool:
        # for res in pool.starmap(g, inputs):
        #     print(res)

        h = partial(g, y=3)
        xs = [i[0] for i in inputs]
        print(xs)

        for res in pool.imap_unordered(h, xs):
            print(res)


# if __name__ == '__main__':
#     # start 4 worker processes
#     with Pool(processes=4) as pool:

#         # print "[0, 1, 4,..., 81]"
#         print(pool.map(f, range(10)))

#         # print same numbers in arbitrary order
#         for i in pool.imap_unordered(f, range(10)):
#             print(i)

#         # evaluate "f(20)" asynchronously
#         res = pool.apply_async(f, (20,))      # runs in *only* one process
#         print(res.get(timeout=1))             # prints "400"

#         # evaluate "os.getpid()" asynchronously
#         res = pool.apply_async(os.getpid, ()) # runs in *only* one process
#         print(res.get(timeout=1))             # prints the PID of that process

#         # launching multiple evaluations asynchronously *may* use more processes
#         multiple_results = [pool.apply_async(os.getpid, ()) for i in range(4)]
#         print([res.get(timeout=1) for res in multiple_results])

#         # make a single worker sleep for 10 seconds
#         res = pool.apply_async(time.sleep, (10,))
#         try:
#             print(res.get(timeout=1))
#         except TimeoutError:
#             print("We lacked patience and got a multiprocessing.TimeoutError")

#         print("For the moment, the pool remains available for more work")

#     # exiting the 'with'-block has stopped the pool
#     print("Now the pool is closed and no longer available")