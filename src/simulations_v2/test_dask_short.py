import multiprocessing as mp
import time
from time import sleep
import itertools
import pdb
from dask_chtc import CHTCCluster
from dask.distributed import Client
import socket


def test(word, x, y):
    z = x * y
    sleep(z)
    # print('sleep {}, return {}').format(y, x**3)
    return('got word={}, {} and {}, slept {}'.format(word, x, y, z))


def test2(value):
    sleep(value)
    return('slept for {} seconds'.format(value))


def log_result(result):
    print("Succesfully get callback! With result: ", result)


def first_test(pool):
    results = pool.starmap_async(test, 'word1', itertools.product(range(1, 7), range(1, 3), repeat=1), callback = log_result)
    print('submitted first pool!')

    moar_results = pool.starmap_async(test, 'word2', itertools.product(range(1, 7), range(1, 3), repeat=1), callback = log_result)
    print('submitted second pool!')

    output = results.get()
    print(output)

    moar_output = moar_results.get()
    print(moar_output)


def second_test(pool):
    arr_1 = [1, 2]
    arr_2 = [1, 2, 3, 4, 5]

    results = []

    for i in arr_1:
        for j in arr_2:
            results.append(pool.apply_async(test2, (j,)))

    for result in results:
        result.get()


def dask_test():
    if socket.gethostname() == 'submit3.chtc.wisc.edu':
        # CHTC execution
        cluster = CHTCCluster(job_extra = {"accounting_group": "COVID19_AFIDSI"})
        cluster.adapt(minimum=10, maximum=20)
        client = Client(cluster)
    else:
        # local execution
        client = Client()

    arr_1 = [1, 2]
    arr_2 = [1, 2, 3, 4, 5]

    results = []

    start_time = time.time()

    for i in arr_1:
        for j in arr_2:
            dt = DaskTester(i)
            results.append(client.submit(dt.test2, j))

    # print(client.gather(results))
    for result in results:
        print(result.result())

    print('Whole process took {} seconds'.format(time.time() - start_time))


class DaskTester:
    def __init__(self, value1):
        self.value1 = value1

    def test2(self, value2):
        sleep(value2)
        return('value1 was {}, slept for {} seconds'.format(self.value1, value2))


if __name__ == '__main__':
    # pool = mp.Pool(processes=12)

    # first_test(pool)
    # second_test(pool)
    dask_test()
