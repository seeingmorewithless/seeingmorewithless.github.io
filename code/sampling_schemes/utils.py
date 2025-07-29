import time as t
import hdf5storage
import numpy as np

def time(f, log="",  *args, **kwargs):
    start = t.time()
    result = f(*args, **kwargs)
    elapsed = t.time() - start
    if log != "":
        print(log)
    print("Test elapsed time is: " + str(elapsed) + " seconds")
    return result

def  read_mat(filename, keys):
    return hdf5storage.loadmat("/home/projects/bagon/dannyh/work/code/docker/pytorch/variable_resolution/matlab/" + filename, variable_names=keys)
    # return hdf5storage.loadmat("/home/avico/danyh_project_code/" + filename, variable_names=keys)

def apply_kernel(image, pixel_idxs, kernel):
    channels = list(map(lambda a: a.reshape(image.shape[0], image.shape[1]), np.dsplit(image, 3)))
    for x in range(3):
        channels[x] = np.sum(np.multiply(channels[x][pixel_idxs], kernel))
    return np.dstack(channels)

class Timer:

    def __init__(self):
        self.total_elapsed = 0

    def time(self, f, positional_arguments, keyword_arguments, log=""):
        start = t.time()
        result = f(*positional_arguments, **keyword_arguments)
        self.total_elapsed += t.time() - start
        if log != "":
            print(log)
        return result

    def print_sum(self, log=""):
        if log != "":
            print(log)
        print("Test elapsed time is: " + str(self.total_elapsed) + " seconds")
