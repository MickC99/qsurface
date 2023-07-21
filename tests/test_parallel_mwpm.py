
from qsurface.main import initialize, run, run_multiprocess, run_multiprocess_superoperator, BenchmarkDecoder
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

benchmarker = BenchmarkDecoder({
"decode": ["duration", "value_to_list"],
"correct_edge": "count_calls"})

# Window division testing
# Test if dividing window functions put windows in accurate positions:

# # Tester function for An division of windows
# def test_divide_into_windows(syndromes, d, parallel_processes):
#    windows = {}

#    window_size = 3 * d

#    # O(n)
#    for syndrome in syndromes:
#       window_index = syndrome // (window_size + d)
#       window_point = float(syndrome / (window_size + d))

#       if window_index not in windows:
#             windows[window_index] = []

#       if 0 <= (window_point % 1) < 0.75:
#             windows[window_index].append(syndrome)

#    return windows

# # Tester function for Bn window division
# def test_second_divide_into_windows(syndromes, d, parallel_processes):
#    windows = {}

#    window_size = 3 * d
            
#    # O(n)
#    for syndrome in syndromes:
#       if syndrome < (2/3)*window_size or syndrome > (4*d*parallel_processes - d) - (2/3)*window_size - 1:
#             continue
#       window_index = (syndrome - d) // (window_size+d)
#       window_point = float((syndrome - d) / (window_size+d))

#       if window_index not in windows:
#             windows[window_index] = []

#       if 0.25 <= (window_point % 1) < 1:
#             windows[window_index].append(syndrome)

#    return windows


# for d in [3,4,5,7,9,11]:
#    for i in range(1000):
#       parallel_processes_list = [1,2,4,8,16]
#       parallel_processes = random.choice(parallel_processes_list)
#       syndromes = []

#       # Create syndrome list with numbers as .z values, two for each z value
#       for i in range((parallel_processes*4 - 1)*d):
#          syndromes.append(i)
#          syndromes.append(i)

      

#       A_n = test_divide_into_windows(syndromes, d, parallel_processes)
#       B_n = test_second_divide_into_windows(syndromes, d, parallel_processes)

#       # Verify that all windows are size 3d, all gaps are size d
#       previous_item = 0
#       for item in A_n.values():
         
#          # Windows of size 3d
#          if len(item) != 2*3*d:
#             print("An windows incorrect size")
#          if previous_item == 0:
#                previous_item = item[-1]
#                continue
         
#          # Gaps of size d
#          if item[0] - previous_item - 1 != d:
#             print("An gaps incorrect size")
#          previous_item = item[-1]

#       # Verify that all second windows are size 3d
#       for item in B_n.values():
#          if len(item) != 2*3*d:
#             print("Bn windows incorrect size")

#       # Verify that the number of windows is correct
#       if (len(A_n) != parallel_processes):
#          print("Not enough An windows")
#       if (len(B_n) != parallel_processes - 1):
#          print("Not enough Bn windows")
#    print(d)