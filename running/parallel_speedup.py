import sys 
sys.setrecursionlimit(1000000)

from qsurface.main import initialize, run, run_multiprocess, run_multiprocess_superoperator, BenchmarkDecoder
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


stats_list_normal = []
stats_list_parallel = []
error_rate = 0.001
for d in [7,9,11,13]:
   list_of_iterations = [20,40,60,80,100,120,140,160,180,200]
   parallel_iteration_average = []
   normal_iteration_average = []
   for current_iterations in list_of_iterations:
      parallel_speed = []
      parallel_error = []
      normal_speed = []
      normal_error = []
      for i in range(10000):
         benchmarker = BenchmarkDecoder({"decode": ["duration", "value_to_list"],"correct_edge": "count_calls"})
         code, decoder = initialize((d,d), "toric", "parallel_mwpm", plotting=False, enabled_errors=["pauli"], faulty_measurements=True, initial_states = (0,0), layers = 15*d)
         parallel = run(code, decoder, iterations=current_iterations, decode_initial=False, error_rates = {"p_bitflip": error_rate, "p_phaseflip": error_rate, "p_bitflip_plaq": error_rate, "p_bitflip_star": error_rate}, benchmark=benchmarker)
         parallel_speed.append(parallel['benchmark']['duration/decode/mean'])
         parallel_error.append(parallel['no_error']/current_iterations)
         benchmarker2 = BenchmarkDecoder({"decode": ["duration", "value_to_list"],"correct_edge": "count_calls"})
         code, decoder = initialize((d,d), "toric", "mwpm", plotting=False, enabled_errors=["pauli"], faulty_measurements=True, initial_states = (0,0), layers = 15*d)
         normal = run(code, decoder, iterations=current_iterations, decode_initial=False, error_rates = {"p_bitflip": error_rate, "p_phaseflip": error_rate, "p_bitflip_plaq": error_rate, "p_bitflip_star": error_rate}, benchmark=benchmarker2)
         normal_speed.append(normal['benchmark']['duration/decode/mean'])
         normal_error.append(normal['no_error']/current_iterations)
      parallel_iteration_average.append([np.average(parallel_speed), np.average(parallel_error)])
      normal_iteration_average.append([np.average(normal_speed), np.average(normal_error)])
   stats_list_parallel.append(parallel_iteration_average)
   stats_list_normal.append(normal_iteration_average)

parallel_times = [[item[0] for item in sublist] for sublist in stats_list_parallel]
normal_times = [[item[0] for item in sublist] for sublist in stats_list_normal]
parallel_error = [[item[1] for item in sublist] for sublist in stats_list_parallel]
normal_error = [[item[1] for item in sublist] for sublist in stats_list_normal]

print(parallel_times)
print(parallel_error)
print(normal_times)
print(normal_error)

for i in range(len(list_of_iterations)):
   print(np.average(normal_times[i])/np.average(parallel_times[i]))

cmap = plt.get_cmap('tab10')

for i in range(len(list_of_iterations)):
   colour = cmap(i)  # Get a unique color from the color map for each line
   plt.plot(list_of_iterations, normal_error[i], 'o-', color = colour, label='Normal ' + str(7 + 2*i) + 'x' + str(7 + 2*i))
   plt.plot(list_of_iterations, parallel_error[i], 'o--', color = colour, label='Parallel ' + str(7 + 2*i) + 'x' + str(7 + 2*i))

plt.xlabel('Number of rounds')
plt.ylabel('Decoding Time')
plt.yscale('log')
plt.legend()
plt.show()