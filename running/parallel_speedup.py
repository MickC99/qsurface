import sys 
sys.setrecursionlimit(1000000)

from qsurface.main import initialize, run, run_multiprocess, run_multiprocess_superoperator, BenchmarkDecoder
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# stats_list_normal = []
# stats_list_parallel = []
# error_rate = 0.001
# for d in [7,9,11,13]:
#    list_of_iterations = [20,40,60,80,100,120,140,160,180,200]
#    parallel_iteration_average = []
#    normal_iteration_average = []
#    for current_iterations in list_of_iterations:
#       parallel_speed = []
#       parallel_error = []
#       normal_speed = []
#       normal_error = []
#       for i in range(10000):
#          benchmarker = BenchmarkDecoder({"decode": ["duration", "value_to_list"],"correct_edge": "count_calls"})
#          code, decoder = initialize((d,d), "toric", "parallel_mwpm", plotting=False, enabled_errors=["pauli"], faulty_measurements=True, initial_states = (0,0), layers = 15*d)
#          parallel = run(code, decoder, iterations=current_iterations, decode_initial=False, error_rates = {"p_bitflip": error_rate, "p_phaseflip": error_rate, "p_bitflip_plaq": error_rate, "p_bitflip_star": error_rate}, benchmark=benchmarker)
#          parallel_speed.append(parallel['benchmark']['duration/decode/mean'])
#          parallel_error.append(parallel['no_error']/current_iterations)
#          benchmarker2 = BenchmarkDecoder({"decode": ["duration", "value_to_list"],"correct_edge": "count_calls"})
#          code, decoder = initialize((d,d), "toric", "mwpm", plotting=False, enabled_errors=["pauli"], faulty_measurements=True, initial_states = (0,0), layers = 15*d)
#          normal = run(code, decoder, iterations=current_iterations, decode_initial=False, error_rates = {"p_bitflip": error_rate, "p_phaseflip": error_rate, "p_bitflip_plaq": error_rate, "p_bitflip_star": error_rate}, benchmark=benchmarker2)
#          normal_speed.append(normal['benchmark']['duration/decode/mean'])
#          normal_error.append(normal['no_error']/current_iterations)
#       parallel_iteration_average.append([np.average(parallel_speed), np.average(parallel_error)])
#       normal_iteration_average.append([np.average(normal_speed), np.average(normal_error)])
#    stats_list_parallel.append(parallel_iteration_average)
#    stats_list_normal.append(normal_iteration_average)

# parallel_times = [[item[0] for item in sublist] for sublist in stats_list_parallel]
# normal_times = [[item[0] for item in sublist] for sublist in stats_list_normal]
# parallel_error = [[item[1] for item in sublist] for sublist in stats_list_parallel]
# normal_error = [[item[1] for item in sublist] for sublist in stats_list_normal]

# print(parallel_times)
# print(parallel_error)
# print(normal_times)
# print(normal_error)



# cmap = plt.get_cmap('tab10')

# for i in range(len(list_of_iterations)):
#    colour = cmap(i)  # Get a unique color from the color map for each line
#    plt.plot(list_of_iterations, normal_error[i], 'o-', color = colour, label='Normal ' + str(7 + 2*i) + 'x' + str(7 + 2*i))
#    plt.plot(list_of_iterations, parallel_error[i], 'o--', color = colour, label='Parallel ' + str(7 + 2*i) + 'x' + str(7 + 2*i))

# plt.xlabel('Number of rounds')
# plt.ylabel('Decoding Time')
# plt.yscale('log')
# plt.legend()
# plt.show()

error_rate = 0.001
total_iterations = 10**6
for d in [7,9,11,13,15,17]:
   code, decoder = initialize((d,d), "toric", "mwpm", plotting=False, enabled_errors=["pauli"], faulty_measurements=True, initial_states = (0,0), layers = 15*d)
   print(run(code, decoder, iterations=total_iterations, decode_initial=False, error_rates = {"p_bitflip": error_rate, "p_phaseflip": error_rate, "p_bitflip_plaq": error_rate, "p_bitflip_star": error_rate}, benchmark=benchmarker))
   code, decoder = initialize((d,d), "toric", "parallel_mwpm", plotting=False, enabled_errors=["pauli"], faulty_measurements=True, initial_states = (0,0), layers = 15*d)
   print(run(code, decoder, iterations=total_iterations, decode_initial=False, error_rates = {"p_bitflip": error_rate, "p_phaseflip": error_rate, "p_bitflip_plaq": error_rate, "p_bitflip_star": error_rate}, benchmark=benchmarker))
   print(d)



# mwpm = [{'no_error': 10000, 'benchmark': {'decoded': 10000, 'iterations': 10000, 'seed': 465564.5924375, 'duration/decode/mean': 0.03145738436001702, 'duration/decode/std': 0.0048423899382532875, 'count_calls/correct_edge/mean': 20.5845, 'count_calls/correct_edge/std': 4.503383144925602}},
# {'no_error': 10000, 'benchmark': {'decoded': 20000, 'iterations': 20000, 'seed': 467690.5538134, 'duration/decode/mean': 0.11082268265022431, 'duration/decode/std': 0.022542712029402112, 'count_calls/correct_edge/mean': 43.7664, 'count_calls/correct_edge/std': 6.6073013432111605}},
# {'no_error': 10000, 'benchmark': {'decoded': 30000, 'iterations': 30000, 'seed': 472841.3779056, 'duration/decode/mean': 0.32074359020989507, 'duration/decode/std': 0.06816790765673988, 'count_calls/correct_edge/mean': 79.7331, 'count_calls/correct_edge/std': 8.932091826106582}},
# {'no_error': 10000, 'benchmark': {'decoded': 40000, 'iterations': 40000, 'seed': 483608.0056902, 'duration/decode/mean': 0.8965963056702633, 'duration/decode/std': 0.15390248668242087, 'count_calls/correct_edge/mean': 131.4445, 'count_calls/correct_edge/std': 11.256514547141135}}]


# parallel = [{'no_error': 9988, 'benchmark': {'decoded': 9982, 'iterations': 10000, 'seed': 466702.6495418, 'duration/decode/mean': 0.011676745429896983, 'duration/decode/std': 0.0027788149441507707, 'count_calls/correct_edge/mean': 20.5354, 'count_calls/correct_edge/std': 4.524527250442857}},
# {'no_error': 9979, 'benchmark': {'decoded': 19925, 'iterations': 20000, 'seed': 470670.8750108, 'duration/decode/mean': 0.02820307208028389, 'duration/decode/std': 0.008465617827416885, 'count_calls/correct_edge/mean': 43.6896, 'count_calls/correct_edge/std': 6.6286689342582195}},
# {'no_error': 9945, 'benchmark': {'decoded': 29789, 'iterations': 30000, 'seed': 479461.6218261, 'duration/decode/mean': 0.06818751769023948, 'duration/decode/std': 0.018633192308810446, 'count_calls/correct_edge/mean': 79.7676, 'count_calls/correct_edge/std': 8.852287288605131}},
# {'no_error': 9898, 'benchmark': {'decoded': 39476, 'iterations': 40000, 'seed': 498149.6506599, 'duration/decode/mean': 0.14267388325057692, 'duration/decode/std': 0.04221741368077295, 'count_calls/correct_edge/mean': 131.7586, 'count_calls/correct_edge/std': 11.427402418747665}}]

# code_distances = [7,9,11,13,15,17]
# physical_qubits = []
# for i in range(len(code_distances)):
#     physical_qubits.append(15*4*(code_distances[i])**3)
# mwpm_time = []
# mwpm_logical_error = []
# parallel_time = []
# parallel_logical_error = []

# for item in mwpm:
#     running_time = item['benchmark']['duration/decode/mean']
#     logical_error = (total_iterations - item['no_error'])/total_iterations
#     mwpm_time.append(running_time)
#     mwpm_logical_error.append(logical_error)

# for item in parallel:
#     running_time = item['benchmark']['duration/decode/mean']
#     logical_error = (total_iterations - item['no_error'])/total_iterations
#     parallel_time.append(running_time)
#     parallel_logical_error.append(logical_error)
   
# plt.plot(physical_qubits, parallel_time, 'o-', color = 'teal', label='Parallel MWPM, Npar = 4')
# plt.plot(physical_qubits, mwpm_time, 'bo-', label='MWPM')
# plt.grid(True)
# plt.xlabel('Number of physical qubits')
# plt.ylabel('Execution time (s)')
# plt.yscale('log')
# plt.legend()
# plt.show()

# plt.plot(physical_qubits, parallel_logical_error, 'o-', color = 'teal', label='Parallel MWPM, Npar = 4')
# plt.plot(physical_qubits, mwpm_logical_error, 'bo-', label='MWPM')
# plt.grid(True)
# plt.xlabel('Number of physical qubits')
# plt.ylabel('Logical Error Rate')
# plt.legend()
# plt.show()