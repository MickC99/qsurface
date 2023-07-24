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
#          code, decoder = initialize((d,d), "toric", "lazy_parallel_mwpm", plotting=False, enabled_errors=["pauli"], faulty_measurements=True, initial_states = (0,0), layers = 15*d)
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


# lazy_parallel = [{'no_error': 9994, 'benchmark': {'decoded': 19972, 'iterations': 20000, 'seed': 541690.2235515, 'duration/decode/mean': 0.022630969579913653, 'duration/decode/std': 0.00303424993258923, 'count_calls/correct_edge/mean': 20.6593, 'count_calls/correct_edge/std': 4.494243374584871}},
# {'no_error': 9990, 'benchmark': {'decoded': 39875, 'iterations': 40000, 'seed': 545363.6766554, 'duration/decode/mean': 0.057298780069046186, 'duration/decode/std': 0.010311578816358732, 'count_calls/correct_edge/mean': 43.8696, 'count_calls/correct_edge/std': 6.662821912673339}},
# {'no_error': 9953, 'benchmark': {'decoded': 59583, 'iterations': 60000, 'seed': 553025.3297367, 'duration/decode/mean': 0.12911263971048176, 'duration/decode/std': 0.020421418507911772, 'count_calls/correct_edge/mean': 80.0673, 'count_calls/correct_edge/std': 8.989258629609006}},
# {'no_error': 9904, 'benchmark': {'decoded': 79004, 'iterations': 80000, 'seed': 568530.752057, 'duration/decode/mean': 0.28834850373953813, 'duration/decode/std': 0.04567276003790709, 'count_calls/correct_edge/mean': 132.1087, 'count_calls/correct_edge/std': 11.600581205698273}}]

# parallel = [{'no_error': 9994, 'benchmark': {'decoded': 9987, 'iterations': 10000, 'seed': 540608.8947646, 'duration/decode/mean': 0.025571762539958583, 'duration/decode/std': 0.0022396290972056293, 'count_calls/correct_edge/mean': 20.6717, 'count_calls/correct_edge/std': 4.544614297165382}},
# {'no_error': 9990, 'benchmark': {'decoded': 29924, 'iterations': 30000, 'seed': 542765.063874, 'duration/decode/mean': 0.0721932211899315, 'duration/decode/std': 0.01031011779783129, 'count_calls/correct_edge/mean': 43.5533, 'count_calls/correct_edge/std': 6.545193588428076}},
# {'no_error': 9964, 'benchmark': {'decoded': 49721, 'iterations': 50000, 'seed': 547831.4996, 'duration/decode/mean': 0.17773375466050348, 'duration/decode/std': 0.023614996149358667, 'count_calls/correct_edge/mean': 79.7313, 'count_calls/correct_edge/std': 8.926785553042036}},
# {'no_error': 9888, 'benchmark': {'decoded': 69282, 'iterations': 70000, 'seed': 557813.1891957, 'duration/decode/mean': 0.5017733625503606, 'duration/decode/std': 0.055702793514673644, 'count_calls/correct_edge/mean': 131.603, 'count_calls/correct_edge/std': 11.425129802326099}}]

error_rate = 0.001
total_iterations = 10**6
for d in [7,9,11,13,15,17]:
   code, decoder = initialize((d,d), "toric", "parallel_mwpm", plotting=False, enabled_errors=["pauli"], faulty_measurements=True, initial_states = (0,0), layers = 15*d)
   print(run(code, decoder, iterations=total_iterations, decode_initial=False, error_rates = {"p_bitflip": error_rate, "p_phaseflip": error_rate, "p_bitflip_plaq": error_rate, "p_bitflip_star": error_rate}, benchmark=benchmarker))
   code, decoder = initialize((d,d), "toric", "lazy_parallel_mwpm", plotting=False, enabled_errors=["pauli"], faulty_measurements=True, initial_states = (0,0), layers = 15*d)
   print(run(code, decoder, iterations=total_iterations, decode_initial=False, error_rates = {"p_bitflip": error_rate, "p_phaseflip": error_rate, "p_bitflip_plaq": error_rate, "p_bitflip_star": error_rate}, benchmark=benchmarker))
   print(d)


# code_distances = [7,9,11,13,15,17]
# physical_qubits = []
# for i in range(len(code_distances)):
#     physical_qubits.append(15*4*(code_distances[i])**3)
# lazy_parallel_time = []
# lazy_parallel_logical_error = []
# parallel_time = []
# parallel_logical_error = []

# for item in lazy_parallel:
#     running_time = item['benchmark']['duration/decode/mean']
#     logical_error = (total_iterations - item['no_error'])/total_iterations
#     lazy_parallel_time.append(running_time)
#     lazy_parallel_logical_error.append(logical_error)

# for item in parallel:
#     running_time = item['benchmark']['duration/decode/mean']
#     logical_error = (total_iterations - item['no_error'])/total_iterations
#     parallel_time.append(running_time)
#     parallel_logical_error.append(logical_error)
   
# plt.plot(physical_qubits, parallel_time, 'o-', color = 'teal', label='Parallel MWPM, Npar = 4')
# plt.plot(physical_qubits, lazy_parallel_time, 'o-', color = 'cyan', label='Lazier + Parallel MWPM, Npar = 4')
# plt.grid(True)
# plt.ylim(10**-2, 10**0)
# plt.xlabel('Number of physical qubits')
# plt.ylabel('Execution time (s)')
# plt.yscale('log')
# plt.legend()
# plt.show()

# plt.plot(physical_qubits, parallel_logical_error, 'o-', color = 'teal', label='Parallel MWPM, Npar = 4')
# plt.plot(physical_qubits, lazy_parallel_logical_error, 'o-', color = 'cyan', label='Lazier + Parallel MWPM, Npar = 4')
# plt.grid(True)
# plt.ylim(5*10**-4, 2*10**-2)
# plt.xlabel('Number of physical qubits')
# plt.ylabel('Logical Error Rate')
# plt.yscale('log')
# plt.legend()
# plt.show()