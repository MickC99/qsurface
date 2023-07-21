# import sys 
# sys.setrecursionlimit(1000000)
# sys.path.append("c:\\users\\mick9\\qsurface\\")
# print(sys.path)

from qsurface.main import initialize, run, run_multiprocess, run_multiprocess_superoperator, BenchmarkDecoder
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

''' SUPEROPERATOR USAGE '''
# code, decoder = initialize((8,8), "toric", "unionfind", plotting=False, superoperator_enable=True, sup_op_file="./data/phenomenological/phenomenological_0.0081995_0.0081995_0.032_0.032_toric.csv", initial_states=(0,0))
# benchmarker = BenchmarkDecoder({
#         "decode": ["duration", "value_to_list"],
#         "correct_edge": "count_calls",})
# print(run(code, decoder, iterations=20, decode_initial=False, benchmark=benchmarker, seed=59))



''' PHENOMENOLOGICAL ORIGINAL QSURFACE '''
# code, decoder = initialize((10,10), "toric", "unionfind", enabled_errors=["pauli"], plotting=False, initial_states=(0,0), faulty_measurements=True)
# p_bitflip = 0.0
# p_phaseflip = 0.0
# p_bitflip_plaq = 0.05
# p_bitflip_star = 0.05
# benchmarker = BenchmarkDecoder({
#         "decode": ["duration", "value_to_list"],
#         "correct_edge": "count_calls",})

# # print(run(code, decoder, iterations=1, error_rates={"p_bitflip": p_bitflip, "p_phaseflip": p_phaseflip, "p_bitflip_plaq": p_bitflip_plaq, "p_bitflip_star": p_bitflip_star},benchmark=benchmarker, decode_initial=False))
# if __name__ == "__main__":
#         run_multiprocess(code, decoder, iterations=100,error_rates={"p_bitflip": p_bitflip, "p_phaseflip": p_phaseflip, "p_bitflip_plaq": p_bitflip_plaq, "p_bitflip_star": p_bitflip_star}, decode_initial=False, seed=59, benchmark=benchmarker)
'''####################################################
                MULTI-PROCESSING SUPEROPERATOR
   ####################################################'''

#code, decoder = initialize((8,8), "toric", "unionfind", plotting=False, superoperator_enable=True, sup_op_file="./data/phenomenological/phenomenological_0.0081995_0.0081995_0.032_0.032_toric.csv", initial_states=(0,0))

#benchmarker = BenchmarkDecoder({
#        "decode": ["duration", "value_to_list"],
#        "correct_edge": "count_calls",})
#if __name__ == '__main__':
#        print(run_multiprocess(code, decoder, iterations=20, decode_initial=False, benchmark=benchmarker, seed=59))
#        print(benchmarker.data)
#        # print(run_multiprocess_superoperator(code, decoder, iterations=100, decode_initial=False, benchmark=benchmarker))

'''####################################################
                WEIGHT-X ARCHITECTURES
   ####################################################'''
benchmarker = BenchmarkDecoder({
"decode": ["duration", "value_to_list"],
"correct_edge": "count_calls"})

# code, decoder = initialize((6,6), "toric", "lazy_mwpm", plotting=False, superoperator_enable=True, sup_op_file="./running/phenomenological_wt_0_toric_rates_px_0.03_pz_0.03_pmx_0.03_pmz_0.03.csv", initial_states=(0,0))
# # code, decoder = initialize((6,6), "weight_3_toric", "unionfind", plotting=False, superoperator_enable=True, sup_op_file="./running/phenomenological_wt_3_toric_px_0.025_pz_0.025_prx_0.025_prz_0.025_pmx_0.025_pmz_0.025_ghz_1.csv", initial_states=(0,0))
# # code, decoder = initialize((6,6), "weight_4_toric", "unionfind", plotting=False, superoperator_enable=True, sup_op_file="./running/phenomenological_wt_4_toric_px_0.01_pz_0.01_pmx_0.01_pmz_0.01_ghz_1.csv", initial_states=(0,0))

# error_rate = 0.01
# code, decoder = initialize((4,4), "toric", "parallel_mwpm", plotting=False, enabled_errors=["pauli"], faulty_measurements=True, initial_states = (0,0), layers = 15*4)
# print(run(code, decoder, iterations=1000, decode_initial=False, error_rates = {"p_bitflip": error_rate, "p_phaseflip": error_rate, "p_bitflip_plaq": error_rate, "p_bitflip_star": error_rate}, benchmark=benchmarker))

# print(run(code, decoder, iterations=20, decode_initial=False, benchmark=benchmarker))

# Fo
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

for i in range(2):
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
# #print([pair.edges['z'].nodes for pair in code.data_qubits.values()])
'''####################################################
        WEIGHT-X ARCHITECTURES' VERIFICATION
   ####################################################'''

# ERROR_RATES = [0, 0.001, 0.002, 0.006, 0.008, 0.010, 0.012, 0.014, 0.016, 0.018, 0.020, 0.022, 0.024, 0.026, 0.027, 0.028, 0.029, 0.030, 0.032, 0.034, 0.036]
# SIZES = [(4,4), (6,6), (8,8), (10,10)]
# 0.83 [2]

# iterations = 8
# error_rates = [float(round(x,3)) for x in np.linspace(0.0, 0.12, 30)]
# SIZES = [(4,4)]
# benchmarker = BenchmarkDecoder({
# "decode": ["duration", "value_to_list"],
# "correct_edge": "count_calls"})


# for num, architecture in zip([0,4,3,-1],["weight_0_toric", "weight_4_toric", "weight_3_toric", "toric"]):
#         plot_points = {}
#         for size in SIZES:
#                 plot_points[size] = []
#         file_location = f"./data/weight_{num}_phenomenological_verify/"
#         export_location = f'./data/weight_{num}_phenomenological_verify/computed_data/threshold_{architecture}_superoperator_data.json'
#         files = [f for f in listdir(file_location) if isfile(join(file_location, f))]
#         FILES = [file_location + f for f in files]

#         for size in SIZES:
#                 for rate in error_rates:
#                         if num == -1:
#                                 code, decoder = initialize(size, architecture, "unionfind",enabled_errors=["pauli"], initial_states=(0,0))
#                                 if __name__ == '__main__':
#                                         no_error = run(code, decoder, iterations=iterations, error_rates = {"p_bitflip": rate, "p_phaseflip": rate})["no_error"]
#                                         plot_points[size].append((rate, no_error /iterations))
#                         else:
#                                 super_op = "NA"
#                                 tmp_files_rates = [float(file[file.find("_px_")+len("_px_"):file.rfind("_pz_")]) for file in FILES]
#                                 for f_rate, sup in zip(tmp_files_rates, FILES):
#                                         if rate == f_rate:
#                                                 super_op = sup
#                                 code, decoder = initialize(size, architecture, "unionfind", layers=1, superoperator_enable=True, sup_op_file=super_op, initial_states=(0,0))
#                                 if __name__ == '__main__':
#                                         no_error = run(code, decoder, iterations=iterations)["no_error"]
#                                         plot_points[size].append((rate, no_error /iterations))

#         export_data = pd.DataFrame(plot_points)

#         export_data.to_json(export_location)

