# import sys 
# sys.setrecursionlimit(1000000)
# sys.path.append("c:\\users\\mick9\\qsurface\\")
# print(sys.path)

from qsurface.main import initialize, run, run_multiprocess, run_multiprocess_superoperator, BenchmarkDecoder
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np

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

# print(run(code, decoder, iterations=20, decode_initial=False, benchmark=benchmarker))


# code, decoder = initialize((5,5), "toric", "lazy_mwpm", plotting=False, enabled_errors=["pauli"], faulty_measurements=False, initial_states = (0,0))
# print(run(code, decoder, iterations=1000, decode_initial=False, error_rates = {"p_bitflip": 0.01, "p_phaseflip": 0.01, "p_bitflip_plaq": 0, "p_bitflip_star": 0}, benchmark=benchmarker))

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

# LAZY SPEEDUP
# Perfect measurements, 2D Toric, p = 10^-3 and code (1225, 1024, 841, 729, 576, 441, 361, 225, 144, 100, 49, 25) -> (35, 32, 29, 27, 24, 21, 19, 15, 12, 10, 7,5)

distance = []
lazy_time = []
mwpm_time = []

for d in [5, 7, 10, 12, 15, 19, 21, 24, 27, 29, 32, 35]:
   
   code, decoder = initialize((d,d), "toric", "lazy_mwpm", plotting=False, enabled_errors=["pauli"], faulty_measurements=False, initial_states = (0,0))
   lazy = run(code, decoder, iterations=1000000, decode_initial=False, error_rates = {"p_bitflip": 0.001, "p_phaseflip": 0.001, "p_bitflip_plaq": 0, "p_bitflip_star": 0}, benchmark=benchmarker)
   lazy_time.append(lazy['benchmark']['duration/decode/mean'])
   code, decoder = initialize((d,d), "toric", "mwpm", plotting=False, enabled_errors=["pauli"], faulty_measurements=False, initial_states = (0,0))
   mwpm = run(code, decoder, iterations=1000000, decode_initial=False, error_rates = {"p_bitflip": 0.001, "p_phaseflip": 0.001, "p_bitflip_plaq": 0, "p_bitflip_star": 0}, benchmark=benchmarker)
   mwpm_time.append(mwpm['benchmark']['duration/decode/mean'])
   print(d)

   # figure here
print(lazy_time)
print(mwpm_time)