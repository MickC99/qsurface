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

# print(run(code, decoder, iterations=20, decode_initial=False, benchmark=benchmarker))


code, decoder = initialize((3,3), "toric", "lazy_mwpm", plotting=False, enabled_errors=["pauli"], faulty_measurements=True, initial_states = (0,0))
print(run(code, decoder, iterations=1, decode_initial=False, error_rates = {"p_bitflip": 0.01, "p_phaseflip": 0.0, "p_bitflip_plaq": 0.01, "p_bitflip_star": 0.0}, benchmark=benchmarker))

#print([pair.edges['z'].nodes for pair in code.data_qubits.values()])
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

# LAZY SPEEDUP 2D
# Perfect measurements, 2D Toric, p = 10^-3 and code (1225, 1024, 841, 729, 576, 441, 361, 225, 144, 100, 49, 25) -> (35, 32, 29, 27, 24, 21, 19, 15, 12, 10, 7,5)

lazy_time = []
lazy_success = []

mwpm_time = []
mwpm_success = []

speedup = []

physical_qubits = []
chosen_iterations = 1000

for d in [3, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16, 18]:
   
   code, decoder = initialize((d,d), "toric", "lazy_mwpm", plotting=False, enabled_errors=["pauli"], faulty_measurements=False, initial_states = (0,0))
   lazy = run(code, decoder, iterations=chosen_iterations, decode_initial=False, error_rates = {"p_bitflip": 0.001, "p_phaseflip": 0.001, "p_bitflip_plaq": 0.0, "p_bitflip_star": 0.0}, benchmark=benchmarker)
   lazy_time.append(lazy['benchmark']['duration/decode/mean'])
   lazy_success.append(float(lazy['no_error']/ chosen_iterations))
   code, decoder = initialize((d,d), "toric", "mwpm", plotting=False, enabled_errors=["pauli"], faulty_measurements=False, initial_states = (0,0))
   mwpm = run(code, decoder, iterations=chosen_iterations, decode_initial=False, error_rates = {"p_bitflip": 0.001, "p_phaseflip": 0.001, "p_bitflip_plaq": 0.0, "p_bitflip_star": 0.0}, benchmark=benchmarker)
   mwpm_time.append(mwpm['benchmark']['duration/decode/mean'])
   mwpm_success.append( float(mwpm['no_error'] / chosen_iterations))

   speedup.append(mwpm['benchmark']['duration/decode/mean']/lazy['benchmark']['duration/decode/mean'])
   physical_qubits.append(4*d*d)
   print(d)

   # figure here
print(speedup)
print(lazy_success, mwpm_success)

plt.plot(physical_qubits, lazy_time, 'bo-', label='Lazy + MWPM')
plt.plot(physical_qubits, mwpm_time, 'ro-', label='None + MWPM')
plt.xlabel('Number of physical qubits')
plt.ylabel('Execution time for pZ = pX = 0.001')
plt.title('Lazy Decoder as decoder accelerator with faulty measurements')
plt.yscale('log')
plt.legend()
plt.show()

# LAZY SPEEDUP
# Perfect measurements, 3D Toric, p = 10^-3 for both data and ancilla qubits, and code (1225, 1024, 841, 729, 576, 441, 361, 225, 144, 100, 49, 25) -> (35, 32, 29, 27, 24, 21, 19, 15, 12, 10, 7,5)

lazy_time = []
lazy_success = []

mwpm_time = []
mwpm_success = []

speedup = []

physical_qubits = []
chosen_iterations = 1000

for d in [3, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16, 18]:
   
   code, decoder = initialize((d,d), "toric", "lazy_mwpm", plotting=False, enabled_errors=["pauli"], faulty_measurements=True, initial_states = (0,0))
   lazy = run(code, decoder, iterations=chosen_iterations, decode_initial=False, error_rates = {"p_bitflip": 0.001, "p_phaseflip": 0.001, "p_bitflip_plaq": 0.001, "p_bitflip_star": 0.001}, benchmark=benchmarker)
   lazy_time.append(lazy['benchmark']['duration/decode/mean'])
   lazy_success.append(float(lazy['no_error']/ chosen_iterations))
   code, decoder = initialize((d,d), "toric", "mwpm", plotting=False, enabled_errors=["pauli"], faulty_measurements=True, initial_states = (0,0))
   mwpm = run(code, decoder, iterations=chosen_iterations, decode_initial=False, error_rates = {"p_bitflip": 0.001, "p_phaseflip": 0.001, "p_bitflip_plaq": 0.001, "p_bitflip_star": 0.001}, benchmark=benchmarker)
   mwpm_time.append(mwpm['benchmark']['duration/decode/mean'])
   mwpm_success.append( float(mwpm['no_error'] / chosen_iterations))

   speedup.append(mwpm['benchmark']['duration/decode/mean']/lazy['benchmark']['duration/decode/mean'])
   physical_qubits.append(4*d*d)
   print(d)

   # figure here
print(speedup)
print(lazy_success, mwpm_success)

plt.plot(physical_qubits, lazy_time, 'bo-', label='Lazy + MWPM')
plt.plot(physical_qubits, mwpm_time, 'ro-', label='None + MWPM')
plt.xlabel('Number of physical qubits')
plt.ylabel('Execution time for pZ = pX = 0.001')
plt.title('Lazy Decoder as decoder accelerator with faulty measurements')
plt.yscale('log')
plt.legend()
plt.show()

# Lazy accuracy and time

# average_time = []
# physical_qubits =[]
# for d in [12, 13, 15, 16, 18]:
#    decoding_time = []
#    decoded = 0

#    for i in range(1000):
#       code, decoder = initialize((d,d), "toric", "lazy_mwpm", plotting=False, enabled_errors=["pauli"], faulty_measurements=True, initial_states = (0,0))
#       result = run(code, decoder, iterations=1, decode_initial=False, error_rates = {"p_bitflip": 0.005, "p_phaseflip": 0.005, "p_bitflip_plaq": 0, "p_bitflip_star": 0}, benchmark=benchmarker)
#       if result['benchmark']['decoded'] == '1':
#          decoded +=1
#       decoding_time.append(result['benchmark']['duration/decode/mean'])
    
#    average_time.append(statistics.mean(decoding_time))
#    physical_qubits.append(4*d*d)
#    print(d)

# plt.plot(physical_qubits, average_time, 'bo-', label='Lazy 3D next()')
# plt.xlabel('Number of physical qubits')
# plt.ylabel('Execution time for pZ = pX = 0.005')
# plt.title('Running time')
# plt.legend()
# plt.show()



# LAZY WEIGHT CHECKING
# TEST TO SEE IF LAZY DECODER FAILS WHEN IT SHOULD: IN CASES WHERE THE WEIGHT IS LARGER THAN THE NUMBER OF VERTICES/2, THERE ARE EDGES WHICH ARE MORE THAN WEIGHT 1
# Comment out ADD1 and ADD2 in sim.py from lazy_mwpm for this test.

# chosen_iterations = 10000
# success = []
# failure = []

# for d in [4, 5, 7, 8, 9, 11, 12, 13, 15, 16, 18]:
   
#    code, decoder = initialize((d,d), "toric", "lazy_mwpm", plotting=False, enabled_errors=["pauli"], faulty_measurements=True, initial_states = (0,0))
#    lazy = run(code, decoder, iterations=chosen_iterations, decode_initial=False, error_rates = {"p_bitflip": 0.0, "p_phaseflip": 0.0, "p_bitflip_plaq": 0.1, "p_bitflip_star": 0}, benchmark=benchmarker)
#    print(d)


# LAZY SUCCESS CHECKING
# For each d, it checks if the lazy decoder actually succeeds in cases it should succeed

# # Calculates manhattan distance for two ancillas in toric code
# def calculate_distance(d, coord1, coord2):
#         (xi, xj, xk) = coord1
#         (yi, yj, yk) = coord2
#         return (min(abs(xi - yi), d - abs(xi - yi)) + min(abs(xj - yj), d - abs(xj - yj)) + abs(xk - yk))

# # Generates a list of ancilla pairs that should be solvable by the lazy decoder
# def generate_pairs(d, N):

#     pairs = []
#     used_items = set()
#     valid_pairs = 0
#     max_attempts = 100

#    # Generate pairs when there are not enough syndromes
#     while valid_pairs < N:
      
#          # Terminate the loop if maximum attempts are reached. To prevent searching for extra errors if not possible
#         if max_attempts == 0:
#             break
         
#          # Randomly selecting ancilla that goes off
#         i = random.randint(0, d - 1)
#         j = random.randint(0, d - 1)
#         k = d-1 #random.randint(0, d-1)

#         item = (i, j, k)
#         neighbor = None

#         # Check if item already exists in the list
#         if item in used_items:
#             max_attempts -= 1
#             continue

#         # Find all possible neighbours
#         if k == 0:
#             neighbors = [
#                 ((i - 1) % d, j, k),
#                 ((i + 1) % d, j, k),
#                 (i, (j - 1) % d, k),
#                 (i, (j + 1) % d, k),
#                 (i, j, k+1)
#             ]
#         elif k == d - 1:
#             neighbors = [
#                 ((i - 1) % d, j, k),
#                 ((i + 1) % d, j, k),
#                 (i, (j - 1) % d, k),
#                 (i, (j + 1) % d, k),
#                 (i, j, k-1)
#             ]
#         else:
#             neighbors = [
#                 ((i - 1) % d, j, k),
#                 ((i + 1) % d, j, k),
#                 (i, (j - 1) % d, k),
#                 (i, (j + 1) % d, k),
#                 (i, j, k-1),
#                 (i, j, k+1)
#             ]
         
#          # Shuffle neighbours so a random pair is created
#         random.shuffle(neighbors)

#         for n in neighbors:
            
#             # Check if neighbour is already an ancilla in the list of syndromes
#             if n in used_items:
#                 max_attempts -= 1
#                 continue

#             # Check if (item, neighbour) does not create impossible ancilla configurations
#             is_neighbor_valid = True
#             for (x, y) in pairs:
#                 if (
#                     (calculate_distance(d, x, item) == 1)
#                     ^ (calculate_distance(d, y, n) == 1)
#                 ) or (
#                     (calculate_distance(d, y, item) == 1)
#                     ^ (calculate_distance(d, x, n) == 1)
#                 ):
#                     is_neighbor_valid = False
#                     break
            
#             if is_neighbor_valid:
#                 neighbor = n
#                 used_items.add(n)
#                 valid_pairs += 1
#                 break
            
#         if neighbor:
#             pairs.append((item, neighbor))
#             used_items.add(item)

#     # Return sorted list of separate ancillas to prevent bias
#     output = [item for pair in pairs for item in pair]
#     return sorted(output)

# def create_dictionary():
#     dictionary = {}

#     # Define the range for i, j, and k
#     i_range = 18
#     j_range = 18
#     k_range = 18

#     # Generate key-value pairs
#     for i in range(i_range):
#         for j in range(j_range):
#             for k in range(k_range):
#                 key = (i, j, k)
#                 value = "A({},{}|{})".format(i, j, k)
#                 dictionary[key] = value

#     return dictionary

# def lazy_testing(syndromes, edges):
#     error_list = []

#     if len(syndromes) == 0:
#         return
    
#     for edge in edges:
#         ancilla1, ancilla2 = edge

#         # Check if both ancilla qubits are in the syndrome set
#         if str(ancilla1) in syndromes and str(ancilla2) in syndromes:
#             error_list.append(edge)
#             syndromes.remove(str(ancilla1))
#             syndromes.remove(str(ancilla2))

#     if syndromes:
#         return "Failure"


# for d in [3, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16, 18]:
#     for i in range(1000):
#         N = random.randint(0,d-2)
#         code, decoder = initialize((d,d), "toric", "lazy_mwpm", plotting=False, enabled_errors=["pauli"], faulty_measurements=True, initial_states = (0,0))
#         n = len(code.data_qubits)
#         plaqs_edges = sum([[code.data_qubits[i][(x, y)].edges['x'].nodes for (x, y) in code.data_qubits[i]] + code.time_edges[i] for i in range(n-1)], []) + [code.data_qubits[n-1][(x, y)].edges['x'].nodes for (x, y) in code.data_qubits[n-1]]
#         ancilla_dictionary = create_dictionary()
#         syndromes = []
#         for item in generate_pairs(d,N):
#             syndromes.append(ancilla_dictionary[item])
#         if lazy_testing(syndromes, plaqs_edges) == "Failure":
#             print(syndromes)
#     print(d)

