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

# LAZY SPEEDUP 2D
# Perfect measurements, 2D Toric, p = 10^-3 and code (1225, 1024, 841, 729, 576, 441, 361, 225, 144, 100, 49, 25) -> (35, 32, 29, 27, 24, 21, 19, 15, 12, 10, 7,5)

# lazy_time = []
# lazy_success = []

# mwpm_time = []
# mwpm_success = []

# uf_time = []
# uf_success = []

# speedup = []

# physical_qubits = []
# chosen_iterations = 100000
# error_rate = 0.001
# for d in [3, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16, 18]:

   
#    code, decoder = initialize((d,d), "toric", "lazy_mwpm", plotting=False, enabled_errors=["pauli"], faulty_measurements=False, initial_states = (0,0))
#    lazy = run(code, decoder, iterations=chosen_iterations, decode_initial=False, error_rates = {"p_bitflip": 0.0, "p_phaseflip": error_rate, "p_bitflip_plaq": 0.0, "p_bitflip_star": 0.0}, benchmark=benchmarker)
#    lazy_time.append(lazy['benchmark']['duration/decode/mean'])
#    lazy_success.append(float(lazy['no_error']/ chosen_iterations))
#    code, decoder = initialize((d,d), "toric", "mwpm", plotting=False, enabled_errors=["pauli"], faulty_measurements=False, initial_states = (0,0))
#    mwpm = run(code, decoder, iterations=chosen_iterations, decode_initial=False, error_rates = {"p_bitflip": 0.0, "p_phaseflip": error_rate, "p_bitflip_plaq": 0.0, "p_bitflip_star": 0.0}, benchmark=benchmarker)
#    mwpm_time.append(mwpm['benchmark']['duration/decode/mean'])
#    mwpm_success.append( float(mwpm['no_error'] / chosen_iterations))
#    code, decoder = initialize((d,d), "toric", "unionfind", plotting=False, enabled_errors=["pauli"], faulty_measurements=False, initial_states = (0,0))
#    uf = run(code, decoder, iterations=chosen_iterations, decode_initial=False, error_rates = {"p_bitflip": 0.0, "p_phaseflip": error_rate, "p_bitflip_plaq": 0.0, "p_bitflip_star": 0.0}, benchmark=benchmarker)
#    uf_time.append(uf['benchmark']['duration/decode/mean'])
#    uf_success.append( float(uf['no_error'] / chosen_iterations))

#    speedup.append(mwpm['benchmark']['duration/decode/mean']/lazy['benchmark']['duration/decode/mean'])
#    physical_qubits.append(4*d*d)
#    print(d)

#    # figure here
# print(speedup)
# print(lazy_success, mwpm_success)
# print(lazy_time)
# print(mwpm_time)
# print(uf_time)


# plt.plot(physical_qubits, lazy_time, 'o-', color = 'cyan', label='Lazier + MWPM')
# plt.plot(physical_qubits, mwpm_time, 'bo-', label='None + MWPM')
# plt.plot(physical_qubits, uf_time, 'o-', color = 'red', label='None + UF')
# plt.xlim(0, 1350)
# plt.ylim(5 * 10**(-5), 10**(-2))
# plt.grid(True)
# plt.xlabel('Number of physical qubits')
# plt.ylabel('Execution time')
# plt.title('pZ = 0.02')
# plt.yscale('log')
# plt.legend()
# plt.show()

# LAZY SPEEDUP 3D
# Faulty measurements, 3D Toric, p = 10^-3 for both data and ancilla qubits, and code (1225, 1024, 841, 729, 576, 441, 361, 225, 144, 100, 49, 25) -> (35, 32, 29, 27, 24, 21, 19, 15, 12, 10, 7,5)

# lazy_time = []
# lazy_success = []

# mwpm_time = []
# mwpm_success = []

# speedup = []

# physical_qubits = []
# chosen_iterations = 10
# error_rate = 0.001

# for d in [3, 4, 5, 7, 9]:

#    code, decoder = initialize((d,d), "toric", "lazy_mwpm", plotting=False, enabled_errors=["pauli"], faulty_measurements=False, initial_states = (0,0))
#    lazy = run(code, decoder, iterations=chosen_iterations, decode_initial=False, error_rates = {"p_bitflip": 0.0, "p_phaseflip": error_rate, "p_bitflip_plaq": 0.0, "p_bitflip_star": error_rate}, benchmark=benchmarker)
#    lazy_time.append(lazy['benchmark']['duration/decode/mean'])
#    lazy_success.append(float(lazy['no_error']/ chosen_iterations))
#    code, decoder = initialize((d,d), "toric", "mwpm", plotting=False, enabled_errors=["pauli"], faulty_measurements=False, initial_states = (0,0))
#    mwpm = run(code, decoder, iterations=chosen_iterations, decode_initial=False, error_rates = {"p_bitflip": 0.0, "p_phaseflip": error_rate, "p_bitflip_plaq": 0.0, "p_bitflip_star": error_rate}, benchmark=benchmarker)
#    mwpm_time.append(mwpm['benchmark']['duration/decode/mean'])
#    mwpm_success.append( float(mwpm['no_error'] / chosen_iterations))

#    speedup.append(mwpm['benchmark']['duration/decode/mean']/lazy['benchmark']['duration/decode/mean'])
#    physical_qubits.append(8*d*d*d)
#    print(d)

#    # figure here
# print(speedup)
# print(lazy_success, mwpm_success)

# plt.plot(physical_qubits, lazy_time, 'o-', color = 'cyan', label='Lazier + MWPM')
# plt.plot(physical_qubits, mwpm_time, 'bo-', label='None + MWPM')
# plt.xlabel('Number of physical qubits')
# plt.ylabel('Execution time for pZ = 0.001')
# plt.title('Lazier Decoder as decoder accelerator with faulty measurements')
# plt.yscale('log')
# plt.legend()
# plt.show()

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
    
#    average_time.append(np.average(decoding_time))
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
