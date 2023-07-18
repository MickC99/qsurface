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

# error_rate = 0.02
# code, decoder = initialize((6,6), "toric", "parallel_mwpm", plotting=False, enabled_errors=["pauli"], faulty_measurements=True, initial_states = (0,0), layers = 63*6)
# print(run(code, decoder, iterations=10, decode_initial=False, error_rates = {"p_bitflip": error_rate, "p_phaseflip": error_rate, "p_bitflip_plaq": error_rate, "p_bitflip_star": error_rate}, benchmark=benchmarker))
# print(code.logical_state)

# print(run(code, decoder, iterations=20, decode_initial=False, benchmark=benchmarker))

# Fo
# stats_list_normal = []
# stats_list_parallel = []
# error_rate = 0.001
# for d in [7,9,11,13,15,17]:
#    list_of_iterations = [20,40,60,80,100,120,140,160,180,200]
#    iteration_list_parallel = []
#    iteration_list_normal = []
#    for current_iterations in list_of_iterations:
#       benchmarker = BenchmarkDecoder({"decode": ["duration", "value_to_list"],"correct_edge": "count_calls"})
#       code, decoder = initialize((d,d), "toric", "parallel_mwpm", plotting=False, enabled_errors=["pauli"], faulty_measurements=True, initial_states = (0,0), layers = 63*d)
#       parallel = run(code, decoder, iterations=current_iterations, decode_initial=False, error_rates = {"p_bitflip": error_rate, "p_phaseflip": error_rate, "p_bitflip_plaq": error_rate, "p_bitflip_star": error_rate}, benchmark=benchmarker)
#       iteration_list_parallel.append(parallel)
#       benchmarker2 = BenchmarkDecoder({"decode": ["duration", "value_to_list"],"correct_edge": "count_calls"})
#       code, decoder = initialize((d,d), "toric", "mwpm", plotting=False, enabled_errors=["pauli"], faulty_measurements=True, initial_states = (0,0), layers = 63*d)
#       normal = run(code, decoder, iterations=current_iterations, decode_initial=False, error_rates = {"p_bitflip": error_rate, "p_phaseflip": error_rate, "p_bitflip_plaq": error_rate, "p_bitflip_star": error_rate}, benchmark=benchmarker2)
#       iteration_list_normal.append(normal)
#    stats_list_parallel.append(iteration_list_parallel)
#    stats_list_normal.append(iteration_list_normal)

# error_rate = 0.001
# code, decoder = initialize((4,4), "toric", "lazy_parallel_mwpm", plotting=False, enabled_errors=["pauli"], faulty_measurements=True, initial_states = (0,0), layers = 15*4)
# print(run(code, decoder, iterations=10, decode_initial=False, error_rates = {"p_bitflip": error_rate, "p_phaseflip": error_rate, "p_bitflip_plaq": error_rate, "p_bitflip_star": error_rate}, benchmark=benchmarker))
# print(stats_list_normal)
# print(stats_list_parallel)

normal = [[{'no_error': 20, 'benchmark': {'decoded': 20, 'iterations': 20, 'seed': 237465.2480224, 'duration/decode/mean': 0.03007294500130229, 'duration/decode/std': 0.002574530530179822, 'count_calls/correct_edge/mean': 19.6, 'count_calls/correct_edge/std': 4.431703961232068}}, {'no_error': 40, 'benchmark': {'decoded': 40, 'iterations': 40, 'seed': 237471.8484843, 'duration/decode/mean': 0.034734149999712825, 'duration/decode/std': 0.00446211050311975, 'count_calls/correct_edge/mean': 20.0, 'count_calls/correct_edge/std': 4.674398357008098}}, {'no_error': 60, 'benchmark': {'decoded': 60, 'iterations': 60, 'seed': 237483.1453358, 'duration/decode/mean': 0.0331134133322242, 'duration/decode/std': 0.003775071225181964, 'count_calls/correct_edge/mean': 20.433333333333334, 'count_calls/correct_edge/std': 4.670355684765585}}, {'no_error': 80, 'benchmark': {'decoded': 80, 'iterations': 80, 'seed': 237498.6221737, 'duration/decode/mean': 0.03480725875160715, 'duration/decode/std': 0.005527644992635417, 'count_calls/correct_edge/mean': 19.7625, 'count_calls/correct_edge/std': 4.672375600270167}}, {'no_error': 100, 'benchmark': {'decoded': 100, 'iterations': 100, 'seed': 237518.7906975, 'duration/decode/mean': 0.03613716900086729, 'duration/decode/std': 0.004990227884938333, 'count_calls/correct_edge/mean': 20.95, 'count_calls/correct_edge/std': 4.980712800393133}}, {'no_error': 120, 'benchmark': {'decoded': 120, 'iterations': 120, 'seed': 237543.7851458, 'duration/decode/mean': 0.03485919916759788, 'duration/decode/std': 0.004625738562293597, 'count_calls/correct_edge/mean': 19.958333333333332, 'count_calls/correct_edge/std': 4.412852126333816}}, {'no_error': 140, 'benchmark': {'decoded': 140, 'iterations': 140, 'seed': 237572.9722592, 'duration/decode/mean': 0.03614584285782517, 'duration/decode/std': 0.0045974990642637, 'count_calls/correct_edge/mean': 20.635714285714286, 'count_calls/correct_edge/std': 4.390266367261809}}, {'no_error': 160, 'benchmark': {'decoded': 160, 'iterations': 160, 'seed': 237607.1584016, 'duration/decode/mean': 0.03498275437468692, 'duration/decode/std': 0.003859451155078997, 'count_calls/correct_edge/mean': 20.6, 'count_calls/correct_edge/std': 4.586937976471886}}, {'no_error': 180, 'benchmark': {'decoded': 180, 'iterations': 180, 'seed': 237645.7467693, 'duration/decode/mean': 0.03710762555565452, 'duration/decode/std': 0.004862402991866544, 'count_calls/correct_edge/mean': 19.738888888888887, 'count_calls/correct_edge/std': 4.318389474598255}}, {'no_error': 200, 'benchmark': {'decoded': 200, 'iterations': 200, 'seed': 237690.182721, 'duration/decode/mean': 0.042470946000830734, 'duration/decode/std': 0.010336432415467379, 'count_calls/correct_edge/mean': 21.09, 'count_calls/correct_edge/std': 4.76045165924411}}], [{'no_error': 20, 'benchmark': {'decoded': 20, 'iterations': 20, 'seed': 237725.8128087, 'duration/decode/mean': 0.14385353499965275, 'duration/decode/std': 0.03299465207256629, 'count_calls/correct_edge/mean': 42.45, 'count_calls/correct_edge/std': 6.651879433663843}}, {'no_error': 40, 'benchmark': {'decoded': 40, 'iterations': 40, 'seed': 237745.2110048, 'duration/decode/mean': 0.13418248749730993, 'duration/decode/std': 0.0441809113138226, 'count_calls/correct_edge/mean': 44.725, 'count_calls/correct_edge/std': 6.782283907357462}}, {'no_error': 60, 'benchmark': {'decoded': 60, 'iterations': 60, 'seed': 237773.3864045, 'duration/decode/mean': 0.11523674833054733, 'duration/decode/std': 0.016133033792943747, 'count_calls/correct_edge/mean': 42.96666666666667, 'count_calls/correct_edge/std': 5.958094400803741}}, {'no_error': 80, 'benchmark': {'decoded': 80, 'iterations': 80, 'seed': 237820.6629697, 'duration/decode/mean': 0.15522014500056686, 'duration/decode/std': 0.05723187612105022, 'count_calls/correct_edge/mean': 44.2875, 'count_calls/correct_edge/std': 6.079049576208439}}, {'no_error': 100, 'benchmark': {'decoded': 100, 'iterations': 100, 'seed': 237879.3686971, 'duration/decode/mean': 0.11546502699900885, 'duration/decode/std': 0.027261033445088345, 'count_calls/correct_edge/mean': 43.15, 'count_calls/correct_edge/std': 6.742959290993829}}, {'no_error': 120, 'benchmark': {'decoded': 120, 'iterations': 120, 'seed': 237937.1750835, 'duration/decode/mean': 0.11479477499936669, 'duration/decode/std': 0.024020320520194538, 'count_calls/correct_edge/mean': 44.733333333333334, 'count_calls/correct_edge/std': 5.767341001035245}}, {'no_error': 140, 'benchmark': {'decoded': 140, 'iterations': 140, 'seed': 238004.8586671, 'duration/decode/mean': 0.10923580214314695, 'duration/decode/std': 0.01787129377301107, 'count_calls/correct_edge/mean': 43.65714285714286, 'count_calls/correct_edge/std': 6.595205432272565}}, {'no_error': 160, 'benchmark': {'decoded': 160, 'iterations': 160, 'seed': 238080.4542824, 'duration/decode/mean': 0.11668858624943823, 'duration/decode/std': 0.021546906870532512, 'count_calls/correct_edge/mean': 44.3625, 'count_calls/correct_edge/std': 6.969117142795062}}, {'no_error': 180, 'benchmark': {'decoded': 180, 'iterations': 180, 'seed': 238169.3507914, 'duration/decode/mean': 0.12156340166661216, 'duration/decode/std': 0.02730440788059051, 'count_calls/correct_edge/mean': 44.35, 'count_calls/correct_edge/std': 6.528973885688317}}, {'no_error': 200, 'benchmark': {'decoded': 200, 'iterations': 200, 'seed': 238272.5632775, 'duration/decode/mean': 0.11870620200046687, 'duration/decode/std': 0.02338878966927848, 'count_calls/correct_edge/mean': 43.38, 'count_calls/correct_edge/std': 6.587533681128318}}], [{'no_error': 20, 'benchmark': {'decoded': 20, 'iterations': 20, 'seed': 238346.7784282, 'duration/decode/mean': 0.34498644500272346, 'duration/decode/std': 0.06750011523348304, 'count_calls/correct_edge/mean': 81.85, 'count_calls/correct_edge/std': 9.133865556269154}}, {'no_error': 40, 'benchmark': {'decoded': 40, 'iterations': 40, 'seed': 238379.0871215, 'duration/decode/mean': 0.32552749249880436, 'duration/decode/std': 0.06165554330642452, 'count_calls/correct_edge/mean': 77.45, 'count_calls/correct_edge/std': 8.452662302493813}}, {'no_error': 60, 'benchmark': {'decoded': 60, 'iterations': 60, 'seed': 238431.6619872, 'duration/decode/mean': 0.3423599483348274, 'duration/decode/std': 0.09689930700065566, 'count_calls/correct_edge/mean': 79.13333333333334, 'count_calls/correct_edge/std': 7.63209596958063}}, {'no_error': 80, 'benchmark': {'decoded': 80, 'iterations': 80, 'seed': 238508.8296544, 'duration/decode/mean': 0.3216198437512503, 'duration/decode/std': 0.0702952464264878, 'count_calls/correct_edge/mean': 79.075, 'count_calls/correct_edge/std': 8.307790019012277}}, {'no_error': 100, 'benchmark': {'decoded': 100, 'iterations': 100, 'seed': 238604.452209, 'duration/decode/mean': 0.3226775339993765, 'duration/decode/std': 0.06602908735512121, 'count_calls/correct_edge/mean': 78.75, 'count_calls/correct_edge/std': 8.617859362974079}}, {'no_error': 120, 'benchmark': {'decoded': 120, 'iterations': 120, 'seed': 238721.2014529, 'duration/decode/mean': 0.32325081583427767, 'duration/decode/std': 0.06817613580685433, 'count_calls/correct_edge/mean': 79.5, 'count_calls/correct_edge/std': 8.947066558375433}}, {'no_error': 140, 'benchmark': {'decoded': 140, 'iterations': 140, 'seed': 238860.039788, 'duration/decode/mean': 0.3263419050006113, 'duration/decode/std': 0.07350701275653518, 'count_calls/correct_edge/mean': 80.51428571428572, 'count_calls/correct_edge/std': 8.668898195178402}}, {'no_error': 160, 'benchmark': {'decoded': 160, 'iterations': 160, 'seed': 239018.879471, 'duration/decode/mean': 0.32832126625089586, 'duration/decode/std': 0.07592098926675035, 'count_calls/correct_edge/mean': 80.09375, 'count_calls/correct_edge/std': 8.59999772892412}}, {'no_error': 180, 'benchmark': {'decoded': 180, 'iterations': 180, 'seed': 239198.0918184, 'duration/decode/mean': 0.3365510405553828, 'duration/decode/std': 0.07938000890261143, 'count_calls/correct_edge/mean': 80.00555555555556, 'count_calls/correct_edge/std': 9.032347561356117}}, {'no_error': 200, 'benchmark': {'decoded': 200, 'iterations': 200, 'seed': 239413.5434919, 'duration/decode/mean': 0.33989385699969715, 'duration/decode/std': 0.07259419050041922, 'count_calls/correct_edge/mean': 79.1, 'count_calls/correct_edge/std': 9.252026804976301}}], [{'no_error': 20, 'benchmark': {'decoded': 20, 'iterations': 20, 'seed': 239573.2315217, 'duration/decode/mean': 0.873316934997274, 'duration/decode/std': 0.13519349938000758, 'count_calls/correct_edge/mean': 131.2, 'count_calls/correct_edge/std': 12.874781551544865}}, {'no_error': 40, 'benchmark': {'decoded': 40, 'iterations': 40, 'seed': 239635.978712, 'duration/decode/mean': 0.939729677500145, 'duration/decode/std': 0.18506863314654676, 'count_calls/correct_edge/mean': 134.375, 'count_calls/correct_edge/std': 12.203867214944614}}, {'no_error': 60, 'benchmark': {'decoded': 60, 'iterations': 60, 'seed': 239744.927723, 'duration/decode/mean': 0.9284547949995613, 'duration/decode/std': 0.1769663338968788, 'count_calls/correct_edge/mean': 131.43333333333334, 'count_calls/correct_edge/std': 12.254749646112272}}, {'no_error': 80, 'benchmark': {'decoded': 80, 'iterations': 80, 'seed': 239899.1080715, 'duration/decode/mean': 0.9296096999984002, 'duration/decode/std': 0.180200673105248, 'count_calls/correct_edge/mean': 130.825, 'count_calls/correct_edge/std': 10.950770520835508}}, {'no_error': 100, 'benchmark': {'decoded': 100, 'iterations': 100, 'seed': 240099.7735342, 'duration/decode/mean': 0.9238835559994913, 'duration/decode/std': 0.15846135649916168, 'count_calls/correct_edge/mean': 130.07, 'count_calls/correct_edge/std': 11.238554177473185}}, {'no_error': 120, 'benchmark': {'decoded': 120, 'iterations': 120, 'seed': 240345.464105, 'duration/decode/mean': 0.9340628533342775, 'duration/decode/std': 0.16661670127773345, 'count_calls/correct_edge/mean': 131.94166666666666, 'count_calls/correct_edge/std': 10.562745723637496}}, {'no_error': 140, 'benchmark': {'decoded': 140, 'iterations': 140, 'seed': 240636.4393157, 'duration/decode/mean': 0.9375371978576628, 'duration/decode/std': 0.15671734691958805, 'count_calls/correct_edge/mean': 132.35, 'count_calls/correct_edge/std': 10.648893302659603}}, {'no_error': 160, 'benchmark': {'decoded': 160, 'iterations': 160, 'seed': 240974.9458568, 'duration/decode/mean': 0.916327921249831, 'duration/decode/std': 0.16537459070111812, 'count_calls/correct_edge/mean': 130.55, 'count_calls/correct_edge/std': 10.791084282869818}}, {'no_error': 180, 'benchmark': {'decoded': 180, 'iterations': 180, 'seed': 241353.7920875, 'duration/decode/mean': 0.9171201222217253, 'duration/decode/std': 0.18399511800087257, 'count_calls/correct_edge/mean': 130.83333333333334, 'count_calls/correct_edge/std': 12.153874553683146}}, {'no_error': 200, 'benchmark': {'decoded': 200, 'iterations': 200, 'seed': 241792.8810302, 'duration/decode/mean': 0.9256833530004951, 'duration/decode/std': 0.17409415284502863, 'count_calls/correct_edge/mean': 129.28, 'count_calls/correct_edge/std': 11.520052083215596}}], [{'no_error': 20, 'benchmark': {'decoded': 20, 'iterations': 20, 'seed': 242127.7879681, 'duration/decode/mean': 2.2358230349986115, 'duration/decode/std': 0.3554482157075378, 'count_calls/correct_edge/mean': 197.65, 'count_calls/correct_edge/std': 14.241751998964173}}, {'no_error': 40, 'benchmark': {'decoded': 40, 'iterations': 40, 'seed': 242247.5012643, 'duration/decode/mean': 2.2265744125004856, 'duration/decode/std': 0.34546580446826647, 'count_calls/correct_edge/mean': 204.825, 'count_calls/correct_edge/std': 15.09948260702995}}, {'no_error': 60, 'benchmark': {'decoded': 60, 'iterations': 60, 'seed': 242455.287983, 'duration/decode/mean': 2.2780626050002564, 'duration/decode/std': 0.38173808431954415, 'count_calls/correct_edge/mean': 205.0, 'count_calls/correct_edge/std': 15.177834716014885}}, {'no_error': 80, 'benchmark': {'decoded': 80, 'iterations': 80, 'seed': 242753.2079631, 'duration/decode/mean': 2.2442665800001125, 'duration/decode/std': 0.3286131959896692, 'count_calls/correct_edge/mean': 202.3875, 'count_calls/correct_edge/std': 14.483174505266447}}, {'no_error': 100, 'benchmark': {'decoded': 100, 'iterations': 100, 'seed': 243136.9839108, 'duration/decode/mean': 2.237558141001791, 'duration/decode/std': 0.3210773024656926, 'count_calls/correct_edge/mean': 202.73, 'count_calls/correct_edge/std': 14.174522919661177}}, {'no_error': 120, 'benchmark': {'decoded': 120, 'iterations': 120, 'seed': 243609.3351354, 'duration/decode/mean': 2.2361416899994464, 'duration/decode/std': 0.34114384335970954, 'count_calls/correct_edge/mean': 199.36666666666667, 'count_calls/correct_edge/std': 13.688762625680313}}, {'no_error': 140, 'benchmark': {'decoded': 140, 'iterations': 140, 'seed': 244168.945896, 'duration/decode/mean': 2.2260265800002212, 'duration/decode/std': 0.3300869108448811, 'count_calls/correct_edge/mean': 202.19285714285715, 'count_calls/correct_edge/std': 12.766527902835506}}, {'no_error': 160, 'benchmark': {'decoded': 160, 'iterations': 160, 'seed': 244814.3992573, 'duration/decode/mean': 2.1851566187497156, 'duration/decode/std': 0.363105786961958, 'count_calls/correct_edge/mean': 201.63125, 'count_calls/correct_edge/std': 13.419026545822913}}, {'no_error': 180, 'benchmark': {'decoded': 180, 'iterations': 180, 'seed': 245523.7085347, 'duration/decode/mean': 2.134015834442754, 'duration/decode/std': 0.3451941945745576, 'count_calls/correct_edge/mean': 201.5888888888889, 'count_calls/correct_edge/std': 13.034905654898266}}, {'no_error': 200, 'benchmark': {'decoded': 200, 'iterations': 200, 'seed': 246310.5869651, 'duration/decode/mean': 2.1637768205007886, 'duration/decode/std': 0.35502203798757415, 'count_calls/correct_edge/mean': 202.535, 'count_calls/correct_edge/std': 15.180209978784879}}], [{'no_error': 20, 'benchmark': {'decoded': 20, 'iterations': 20, 'seed': 246969.8341002, 'duration/decode/mean': 4.6371710000035815, 'duration/decode/std': 0.4680037115683334, 'count_calls/correct_edge/mean': 294.5, 'count_calls/correct_edge/std': 17.290170618012997}}, {'no_error': 40, 'benchmark': {'decoded': 40, 'iterations': 40, 'seed': 247182.9497549, 'duration/decode/mean': 4.627951560000656, 'duration/decode/std': 0.6114110603704243, 'count_calls/correct_edge/mean': 295.8, 'count_calls/correct_edge/std': 17.265862272125304}}, {'no_error': 60, 'benchmark': {'decoded': 60, 'iterations': 60, 'seed': 247557.9057658, 'duration/decode/mean': 4.498397073334005, 'duration/decode/std': 0.6413008803867732, 'count_calls/correct_edge/mean': 292.81666666666666, 'count_calls/correct_edge/std': 16.28137142735696}}, {'no_error': 80, 'benchmark': {'decoded': 80, 'iterations': 80, 'seed': 248086.2156735, 'duration/decode/mean': 4.55209255750051, 'duration/decode/std': 0.6039680607264781, 'count_calls/correct_edge/mean': 293.775, 'count_calls/correct_edge/std': 16.503768509040594}}, {'no_error': 100, 'benchmark': {'decoded': 100, 'iterations': 100, 'seed': 248777.591759, 'duration/decode/mean': 4.553864934000303, 'duration/decode/std': 0.6272690688482956, 'count_calls/correct_edge/mean': 291.96, 'count_calls/correct_edge/std': 15.48607116088519}}, {'no_error': 120, 'benchmark': {'decoded': 120, 'iterations': 120, 'seed': 249631.6706261, 'duration/decode/mean': 4.656172245833781, 'duration/decode/std': 0.6364843207996258, 'count_calls/correct_edge/mean': 293.26666666666665, 'count_calls/correct_edge/std': 18.744658498415546}}, {'no_error': 140, 'benchmark': {'decoded': 140, 'iterations': 140, 'seed': 250659.5485558, 'duration/decode/mean': 4.614988684284617, 'duration/decode/std': 0.5669053489537255, 'count_calls/correct_edge/mean': 296.77142857142854, 'count_calls/correct_edge/std': 16.844389849078983}}, {'no_error': 160, 'benchmark': {'decoded': 160, 'iterations': 160, 'seed': 251840.1344213, 'duration/decode/mean': 4.685562140625552, 'duration/decode/std': 0.6974177077252074, 'count_calls/correct_edge/mean': 293.50625, 'count_calls/correct_edge/std': 15.828770038682729}}, {'no_error': 180, 'benchmark': {'decoded': 180, 'iterations': 180, 'seed': 253201.2573093, 'duration/decode/mean': 4.607202032222833, 'duration/decode/std': 0.6230368368786293, 'count_calls/correct_edge/mean': 292.7888888888889, 'count_calls/correct_edge/std': 15.311937568406078}}, {'no_error': 200, 'benchmark': {'decoded': 200, 'iterations': 200, 'seed': 254701.891995, 'duration/decode/mean': 4.567952724000497, 'duration/decode/std': 0.604227579400658, 'count_calls/correct_edge/mean': 295.68, 'count_calls/correct_edge/std': 17.80948062128708}}]]

parallel = [[{'no_error': 20, 'benchmark': {'decoded': 20, 'iterations': 20, 'seed': 237463.1034629, 'duration/decode/mean': 0.011857495001459028, 'duration/decode/std': 0.0012368114730758532, 'count_calls/correct_edge/mean': 19.95, 'count_calls/correct_edge/std': 3.761316258971054}}, {'no_error': 40, 'benchmark': {'decoded': 40, 'iterations': 40, 'seed': 237467.6924274, 'duration/decode/mean': 0.012360107501444872, 'duration/decode/std': 0.0019592335176193995, 'count_calls/correct_edge/mean': 21.05, 'count_calls/correct_edge/std': 5.458708638496838}}, {'no_error': 60, 'benchmark': {'decoded': 60, 'iterations': 60, 'seed': 237477.0515158, 'duration/decode/mean': 0.012249853331498646, 'duration/decode/std': 0.0014720329694231555, 'count_calls/correct_edge/mean': 19.866666666666667, 'count_calls/correct_edge/std': 4.161196409153929}}, {'no_error': 80, 'benchmark': {'decoded': 80, 'iterations': 80, 'seed': 237490.6049927, 'duration/decode/mean': 0.01241284375100804, 'duration/decode/std': 0.0017181166584449633, 'count_calls/correct_edge/mean': 20.9375, 'count_calls/correct_edge/std': 5.136496252310518}}, {'no_error': 98, 'benchmark': {'decoded': 99, 'iterations': 100, 'seed': 237508.6000633, 'duration/decode/mean': 0.012410910000326111, 'duration/decode/std': 0.0016753149814950248, 'count_calls/correct_edge/mean': 20.8, 'count_calls/correct_edge/std': 4.804164859785725}}, {'no_error': 120, 'benchmark': {'decoded': 120, 'iterations': 120, 'seed': 237531.5741053, 'duration/decode/mean': 0.012601798334314178, 'duration/decode/std': 0.0016003089327203254, 'count_calls/correct_edge/mean': 20.741666666666667, 'count_calls/correct_edge/std': 4.760069735716998}}, {'no_error': 140, 'benchmark': {'decoded': 140, 'iterations': 140, 'seed': 237558.6654226, 'duration/decode/mean': 0.012192495714823183, 'duration/decode/std': 0.0016571735141327488, 'count_calls/correct_edge/mean': 20.335714285714285, 'count_calls/correct_edge/std': 4.099844447024291}}, {'no_error': 160, 'benchmark': {'decoded': 160, 'iterations': 160, 'seed': 237590.7772187, 'duration/decode/mean': 0.012391684374233592, 'duration/decode/std': 0.0018477975369264237, 'count_calls/correct_edge/mean': 21.05625, 'count_calls/correct_edge/std': 4.69207693218046}}, {'no_error': 180, 'benchmark': {'decoded': 180, 'iterations': 180, 'seed': 237627.0046689, 'duration/decode/mean': 0.012409848889445938, 'duration/decode/std': 0.0016214159505017396, 'count_calls/correct_edge/mean': 21.033333333333335, 'count_calls/correct_edge/std': 4.280316706870078}}, {'no_error': 200, 'benchmark': {'decoded': 199, 'iterations': 200, 'seed': 237669.2932956, 'duration/decode/mean': 0.012832424499792978, 'duration/decode/std': 0.0018783692479032816, 'count_calls/correct_edge/mean': 20.345, 'count_calls/correct_edge/std': 4.40635620439383}}], [{'no_error': 20, 'benchmark': {'decoded': 20, 'iterations': 20, 'seed': 237720.0460856, 'duration/decode/mean': 0.03224723000021186, 'duration/decode/std': 0.009091897597387873, 'count_calls/correct_edge/mean': 45.2, 'count_calls/correct_edge/std': 4.545327270945405}}, {'no_error': 40, 'benchmark': {'decoded': 40, 'iterations': 40, 'seed': 237734.3814157, 'duration/decode/mean': 0.03106883500222466, 'duration/decode/std': 0.007391638321424054, 'count_calls/correct_edge/mean': 46.125, 'count_calls/correct_edge/std': 7.359305334065166}}, {'no_error': 60, 'benchmark': {'decoded': 59, 'iterations': 60, 'seed': 237759.8879149, 'duration/decode/mean': 0.025363574999695025, 'duration/decode/std': 0.004662726190826548, 'count_calls/correct_edge/mean': 43.68333333333333, 'count_calls/correct_edge/std': 6.631972222164049}}, {'no_error': 80, 'benchmark': {'decoded': 79, 'iterations': 80, 'seed': 237792.6632558, 'duration/decode/mean': 0.037053733751235995, 'duration/decode/std': 0.015827133993109265, 'count_calls/correct_edge/mean': 43.4625, 'count_calls/correct_edge/std': 6.8846636628088085}}, {'no_error': 100, 'benchmark': {'decoded': 99, 'iterations': 100, 'seed': 237855.3017755, 'duration/decode/mean': 0.02714059300109511, 'duration/decode/std': 0.006590533542842112, 'count_calls/correct_edge/mean': 44.3, 'count_calls/correct_edge/std': 6.902897942168926}}, {'no_error': 120, 'benchmark': {'decoded': 120, 'iterations': 120, 'seed': 237910.5026224, 'duration/decode/mean': 0.025448788333839425, 'duration/decode/std': 0.004626571401029396, 'count_calls/correct_edge/mean': 43.24166666666667, 'count_calls/correct_edge/std': 6.302110537131369}}, {'no_error': 140, 'benchmark': {'decoded': 140, 'iterations': 140, 'seed': 237974.1486648, 'duration/decode/mean': 0.02482289142811039, 'duration/decode/std': 0.004602950123692022, 'count_calls/correct_edge/mean': 43.392857142857146, 'count_calls/correct_edge/std': 6.863668769658967}}, {'no_error': 160, 'benchmark': {'decoded': 160, 'iterations': 160, 'seed': 238046.3044554, 'duration/decode/mean': 0.024260646873153748, 'duration/decode/std': 0.005060513484052741, 'count_calls/correct_edge/mean': 43.4625, 'count_calls/correct_edge/std': 7.009714241679186}}, {'no_error': 180, 'benchmark': {'decoded': 179, 'iterations': 180, 'seed': 238130.1717041, 'duration/decode/mean': 0.025358012221436688, 'duration/decode/std': 0.004227838208000967, 'count_calls/correct_edge/mean': 44.15555555555556, 'count_calls/correct_edge/std': 6.4985658436836164}}, {'no_error': 200, 'benchmark': {'decoded': 200, 'iterations': 200, 'seed': 238227.9731576, 'duration/decode/mean': 0.026599251001171068, 'duration/decode/std': 0.011907365661381624, 'count_calls/correct_edge/mean': 43.65, 'count_calls/correct_edge/std': 6.77624527301071}}], [{'no_error': 20, 'benchmark': {'decoded': 20, 'iterations': 20, 'seed': 238336.0884691, 'duration/decode/mean': 0.07133109999558655, 'duration/decode/std': 0.024838371569307963, 'count_calls/correct_edge/mean': 80.1, 'count_calls/correct_edge/std': 6.729784543356496}}, {'no_error': 40, 'benchmark': {'decoded': 40, 'iterations': 40, 'seed': 238361.5968051, 'duration/decode/mean': 0.0590960224995797, 'duration/decode/std': 0.010847130314240314, 'count_calls/correct_edge/mean': 80.375, 'count_calls/correct_edge/std': 9.427851027673274}}, {'no_error': 60, 'benchmark': {'decoded': 59, 'iterations': 60, 'seed': 238407.1047985, 'duration/decode/mean': 0.05712634666560916, 'duration/decode/std': 0.009839695345752363, 'count_calls/correct_edge/mean': 79.7, 'count_calls/correct_edge/std': 7.746612162745725}}, {'no_error': 80, 'benchmark': {'decoded': 79, 'iterations': 80, 'seed': 238475.1470038, 'duration/decode/mean': 0.059103676249651474, 'duration/decode/std': 0.014786796976673589, 'count_calls/correct_edge/mean': 79.4125, 'count_calls/correct_edge/std': 8.528912225483388}}, {'no_error': 100, 'benchmark': {'decoded': 97, 'iterations': 100, 'seed': 238563.3773071, 'duration/decode/mean': 0.05871468700002879, 'duration/decode/std': 0.011300101152749273, 'count_calls/correct_edge/mean': 79.78, 'count_calls/correct_edge/std': 10.333034404278347}}, {'no_error': 120, 'benchmark': {'decoded': 119, 'iterations': 120, 'seed': 238671.9996425, 'duration/decode/mean': 0.059075722500710984, 'duration/decode/std': 0.025446966581697174, 'count_calls/correct_edge/mean': 80.13333333333334, 'count_calls/correct_edge/std': 8.793874130451393}}, {'no_error': 138, 'benchmark': {'decoded': 138, 'iterations': 140, 'seed': 238802.4776463, 'duration/decode/mean': 0.05892987071399278, 'duration/decode/std': 0.012894424022298345, 'count_calls/correct_edge/mean': 79.04285714285714, 'count_calls/correct_edge/std': 8.78136942507214}}, {'no_error': 160, 'benchmark': {'decoded': 159, 'iterations': 160, 'seed': 238954.6042954, 'duration/decode/mean': 0.05708027375094389, 'duration/decode/std': 0.01112768376680044, 'count_calls/correct_edge/mean': 79.24375, 'count_calls/correct_edge/std': 8.200416814863743}}, {'no_error': 180, 'benchmark': {'decoded': 175, 'iterations': 180, 'seed': 239127.3618021, 'duration/decode/mean': 0.05323935611005355, 'duration/decode/std': 0.010162065207241753, 'count_calls/correct_edge/mean': 78.68333333333334, 'count_calls/correct_edge/std': 9.190257522688542}}, {'no_error': 198, 'benchmark': {'decoded': 197, 'iterations': 200, 'seed': 239324.0737875, 'duration/decode/mean': 0.07237619449922932, 'duration/decode/std': 0.02622507503256798, 'count_calls/correct_edge/mean': 79.97, 'count_calls/correct_edge/std': 9.085653526301781}}], [{'no_error': 20, 'benchmark': {'decoded': 20, 'iterations': 20, 'seed': 239556.5730237, 'duration/decode/mean': 0.14984199500031536, 'duration/decode/std': 0.02063659945284618, 'count_calls/correct_edge/mean': 130.65, 'count_calls/correct_edge/std': 11.49032201463475}}, {'no_error': 40, 'benchmark': {'decoded': 40, 'iterations': 40, 'seed': 239604.3851801, 'duration/decode/mean': 0.15922672499509644, 'duration/decode/std': 0.03942753713344332, 'count_calls/correct_edge/mean': 131.35, 'count_calls/correct_edge/std': 9.637297338984618}}, {'no_error': 58, 'benchmark': {'decoded': 59, 'iterations': 60, 'seed': 239698.9977057, 'duration/decode/mean': 0.15494006000080845, 'duration/decode/std': 0.028408287482315876, 'count_calls/correct_edge/mean': 131.51666666666668, 'count_calls/correct_edge/std': 11.651168277139517}}, {'no_error': 80, 'benchmark': {'decoded': 78, 'iterations': 80, 'seed': 239838.002339, 'duration/decode/mean': 0.15654952624827273, 'duration/decode/std': 0.027313230856461295, 'count_calls/correct_edge/mean': 131.5125, 'count_calls/correct_edge/std': 11.586623483569317}}, {'no_error': 100, 'benchmark': {'decoded': 96, 'iterations': 100, 'seed': 240022.3914324, 'duration/decode/mean': 0.16539708599855657, 'duration/decode/std': 0.05231893571057384, 'count_calls/correct_edge/mean': 132.55, 'count_calls/correct_edge/std': 11.424863237693483}}, {'no_error': 120, 'benchmark': {'decoded': 117, 'iterations': 120, 'seed': 240252.945149, 'duration/decode/mean': 0.161225531664968, 'duration/decode/std': 0.0487686959379448, 'count_calls/correct_edge/mean': 130.4, 'count_calls/correct_edge/std': 11.828214855449097}}, {'no_error': 140, 'benchmark': {'decoded': 138, 'iterations': 140, 'seed': 240530.2458305, 'duration/decode/mean': 0.15953009857143374, 'duration/decode/std': 0.044798425125502965, 'count_calls/correct_edge/mean': 131.2642857142857, 'count_calls/correct_edge/std': 11.944281784474894}}, {'no_error': 158, 'benchmark': {'decoded': 157, 'iterations': 160, 'seed': 240852.7887652, 'duration/decode/mean': 0.16159635687436094, 'duration/decode/std': 0.043989791241138894, 'count_calls/correct_edge/mean': 132.43125, 'count_calls/correct_edge/std': 11.10158877987741}}, {'no_error': 178, 'benchmark': {'decoded': 173, 'iterations': 180, 'seed': 241217.1126562, 'duration/decode/mean': 0.1589216916660209, 'duration/decode/std': 0.030972523121154685, 'count_calls/correct_edge/mean': 132.60555555555555, 'count_calls/correct_edge/std': 12.283456454481199}}, {'no_error': 196, 'benchmark': {'decoded': 192, 'iterations': 200, 'seed': 241626.5786024, 'duration/decode/mean': 0.1745235450002656, 'duration/decode/std': 0.050857587884148556, 'count_calls/correct_edge/mean': 131.65, 'count_calls/correct_edge/std': 11.510321455111495}}], [{'no_error': 20, 'benchmark': {'decoded': 20, 'iterations': 20, 'seed': 242099.3768439, 'duration/decode/mean': 0.3413614799937932, 'duration/decode/std': 0.07726965882884876, 'count_calls/correct_edge/mean': 196.05, 'count_calls/correct_edge/std': 14.080039062445815}}, {'no_error': 40, 'benchmark': {'decoded': 40, 'iterations': 40, 'seed': 242193.6763047, 'duration/decode/mean': 0.3780239699975937, 'duration/decode/std': 0.06449413821251612, 'count_calls/correct_edge/mean': 204.0, 'count_calls/correct_edge/std': 13.861818062577505}}, {'no_error': 56, 'benchmark': {'decoded': 54, 'iterations': 60, 'seed': 242375.0056569, 'duration/decode/mean': 0.39238543500202167, 'duration/decode/std': 0.11409065386835017, 'count_calls/correct_edge/mean': 204.41666666666666, 'count_calls/correct_edge/std': 13.607953148394099}}, {'no_error': 80, 'benchmark': {'decoded': 75, 'iterations': 80, 'seed': 242648.9313627, 'duration/decode/mean': 0.37233632374809533, 'duration/decode/std': 0.08795394229306111, 'count_calls/correct_edge/mean': 202.0125, 'count_calls/correct_edge/std': 13.913566895300429}}, {'no_error': 98, 'benchmark': {'decoded': 97, 'iterations': 100, 'seed': 243006.7707384, 'duration/decode/mean': 0.37055903700063936, 'duration/decode/std': 0.09304350233726845, 'count_calls/correct_edge/mean': 202.09, 'count_calls/correct_edge/std': 13.768148023608694}}, {'no_error': 118, 'benchmark': {'decoded': 113, 'iterations': 120, 'seed': 243453.8143319, 'duration/decode/mean': 0.37813421749936726, 'duration/decode/std': 0.11826499940959788, 'count_calls/correct_edge/mean': 203.89166666666668, 'count_calls/correct_edge/std': 15.711882888084702}}, {'no_error': 133, 'benchmark': {'decoded': 132, 'iterations': 140, 'seed': 243986.7054801, 'duration/decode/mean': 0.37822469857222, 'duration/decode/std': 0.09957130081696823, 'count_calls/correct_edge/mean': 204.25, 'count_calls/correct_edge/std': 13.424457956601877}}, {'no_error': 158, 'benchmark': {'decoded': 152, 'iterations': 160, 'seed': 244608.4795915, 'duration/decode/mean': 0.37583680000061576, 'duration/decode/std': 0.12855824614988098, 'count_calls/correct_edge/mean': 202.49375, 'count_calls/correct_edge/std': 13.46801622131114}}, {'no_error': 176, 'benchmark': {'decoded': 166, 'iterations': 180, 'seed': 245307.4423649, 'duration/decode/mean': 0.3276150449997254, 'duration/decode/std': 0.09068566422086274, 'count_calls/correct_edge/mean': 202.45, 'count_calls/correct_edge/std': 14.256100058883177}}, {'no_error': 192, 'benchmark': {'decoded': 186, 'iterations': 200, 'seed': 246066.6207675, 'duration/decode/mean': 0.33893079249799485, 'duration/decode/std': 0.11539566365832694, 'count_calls/correct_edge/mean': 202.28, 'count_calls/correct_edge/std': 13.440669626175625}}], [{'no_error': 18, 'benchmark': {'decoded': 18, 'iterations': 20, 'seed': 246921.5740061, 'duration/decode/mean': 0.8838088799981051, 'duration/decode/std': 0.2246974989730323, 'count_calls/correct_edge/mean': 294.65, 'count_calls/correct_edge/std': 15.463747928623254}}, {'no_error': 40, 'benchmark': {'decoded': 38, 'iterations': 40, 'seed': 247093.1065088, 'duration/decode/mean': 0.8447580600004585, 'duration/decode/std': 0.23284465497380558, 'count_calls/correct_edge/mean': 293.925, 'count_calls/correct_edge/std': 16.86918418300067}}, {'no_error': 54, 'benchmark': {'decoded': 53, 'iterations': 60, 'seed': 247424.0531971, 'duration/decode/mean': 0.8643835033357997, 'duration/decode/std': 0.26723624272917945, 'count_calls/correct_edge/mean': 292.8333333333333, 'count_calls/correct_edge/std': 17.915697648214046}}, {'no_error': 78, 'benchmark': {'decoded': 70, 'iterations': 80, 'seed': 247908.9724672, 'duration/decode/mean': 0.8573585800008005, 'duration/decode/std': 0.2801375486591072, 'count_calls/correct_edge/mean': 292.75, 'count_calls/correct_edge/std': 17.73027072551347}}, {'no_error': 96, 'benchmark': {'decoded': 93, 'iterations': 100, 'seed': 248557.610278, 'duration/decode/mean': 0.861595185000624, 'duration/decode/std': 0.2808064837960766, 'count_calls/correct_edge/mean': 298.11, 'count_calls/correct_edge/std': 15.946093565510015}}, {'no_error': 110, 'benchmark': {'decoded': 106, 'iterations': 120, 'seed': 249369.0270054, 'duration/decode/mean': 0.8565489199980221, 'duration/decode/std': 0.25204271504279235, 'count_calls/correct_edge/mean': 296.7083333333333, 'count_calls/correct_edge/std': 19.509568521340725}}, {'no_error': 128, 'benchmark': {'decoded': 119, 'iterations': 140, 'seed': 250351.1676159, 'duration/decode/mean': 0.8675704042855484, 'duration/decode/std': 0.2311879892743093, 'count_calls/correct_edge/mean': 294.9, 'count_calls/correct_edge/std': 16.804463692721647}}, {'no_error': 156, 'benchmark': {'decoded': 145, 'iterations': 160, 'seed': 251492.5301099, 'duration/decode/mean': 0.8454221206262446, 'duration/decode/std': 0.22916232679478654, 'count_calls/correct_edge/mean': 294.675, 'count_calls/correct_edge/std': 17.604527116625427}}, {'no_error': 172, 'benchmark': {'decoded': 160, 'iterations': 180, 'seed': 252804.8296267, 'duration/decode/mean': 0.8676554699995904, 'duration/decode/std': 0.27537302428209176, 'count_calls/correct_edge/mean': 295.03333333333336, 'count_calls/correct_edge/std': 18.222361110581815}}, {'no_error': 190, 'benchmark': {'decoded': 173, 'iterations': 200, 'seed': 254267.4979318, 'duration/decode/mean': 0.8521638065003208, 'duration/decode/std': 0.2391534105608894, 'count_calls/correct_edge/mean': 293.665, 'count_calls/correct_edge/std': 16.340831527189795}}]]

normal_decoding_time = [[dictionary['benchmark']['duration/decode/mean'] for dictionary in sublist] for sublist in normal]
normal_logical_error = [[(dictionary['no_error'])/dictionary['benchmark']['iterations'] for dictionary in sublist] for sublist in normal]
parallel_decoding_time = [[dictionary['benchmark']['duration/decode/mean'] for dictionary in sublist] for sublist in parallel]
parallel_logical_error = [[(dictionary['no_error'])/dictionary['benchmark']['iterations']for dictionary in sublist] for sublist in parallel]

for i in range(6):
   print(np.average(normal_decoding_time[i])/np.average(parallel_decoding_time[i]))
list_of_iterations = [20,40,60,80,100,120,140,160,180,200]


cmap = plt.get_cmap('tab10')

for i in range(6):
   colour = cmap(i)  # Get a unique color from the color map for each line
   plt.plot(list_of_iterations, normal_logical_error[i], 'o-', color = colour, label='Normal ' + str(7 + 2*i) + 'x' + str(7 + 2*i))
   plt.plot(list_of_iterations, parallel_logical_error[i], 'o--', color = colour, label='Parallel ' + str(7 + 2*i) + 'x' + str(7 + 2*i))

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

# for d in [3, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16, 18]:

   
#    code, decoder = initialize((d,d), "toric", "lazy_mwpm", plotting=False, enabled_errors=["pauli"], faulty_measurements=False, initial_states = (0,0))
#    lazy = run(code, decoder, iterations=chosen_iterations, decode_initial=False, error_rates = {"p_bitflip": 0.0, "p_phaseflip": 0.0001, "p_bitflip_plaq": 0.0, "p_bitflip_star": 0.0}, benchmark=benchmarker)
#    lazy_time.append(lazy['benchmark']['duration/decode/mean'])
#    lazy_success.append(float(lazy['no_error']/ chosen_iterations))
#    code, decoder = initialize((d,d), "toric", "mwpm", plotting=False, enabled_errors=["pauli"], faulty_measurements=False, initial_states = (0,0))
#    mwpm = run(code, decoder, iterations=chosen_iterations, decode_initial=False, error_rates = {"p_bitflip": 0.0, "p_phaseflip": 0.0001, "p_bitflip_plaq": 0.0, "p_bitflip_star": 0.0}, benchmark=benchmarker)
#    mwpm_time.append(mwpm['benchmark']['duration/decode/mean'])
#    mwpm_success.append( float(mwpm['no_error'] / chosen_iterations))
#    code, decoder = initialize((d,d), "toric", "unionfind", plotting=False, enabled_errors=["pauli"], faulty_measurements=False, initial_states = (0,0))
#    uf = run(code, decoder, iterations=chosen_iterations, decode_initial=False, error_rates = {"p_bitflip": 0.0, "p_phaseflip": 0.0001, "p_bitflip_plaq": 0.0, "p_bitflip_star": 0.0}, benchmark=benchmarker)
#    uf_time.append(uf['benchmark']['duration/decode/mean'])
#    uf_success.append( float(uf['no_error'] / chosen_iterations))

#    speedup.append(mwpm['benchmark']['duration/decode/mean']/lazy['benchmark']['duration/decode/mean'])
#    physical_qubits.append(4*d*d)
#    print(d)

#    # figure here
# print(speedup)
# print(lazy_success, mwpm_success)

# plt.plot(physical_qubits, lazy_time, 'bo--', label='Lazier + MWPM')
# plt.plot(physical_qubits, mwpm_time, 'bo-', label='None + MWPM')
# plt.plot(physical_qubits, uf_time, 'ro-', label='None + UF')
# plt.xlabel('Number of physical qubits')
# plt.ylabel('Execution time for pZ = 0.001')
# plt.title('Lazier Decoder as decoder accelerator with perfect measurements')
# plt.yscale('log')
# plt.legend()
# plt.show()

# LAZY SPEEDUP WITH CIRCUIT-LEVEL NOISE, ADJUST TORIC TO RUN THIS
# Perfect measurements, 3D Toric, p = 10^-3 for both data and ancilla qubits, and code (1225, 1024, 841, 729, 576, 441, 361, 225, 144, 100, 49, 25) -> (35, 32, 29, 27, 24, 21, 19, 15, 12, 10, 7,5)

# lazy_time = []
# lazy_success = []

# mwpm_time = []
# mwpm_success = []

# speedup = []

# physical_qubits = []
# chosen_iterations = 100

# for d in [3, 4, 5, 7, 9]:

#    code, decoder = initialize((d,d), "weight_0_toric", "lazy_mwpm", plotting=False, superoperator_enable=True, sup_op_file="./running/phenomenological_wt_0_toric_rates_px_0.03_pz_0.03_pmx_0.03_pmz_0.03.csv", initial_states=(0,0))
#    lazy = run(code, decoder, iterations=chosen_iterations, decode_initial=False, benchmark = benchmarker)
#    lazy_time.append(lazy['benchmark']['duration/decode/mean'])
#    lazy_success.append(float(lazy['no_error']/ chosen_iterations))
#    code, decoder = initialize((d,d), "weight_0_toric", "mwpm", plotting=False, superoperator_enable=True, sup_op_file="./running/phenomenological_wt_0_toric_rates_px_0.03_pz_0.03_pmx_0.03_pmz_0.03.csv", initial_states=(0,0))
#    mwpm = run(code, decoder, iterations=chosen_iterations, decode_initial=False, benchmark = benchmarker)
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