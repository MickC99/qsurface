from typing import List, Tuple
from qsurface.codes.elements import AncillaQubit
from .._template import Sim
import networkx as nx
from numpy.ctypeslib import ndpointer
import ctypes
import os

LA = List[AncillaQubit]
LAZ = LA

class Toric(Sim):

    name = "Lazy + Minimum-Weight Perfect Matching"
    short = "lazy_mwpm"

    compatibility_measurements = dict(
        PerfectMeasurements=True,
        FaultyMeasurements=False,
    )
    compatibility_errors = dict(
        pauli=True,
        erasure=True,
    )
    
    def decode(self, **kwargs):
        # Inherited docstring
        plaqs, stars = self.get_syndrome()
        self.lazy_checking(plaqs, plaqs, **kwargs)
        self.lazy_checking(stars, stars, **kwargs)

    def lazy_checking(self, syndromes: LA, uncorrected: LAZ, **kwargs):

        failure = False
        distance = self.code.size[0]
        time_layer = self.code.layers
        error_list = []

        for a, ancilla in enumerate(syndromes):
            if ancilla.syndrome:
                if a % distance != 1 and syndromes[a+1].syndrome:
                    shared_data_qubit_index = next(i for i, dq1 in syndromes[a].parity_qubits for dq2 in syndromes[a+1].parity_qubits if dq1 == dq2)
                    error_list.append([ancilla, syndromes[a].parity_qubits[shared_data_qubit_index]])
                    syndromes[a].syndrome = False
                    syndromes[a+1].syndrome = False
                elif a % distance*(distance-1) not in (1, 2, 3, 4) and syndromes[a+distance].syndrome:
                    shared_data_qubit_index = next(i for i, dq1 in syndromes[a].parity_qubits for dq2 in syndromes[a+distance].parity_qubits if dq1 == dq2)
                    error_list.append([ancilla, syndromes[a].parity_qubits[shared_data_qubit_index]])
                    syndromes[a].syndrome = False
                    syndromes[a+distance].syndrome = False
                elif time_layer == 1 and a <= distance*distance*(distance-1) and syndromes[a+distance**2].syndrome:
                    shared_data_qubit_index = next(i for i, dq1 in syndromes[a].parity_qubits for dq2 in syndromes[a+distance**2].parity_qubits if dq1 == dq2)
                    error_list.append([ancilla, syndromes[a].parity_qubits[shared_data_qubit_index]])
                    syndromes[a].syndrome = False
                    syndromes[a+distance**2].syndrome = False
                else:
                    syndromes = LAZ
                    error_list = []
                    failure = True
                    break
        
        if failure == True:
            return "Failure"
    
        else:
            for correction in error_list:
                self.correct_edge(correction[0], correction[1])

        # Checked per row per time layer
        # Do not check backwards, not useful

        # Idea: if x v y v t = 0 or x v y v t = d-1, then check three places, else check four places
        # Example:
        # for LA[0] = (0,0,0), check LA[1] = (1,0,0), check LA[d] = (0,1,0) and check LA[d*d] = (0,0,1)
        
        # # Initial code without optimisation and reset
        # for a in len(syndromes):
        #     if syndromes[a] == True:
        #         if a % distance != 1 and :
        #             if syndromes[a+1] == True:
        #                 error_list.append((a,a+1))
        #                 syndromes[a] == False
        #                 syndromes[a+1] == False
        #                 continue
        #         if a % distance*(distance-1) not in (1,2,3,4):
        #             if syndromes[a+distance] == True:
        #                 error_list.append((a,a+distance))
        #                 syndromes[a] == False
        #                 syndromes[a+distance] == False
        #                 continue
        #         if a <= distance*distance*(distance-1):
        #              if syndromes[a+distance**2] == True:
        #                 error_list.append((a,a+distance**2))
        #                 syndromes[a] == False
        #                 syndromes[a+distance**2] == False
        #                 continue
        #         else:
        #             break
        

        # for a in len(syndromes):
        #   if syndrome[a] == True
        #       check if surroundings are true
        #       if surrounding one is true:
        #           add (syndrome[a],syndrome[surround]) to error list
        #           syndrome[a] == False
        #           syndrome[surround] == False
        #       else reset syndrome and return failure
        # cases:
        # if x = d-1: don't check L[a+1] -> if a is of the form n*d -1 with n in {1,...,d}
        # if y = d-1: don't check L[a+d] -> if a is of the form n*(d-1)*(d-1) + m with n,m in {1,....,d}
        # if t = d-1: don't check L[a+d^2] -> if a > d*d*(d-1) - 1
        # if x = y = t = d-1, immediately return failure if syndrome[a] == True