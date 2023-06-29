from typing import List, Tuple
from qsurface.codes.elements import AncillaQubit, Edge
from .._template import Sim
import networkx as nx
from numpy.ctypeslib import ndpointer
import ctypes
import os
import itertools


LA = List[AncillaQubit]
LE = List[Edge]


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
        
        # Make copies of syndrome lists
        plaqs, stars = self.get_syndrome()
        plaqs_copy = plaqs.copy()
        stars_copy = stars.copy()

        # ADD1: THIS PART SHOULD BE RUN WHEN WEIGHT TESTING IS RUN FOR THE LAZY DECODER.
        # plaqs_weight = self._weight_correction(plaqs, self.match_syndromes(plaqs, **kwargs))
        
        # Create list of edges
        plaqs_edges = [self.code.data_qubits[i][(x, y)].edges['x'].nodes for i in self.code.data_qubits for (x, y) in self.code.data_qubits[i]]
        stars_edges = [self.code.data_qubits[i][(x, y)].edges['z'].nodes for i in self.code.data_qubits for (x, y) in self.code.data_qubits[i]]

        # Add vertical edges in case of faulty measurements, such that each layer is checked first and then its vertical edges
        if type(self.code).__name__ == "FaultyMeasurements":
            n = len(self.code.data_qubits)
            plaqs_edges = sum([[self.code.data_qubits[i][(x, y)].edges['x'].nodes for (x, y) in self.code.data_qubits[i]] + self.code.time_edges[i] for i in range(n-1)], []) + [self.code.data_qubits[n-1][(x, y)].edges['x'].nodes for (x, y) in self.code.data_qubits[n-1]]
            stars_edges = sum([[self.code.data_qubits[i][(x, y)].edges['z'].nodes for (x, y) in self.code.data_qubits[i]] + self.code.time_edges[i] for i in range(n-1)], []) + [self.code.data_qubits[n-1][(x, y)].edges['z'].nodes for (x, y) in self.code.data_qubits[n-1]]

        # Performs lazy decoder first, mwpm when lazy fails FIX FIRST LAYER THEN TIME ETC.
        plaqs_lazy = self.lazy_checking(plaqs, plaqs_edges, **kwargs)
        stars_lazy = self.lazy_checking(stars, stars_edges, **kwargs)
        
        # Reset syndrome list in case of failure and run mwpm
        if plaqs_lazy == 'Failure':
            plaqs = plaqs_copy
            self.correct_matching(plaqs, self.match_syndromes(plaqs, **kwargs))
        
        if stars_lazy == 'Failure':
            stars = stars_copy
            self.correct_matching(stars, self.match_syndromes(stars, **kwargs))

        
        # ADD2: THIS PART SHOULD BE RUN WHEN WEIGHT TESTING IS RUN FOR THE LAZY DECODER.
        # if plaqs_lazy != "Failure":
        #     weight = float(len(plaqs_copy) / plaqs_weight) == 2 if plaqs_weight != 0 else True
        #     if not weight:
        #         print("Lazy decoder solved higher than weight 1")
        #         print(plaqs_copy, plaqs_weight)
        

    def lazy_checking(self, syndromes: LA, edges, **kwargs):
        error_list = []

        if len(syndromes) == 0:
            return
        
        # Iterate over all edges
        for edge in edges:
            ancilla1, ancilla2 = edge

            # Check if both ancilla qubits are in the syndrome set
            if ancilla1 in syndromes and ancilla2 in syndromes:
                error_list.append(edge)
                syndromes.remove(ancilla1)
                syndromes.remove(ancilla2)
        
        # Return failure if there are syndromes left
        if syndromes:
            return "Failure"

        # Correct all data qubit errors found
        for pair in error_list:
            top_layer_ancilla, key = self._lazy_walk_direction(pair[0].loc, pair[1].loc, self.code.size)
            if key != "Time":
                self.correct_edge(top_layer_ancilla, key)
        

    def match_syndromes(self, syndromes: LA, use_blossomv: bool = False, **kwargs) -> list:
        """Decodes a list of syndromes of the same type.

        A graph is constructed with the syndromes in ``syndromes`` as nodes and the distances between each of the syndromes as the edges. The distances are dependent on the boundary conditions of the code and is calculated by `get_qubit_distances`. A minimum-weight matching is then found by either `match_networkx` or `match_blossomv`.

        Parameters
        ----------
        syndromes
            Syndromes of the code.
        use_blossomv
            Use external C++ Blossom V library for minimum-weight matching. Needs to be downloaded and compiled by calling `.get_blossomv`.

        Returns
        -------
        list of `~.codes.elements.AncillaQubit`
            Minimum-weight matched ancilla-qubits.

        """
        matching_graph = self.match_blossomv if use_blossomv else self.match_networkx
        edges = self.get_qubit_distances(syndromes, self.code.size)
        matching = matching_graph(
            edges,
            maxcardinality=self.config["max_cardinality"],
            num_nodes=len(syndromes),
            **kwargs,
        )
        return matching

    def correct_matching(self, syndromes: LA, matching: list, **kwargs):
        """Applies the matchings as a correction to the code."""
        weight = 0
        for i0, i1 in matching:
            weight += self._correct_matched_qubits(syndromes[i0], syndromes[i1])
        return weight

    @staticmethod
    def match_networkx(edges: list, maxcardinality: float, **kwargs) -> list:
        """Finds the minimum-weight matching of a list of ``edges`` using `networkx.algorithms.matching.max_weight_matching`.

        Parameters
        ----------
        edges :  [[nodeA, nodeB, distance(nodeA,nodeB)],...]
            A graph defined by a list of edges.
        maxcardinality
            See `networkx.algorithms.matching.max_weight_matching`.

        Returns
        -------
        list
            Minimum weight matching in the form of [[nodeA, nodeB],..].
        """
        nxgraph = nx.Graph()
        for i0, i1, weight in edges:
            nxgraph.add_edge(i0, i1, weight=-weight)
        return nx.algorithms.matching.max_weight_matching(nxgraph, maxcardinality=maxcardinality)

    @staticmethod
    def match_blossomv(edges: list, num_nodes: float = 0, **kwargs) -> list:
        """Finds the minimum-weight matching of a list of ``edges`` using `Blossom V <https://pub.ist.ac.at/~vnk/software.html>`_.

        Parameters
        ----------
        edges : [[nodeA, nodeB, distance(nodeA,nodeB)],...]
            A graph defined by a list of edges.

        Returns
        -------
        list
            Minimum weight matching in the form of [[nodeA, nodeB],..].
        """

        if num_nodes == 0:
            return []
        try:
            folder = os.path.dirname(os.path.abspath(__file__))
            PMlib = ctypes.CDLL(folder + "/blossom5-v2.05.src/PMlib.so")
        except:
            raise FileNotFoundError("Blossom5 library not found. See docs.")

        PMlib.pyMatching.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        PMlib.pyMatching.restype = ndpointer(dtype=ctypes.c_int, shape=(num_nodes,))

        # initialize ctypes array and fill with edge data
        numEdges = len(edges)
        nodes1 = (ctypes.c_int * numEdges)()
        nodes2 = (ctypes.c_int * numEdges)()
        weights = (ctypes.c_int * numEdges)()

        for i in range(numEdges):
            nodes1[i] = edges[i][0]
            nodes2[i] = edges[i][1]
            weights[i] = edges[i][2]

        matching = PMlib.pyMatching(ctypes.c_int(num_nodes), ctypes.c_int(numEdges), nodes1, nodes2, weights)
        return [[i0, i1] for i0, i1 in enumerate(matching) if i0 > i1]

    @staticmethod
    def get_qubit_distances(qubits: LA, size: Tuple[float, float]):
        """Computes the distance between a list of qubits.

        On a toric lattice, the shortest distance between two qubits may be one in four directions due to the periodic boundary conditions. The ``size`` parameters indicates the length in both x and y directions to find the shortest distance in all directions.
        """
        edges = []
        for i0, q0 in enumerate(qubits[:-1]):
            (x0, y0), z0 = q0.loc, q0.z
            for i1, q1 in enumerate(qubits[i0 + 1 :]):
                (x1, y1), z1 = q1.loc, q1.z
                wx = int(x0 - x1) % (size[0])
                wy = int(y0 - y1) % (size[1])
                wz = int(abs(z0 - z1))
                weight = min([wy, size[1] - wy]) + min([wx, size[0] - wx]) + wz
                edges.append([i0, i1 + i0 + 1, weight])
        return edges

    def _correct_matched_qubits(self, aq0: AncillaQubit, aq1: AncillaQubit) -> float:
        """Flips the values of edges between two matched qubits by doing a walk in between."""
        ancillas = self.code.ancilla_qubits[self.code.decode_layer]
        pseudos = self.code.pseudo_qubits[self.code.decode_layer]
        dq0 = ancillas[aq0.loc] if aq0.loc in ancillas else pseudos[aq0.loc]
        dq1 = ancillas[aq1.loc] if aq1.loc in ancillas else pseudos[aq1.loc]
        dx, dy, xd, yd = self._walk_direction(aq0, aq1, self.code.size)
        xv = self._walk_and_correct(dq0, dy, yd)
        self._walk_and_correct(dq1, dx, xd)
        return dy + dx + abs(aq0.z - aq1.z)

    @staticmethod
    def _walk_direction(q0: AncillaQubit, q1: AncillaQubit, size: Tuple[float, float]):
        """Finds the closest walking distance and direction."""
        (x0, y0) = q0.loc
        (x1, y1) = q1.loc
        dx0 = int(x0 - x1) % size[0]
        dx1 = int(x1 - x0) % size[0]
        dy0 = int(y0 - y1) % size[1]
        dy1 = int(y1 - y0) % size[1]
        dx, xd = (dx0, (0.5, 0)) if dx0 < dx1 else (dx1, (-0.5, 0))
        dy, yd = (dy0, (0, -0.5)) if dy0 < dy1 else (dy1, (0, 0.5))
        return dx, dy, xd, yd

    def _walk_and_correct(self, qubit: AncillaQubit, length: float, key: str):
        """Corrects the state of a qubit as it traversed during a walk."""
        for _ in range(length):
            try:
                qubit = self.correct_edge(qubit, key)
            except:
                break
        return qubit

    def _get_weight_of_full_correction(self, aq0: AncillaQubit, aq1: AncillaQubit) -> float:
        """Aids in determining the weight for the correction used in testing the lazy decoder."""
        dx, dy, xd, yd = self._walk_direction(aq0, aq1, self.code.size)
        return dy + dx + abs(aq0.z - aq1.z)
    
    def _weight_correction(self, syndromes: LA, matching: list, **kwargs):
        """Aids in determining the weight for the correction used in testing the lazy decoder."""
        weight = 0
        for i0, i1 in matching:
            weight += self._get_weight_of_full_correction(syndromes[i0], syndromes[i1])
        return weight
    
    def _lazy_walk_direction(self, loc_q0: Tuple[float, float], loc_q1: Tuple[float, float], size: Tuple[float, float]):
        
        # Ensures that the correct time-layer is used
        ancillas = self.code.ancilla_qubits[self.code.decode_layer]
        aq1 = ancillas[loc_q1]

        x0, y0 = loc_q0
        x1, y1 = loc_q1

        dx0 = int(x0 - x1) % size[0]
        dx1 = int(x1 - x0) % size[0]
        dy0 = int(y0 - y1) % size[1]
        dy1 = int(y1 - y0) % size[1]
        
        # Calculate direction key for aq1 for edge to aq0
        xd = (0.5, 0) if dx0 < dx1 else (-0.5, 0) if dx0 > dx1 else (0, 0)
        yd = (0, 0.5) if dy0 < dy1 else (0, -0.5) if dy0 > dy1 else (0, 0)
        
        # Return correct key in cases of either faulty measurement or qubit error
        if xd != (0, 0):
            return (aq1, xd)
        elif yd != (0, 0):
            return (aq1, yd)
        else:
            return (aq1, "Time")

    
# PROFILER FUNCTIONS

    # def lazy_checking_profiler_1(self, syndromes: LA, **kwargs):
    #     profiler = line_profiler.LineProfiler()
    #     profiler.add_function(self.correct_edge)  # Add any additional functions to profile

    #     # Profile the code line by line
    #     @profiler
    #     def profiled_lazy_checking(syndromes):
    #         failure = False
    #         error_list = []
    #         match_found = True

    #         if len(syndromes) == 0:
    #             return

    #         while len(syndromes) > 0:
    #             match_found = False
    #             data_qubits_dictionary = syndromes[0].parity_qubits.items()

    #             for a in range(1, len(syndromes)):
    #                 try:
    #                     shared_data_qubit_key = next((key for key, dq1 in data_qubits_dictionary for j, dq2 in syndromes[a].parity_qubits.items() if dq1 == dq2), None)
    #                     if shared_data_qubit_key is not None:
    #                         error_list.append([syndromes[0], shared_data_qubit_key])
    #                         del syndromes[a]
    #                         del syndromes[0]
    #                         match_found = True
    #                         break
    #                 except StopIteration:
    #                     pass

    #             if not match_found:
    #                 failure = True
    #                 break

    #         if failure:
    #             return "Failure"
    #         else:
    #             for correction in error_list:
    #                 self.correct_edge(correction[0], correction[1])

    #     # Run the profiled version of the function
    #     profiled_lazy_checking(syndromes)

    #     # Print the profiling results
    #     profiler.print_stats()

    # def lazy_checking_profiler_2(self, syndromes: LA, edges, **kwargs):
    #     profiler = line_profiler.LineProfiler()

    #     @profiler
    #     def populate_error_list(edges, syndromes):
    #         error_list = set()
    #         syndromes = set(syndromes)
    #         for edge in edges:
    #             ancilla1, ancilla2 = edge

    #             if ancilla1 in syndromes and ancilla2 in syndromes:
    #                 error_list.add(tuple(edge))
    #                 syndromes.remove(ancilla1)
    #                 syndromes.remove(ancilla2)
    #         return error_list

    #     @profiler
    #     def correct_errors(error_list):
    #         for pair in error_list:
    #             key = self._lazy_walk_direction(pair[0].loc, pair[1].loc, self.code.size)
    #             self.correct_edge(pair[1], key)

    #     error_list = populate_error_list(edges, syndromes)

    #     if syndromes:
    #         return "Failure"

    #     correct_errors(error_list)

    #     profiler.print_stats()