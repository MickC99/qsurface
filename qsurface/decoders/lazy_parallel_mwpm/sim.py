from typing import List, Tuple
from qsurface.codes.elements import AncillaQubit
from .._template import Sim
import networkx as nx
from numpy.ctypeslib import ndpointer
import ctypes
import os
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures





LA = List[AncillaQubit]


class Toric(Sim):
    """Minimum-Weight Perfect Matching decoder for the toric lattice.

    Parameters
    ----------
    args, kwargs
        Positional and keyword arguments are passed on to `.decoders._template.Sim`.
    """

    name = "Minimum-Weight Perfect Matching"
    short = "mwpm"

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
        parallel_processes = 4

        # Retrieve edge lists
        n = len(self.code.data_qubits)
        plaqs_edges = sum([[self.code.data_qubits[i][(x, y)].edges['x'].nodes for (x, y) in self.code.data_qubits[i]] + self.code.time_edges[i] for i in range(n-1)], []) + [self.code.data_qubits[n-1][(x, y)].edges['x'].nodes for (x, y) in self.code.data_qubits[n-1]]
        stars_edges = sum([[self.code.data_qubits[i][(x, y)].edges['z'].nodes for (x, y) in self.code.data_qubits[i]] + self.code.time_edges[i] for i in range(n-1)], []) + [self.code.data_qubits[n-1][(x, y)].edges['z'].nodes for (x, y) in self.code.data_qubits[n-1]]

        # Run parallel decoding
        self.lazy_parallel_decoding(plaqs, plaqs_edges, self.code.size[0], parallel_processes)
        self.lazy_parallel_decoding(stars, stars_edges, self.code.size[0], parallel_processes)


    def lazy_parallel_decoding(self, syndromes_list: LA, edges, d: int, parallel_processes: int):
        
        # Divide syndromes into An windows
        A_n_windows = self.divide_into_windows(syndromes_list, d, parallel_processes)
        
        edge_list_A = [edges.copy() for _ in A_n_windows]
        edge_list_B = [edges.copy() for _ in range(parallel_processes-1)]

        # Create parallel threads to run An windows in parallel
        with ThreadPoolExecutor(max_workers=parallel_processes) as executor:

            # Match syndromes in An windows
            matching_results_A_n = executor.map(self.lazy_attempt, A_n_windows.values(),  edge_list_A)
            syndrome_lists = list(A_n_windows.values())
            futures_A_n = [[] for _ in range(parallel_processes)]

            processed_indices = set()
            b_futures = []

            # Checks if syndromes can be committed to after completion of i window decoding
            for i, (matching, syndromes_sublist) in enumerate(zip(matching_results_A_n, syndrome_lists)):
                future = executor.submit(self.process_matching, syndromes_list, syndromes_sublist, matching, d, parallel_processes)
                futures_A_n[i] = future
                

                # Run Bn windows in parallel after adjacent An windows have been completed
                for i in range(parallel_processes-1):
                    if futures_A_n[i] != [] and futures_A_n[i + 1] != [] and futures_A_n[i].done() and futures_A_n[i+1].done():
                        
                        # Ensure that Bn window is not decoded more than once
                        if i not in processed_indices:
                            B_n_windows = self.second_divide_into_windows(syndromes_list, d, parallel_processes)
                            b_future = executor.submit(self.Bn_lazy_windows_matching, B_n_windows, edges, i)
                            b_futures.append(b_future)
                            processed_indices.add(i)

            # Wait for all An windows to complete
            for future in futures_A_n:
                future.result()
            
            # Final adjustment of Bn windows
            B_n_windows = self.second_divide_into_windows(syndromes_list, d, parallel_processes)
            
            # Correct the Bn windows of which the adjacent An windows have been decoded last
            for i in range(parallel_processes-1):
                if i not in processed_indices:
                    b_future = executor.submit(self.Bn_lazy_windows_matching, B_n_windows, edges, i)
                    b_futures.append(b_future)

            # Wait for all Bn windows to complete
            for future in b_futures:
                future.result()
    
    def lazy_attempt(self, syndromes: LA, edges, **kwargs):
        attempt = self.lazy_checking(syndromes, edges, **kwargs)
        if attempt == "Failure":
            matching = self.match_syndromes(syndromes)
            return matching
        else:
            return attempt
        
    # Implements the lazy decoder
    def lazy_checking(self, syndromes: LA, edges, **kwargs):
        matching = set()

        if len(syndromes) == 0:
            return matching
        
        # Create syndromes copy for original indices
        syndromes_copy = syndromes.copy()

        # Iterate over all edges
        for edge in edges:
            ancilla1, ancilla2 = edge

            # Check if both ancilla qubits are in the syndrome set
            if ancilla1 in syndromes_copy and ancilla2 in syndromes_copy:
                index1 = syndromes.index(ancilla1)
                index2 = syndromes.index(ancilla2)
                matching.add((index1, index2))
                syndromes_copy.remove(ancilla1)
                syndromes_copy.remove(ancilla2)
        
        # Return failure if there are syndromes left
        if syndromes_copy:
            return "Failure"
        else:
            return matching

    # Decodes the Bn-windows with Lazy Decoder implementation
    def Bn_lazy_windows_matching(self, B_n_plaqs: LA, edges: list, i: int, **kwargs):
        
        matching_b = self.lazy_checking(list(B_n_plaqs.values())[i], edges, **kwargs)
        if matching_b == "Failure":
            matching_b = self.match_syndromes(list(B_n_plaqs.values())[i], **kwargs)
        self.correct_matching(list(B_n_plaqs.values())[i], matching_b)
        
    # Divides decoding graph into windows and gaps
    def divide_into_windows(self, syndromes, d, parallel_processes):
        windows = {}

        window_size = 3*d
        
        for i in range(parallel_processes):
            windows[i] = []
        # O(n)
        for syndrome in syndromes:
            window_index = syndrome.z // (window_size * (4/3))
            window_point = float(syndrome.z / (window_size * (4/3)))

            if 0 <= (window_point % 1) < 0.75:
                windows[window_index].append(syndrome)

        return windows
    
    # Divides decoding graph into windows and gaps
    def second_divide_into_windows(self, syndromes, d, parallel_processes):
        windows = {}

        window_size = 3 * d

        for i in range(parallel_processes-1):
            windows[i] = []
                
        # O(n)
        for syndrome in syndromes:
            if syndrome.z < (2/3)*window_size or syndrome.z > (4*d*parallel_processes - d) - (2/3)*window_size - 1:
                continue
            window_index = (syndrome.z - d) // (window_size+d)
            window_point = float((syndrome.z - d) / (window_size+d))

            if 0.25 <= (window_point % 1) < 1:
                windows[window_index].append(syndrome)

        return windows
    

    # Checks whether a given qubit is in the committed region of a window
    def commit(self, qubit, d, parallel_processes):
        window_size = 3 * d
        total_windows = parallel_processes

        layer_index = qubit.z // (window_size * (4 /3))

        if layer_index == 0:
            return 0 <= qubit.z < (2 / 3) * window_size
        elif layer_index == total_windows - 1:
            last_window_start = total_windows * window_size
            return (last_window_start + (1/3)*window_size - 1) < qubit.z
        else:
            window_start = layer_index * (window_size + d)
            return (window_size / 3 <= qubit.z - window_start < 2 * window_size / 3)
    
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
    
    def _correct_matched_qubits_nbuf_ncom(self, aq0: AncillaQubit, aq1: AncillaQubit, d: int, parallel_processes: int, full_syndromes: LA) -> float:
        """Flips the values of edges between two matched qubits by doing a walk in between."""
        ancillas = self.code.ancilla_qubits[self.code.decode_layer]
        pseudos = self.code.pseudo_qubits[self.code.decode_layer]
        dq0 = ancillas[aq0.loc] if aq0.loc in ancillas else pseudos[aq0.loc]
        dq1 = ancillas[aq1.loc] if aq1.loc in ancillas else pseudos[aq1.loc]

        dx, dy, xd, yd = self._walk_direction(aq0, aq1, self.code.size)
        correction = False
        if self.commit(aq0, d, parallel_processes):
            buffer_layer_ancilla_aq0 = self.code.ancilla_qubits[aq1.z][aq0.loc]
            if buffer_layer_ancilla_aq0 == aq1:
                full_syndromes.remove(aq1)
                correction = True
            else:
                full_syndromes.append(buffer_layer_ancilla_aq0)
        elif self.commit(aq1, d, parallel_processes):
            buffer_layer_ancilla_aq1 = self.code.ancilla_qubits[aq0.z][aq1.loc]
            if buffer_layer_ancilla_aq1 == aq0:
                full_syndromes.remove(aq0)
                correction = True
            else:
                full_syndromes.append(buffer_layer_ancilla_aq1)

        if correction: 
            xv = self._walk_and_correct_nbuf_ncom(dq0, dy, yd)
            self._walk_and_correct_nbuf_ncom(dq1, dx, xd)
        
        
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
    
    def _walk_and_correct_nbuf_ncom(self, qubit: AncillaQubit, length: float, key: str):
        """Corrects the state of a qubit as it traversed during a walk."""
        for _ in range(length):
            try:
                qubit = self.correct_edge(qubit, key)
            except:
                break
        return qubit

    # Determines whether corrections need to be applied or not based on presence in ncom
    def process_matching(self, full_syndromes: LA, syndromes: LA, matching: list, d: int, parallel_processes: int):
        for i0, i1 in matching:
            q0 = syndromes[i0]
            q1 = syndromes[i1]

            # Commit to corrections only if both elements are in the committed region
            if self.commit(q0, d, parallel_processes) and self.commit(q1, d, parallel_processes):
                self._correct_matched_qubits(q0, q1)
            
            # Construct path when only one syndrome is in the committed region
            elif self.commit(q0, d, parallel_processes) ^ self.commit(q1, d, parallel_processes):
                self._correct_matched_qubits_nbuf_ncom(q0,q1,d,parallel_processes,full_syndromes)


class Planar(Toric):
    """Minimum-Weight Perfect Matching decoder for the planar lattice.

    Additionally to all edges, virtual qubits are added to the boundary, which connect to their main qubits.Edges between all virtual qubits are added with weight zero.
    """

    def decode(self, **kwargs):
        # Inherited docstring
        plaqs, stars = self.get_syndrome(find_pseudo=True)
        self.correct_matching(plaqs, self.match_syndromes(plaqs, **kwargs))
        self.correct_matching(stars, self.match_syndromes(stars, **kwargs))

    def correct_matching(self, syndromes: List[Tuple[AncillaQubit, AncillaQubit]], matching: list):
        # Inherited docstring
        weight = 0
        for i0, i1 in matching:
            if i0 < len(syndromes) or i1 < len(syndromes):
                aq0 = syndromes[i0][0] if i0 < len(syndromes) else syndromes[i0 - len(syndromes)][1]
                aq1 = syndromes[i1][0] if i1 < len(syndromes) else syndromes[i1 - len(syndromes)][1]
                weight += self._correct_matched_qubits(aq0, aq1)
        return weight

    @staticmethod
    def get_qubit_distances(qubits, *args):
        """Computes the distance between a list of qubits.

        On a planar lattice, any qubit can be paired with the boundary, which is inhabited by `~.codes.elements.PseudoQubit` objects. The graph of syndromes that supports minimum-weight matching algorithms must be fully connected, with each syndrome connecting additionally to its boundary pseudo-qubit, and a fully connected graph between all pseudo-qubits with weight 0.
        """
        edges = []

        # Add edges between all ancilla-qubits
        for i0, (a0, _) in enumerate(qubits):
            (x0, y0), z0 = a0.loc, a0.z
            for i1, (a1, _) in enumerate(qubits[i0 + 1 :], start=i0 + 1):
                (x1, y1), z1 = a1.loc, a1.z
                wx = int(abs(x0 - x1))
                wy = int(abs(y0 - y1))
                wz = int(abs(z0 - z1))
                weight = wy + wx + wz
                edges.append([i0, i1, weight])

        # Add edges between ancilla-qubits and their boundary pseudo-qubits
        for i, (ancilla, pseudo) in enumerate(qubits):
            (xs, ys) = ancilla.loc
            (xb, yb) = pseudo.loc
            weight = xb - xs if ancilla.state_type == "x" else yb - ys
            edges.append([i, len(qubits) + i, int(abs(weight))])

        # Add edges of weight 0 between all pseudo-qubits
        for i0 in range(len(qubits), len(qubits)):
            for i1 in range(i0 + 1, len(qubits) - 1):
                edges.append([i0, i1, 0])
        return edges

    @staticmethod
    def _walk_direction(q0, q1, *args):
        # Inherited docsting
        (x0, y0), (x1, y1) = q0.loc, q1.loc
        dx, dy = int(x0 - x1), int(y0 - y1)
        xd = (0.5, 0) if dx > 0 else (-0.5, 0)
        yd = (0, -0.5) if dy > 0 else (0, 0.5)
        return abs(dx), abs(dy), xd, yd

class Weight_0_toric(Toric):
    pass
class Weight_3_toric(Toric):
    pass

class Weight_4_toric(Toric):
    pass