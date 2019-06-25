import toric_lat as tl
import planar_lat as pl
import sys

sys.path.insert(0, '../fault_tolerance_simulations/')
# sys.path.insert(0, '../Code_Eduardo/')
#
# import Eduardo
import st



def toric_2D_MWPM(size, pX, pZ, savefile=False, pauli_file=None, plot_load=False):
    TL = tl.lattice(size, pauli_file, None, plot_load)
    TL.init_pauli_errors(pX, pZ, savefile)
    TL.measure_stab()
    TL.get_matching_MWPM()
    logical_error = TL.logical_error()
    correct = True if logical_error == [False, False, False, False] else False
    return correct


def toric_2D_peeling(size, pE, pX=0, pZ=0, savefile=False, erasure_file=None, pauli_file=None, plot_load=False):
    TL = tl.lattice(size, pauli_file, erasure_file, plot_load)
    TL.init_erasure_errors_region(pE, savefile)
    TL.init_pauli_errors(pX, pZ, savefile)
    TL.measure_stab()
    TL.get_matching_peeling()
    logical_error = TL.logical_error()
    correct = True if logical_error == [False, False, False, False] else False
    return correct


def planar_2D_MWPM(size, pX, pZ, savefile=False, pauli_file=None, plot_load=False):
    PL = pl.lattice(size, pauli_file, None, plot_load)
    PL.init_pauli_errors(pX, pZ, savefile)
    PL.measure_stab()
    PL.get_matching_MWPM()
    logical_error = PL.logical_error()
    correct = True if logical_error == [False, False] else False
    return correct


def toric_2D_nickerson(size, pX=0, pZ=0):
    output = st.run2D(size, pX, pZ)
    correct = True if output == [[1, 1], [1, 1]] else False
    return correct


# def toric_2D_eduardo(size, pX=0, pZ=0):
#     output = Eduardo.thres_calc("toric", size, pX, pZ, 0, 1, 1, 0)
#     correct = True if output == 0 else False
#     return correct