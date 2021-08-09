# Copyright 2021 Kun Liu
 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
 
#     http://www.apache.org/licenses/LICENSE-2.0
 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


'''
Created Date: Friday, July 30th 2021, 2:56:48 pm

Author: Kun Liu

Description: The top main program of getting 2-RDM from N-representability.

Copyright (c) 2021 Your Company
'''

import numpy as np
import importlib
from rdm_mapping_functions import *
from typing import Tuple
from NRepresentability import NRepresentability
from pyscf import gto
import sys


def save_ideal_and_noisy_2rdm(output_path: str, ideal_rdms: np.ndarray, noisy_rdms: np.ndarray) -> None:
    """Save the rdm with numpy array

    Args:
        output_path (str): The filepath to save the 2-RDM
        ideal_rdms (np.ndarray): A list of 2-RDM, and the element of it is a 4-dimensional np.ndarray,
        so it is a 5-dimensional np.ndarray
        noisy_rdms (np.ndarray): Same with ideal_rmds

    Returns:
        None: None
    """
    ideal_narray = np.array(ideal_rdms) # 5 dimension array
    noisy_narray = np.array(noisy_rdms)
    np.savez(output_path, ideal_narray, noisy_narray)
    print('narray saved to %s' % output_path)
    return None


def get_energy(two_D: np.array, atom: str, basis: str, num_particles: Tuple[int, int]) -> float:
    """Hartree-Fock on a superconducting qubit quantum computer
        https://arxiv.org/pdf/2004.04174
        Calculate energe from 2-RDM and 1-RDM, according to Equation A14 in the paper.

        The method of obtaining h_{ij} and V_{ijkl},
        see https://mattermodeling.stackexchange.com/questions/4284/how-are-1-electron-and-2-electron-integrals-done-in-pyscf
    Args:
        two_D: 4-index tensor

    Returns:
        float: energy
    """
    # reshape 2-RDM
    dim_two_D = two_D.shape[0]
    new_dim_two_D = dim_two_D // 2
    new_two_D = np.zeros((new_dim_two_D, new_dim_two_D, new_dim_two_D, new_dim_two_D))
    for i in range(new_dim_two_D):
        for j in range(new_dim_two_D):
            for k in range(new_dim_two_D):
                for l in range(new_dim_two_D):
                    new_two_D[i,j,k,l] = two_D[i,j,k,l] \
                        + two_D[i+new_dim_two_D, j+new_dim_two_D, k+new_dim_two_D, l+new_dim_two_D]

    # reshape 1-RDM
    one_D = np.einsum('prrq', two_D) / (sum(num_particles) - 1)
    dim_one_D = one_D.shape[0]
    new_dim_one_D = dim_one_D // 2
    new_one_D = np.zeros((new_dim_one_D, new_dim_one_D))
    for i in range(new_dim_one_D):
        for j in range(new_dim_one_D):
            new_one_D[i,j] = one_D[i,j] \
                + one_D[i+new_dim_one_D, j+new_dim_one_D]

    # get h_{ij} and V_{i,j,k,l}
    mol = gto.M(atom=atom, basis=basis)
    h_matrix = mol.get_hcore()
    V_matrix = mol.intor('int2e', aosym='s1')

    energy = np.sum(h_matrix * new_one_D) + np.sum(V_matrix * new_two_D)
    
    return energy


def read_ideal_and_noisy_2rdm(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    # read the rdm with numpy array
    # see docs of reading narray: https://blog.csdn.net/kaever/article/details/106472316
    # try:
    datafile = np.load(filepath)
    # catch
    
    noisy_rdms = datafile['arr_1']
    ideal_rdms = datafile['arr_0']
    return ideal_rdms, noisy_rdms


def test_error_mitigate():
    probs1 = [0.005]
    probs2 = [0.01]
    atoms = ['H 0.0 0.0 0.0; H 0.0 0.0 0.735',
             'H .0 .0 .0; H .0 .0 2.5; H .0 .0 5; H .0 .0 7.5']

    dirpath = './2rdm_array_data/'
    atom = atoms[1]
    basis = 'sto3g'
    num_particles = (2, 2)
    num_spin_orbitals = 8


    for i1, p1 in enumerate(probs1):
        for i2, p2 in enumerate(probs2):
            filename = 'atom_%s_p1_%f_p2_%f.npz' % (atom, p1, p2)
            filepath = dirpath + filename

            ideal_rdms, noisy_rdms = read_ideal_and_noisy_2rdm(filepath)
            ideal_rdm = ideal_rdms[0]
            noisy_rdm = noisy_rdms[0]

            print('1-qubit and 2-qubit gate error rate: ', p1, p2)
            nrep = NRepresentability(num_spin_orbital=num_spin_orbitals,
                    num_particle=num_particles,
                    two_particle_rdm_ideal=ideal_rdm,
                    two_particle_rdm_noisy=noisy_rdm)

            print('trace of ideal and noisy 2-RDM: ',
                  nrep.get_matrix_from_2rdm(ideal_rdm).trace(),
                  nrep.get_matrix_from_2rdm(noisy_rdm).trace()) 
                  
            init_fidelity = nrep.dist_of_2_2rdm('fidelity', noisy_rdm, ideal_rdm)
            init_norm = nrep.dist_of_2_2rdm('norm', noisy_rdm, ideal_rdm)
            print('fidelity and norm distance before error mitigation: ', init_fidelity, init_norm)

            nrep.optimize()
            final_fidelity = nrep.loss('fidelity')
            final_norm = nrep.loss('norm')
            print('fidelity and norm distance after error mitigation: ', final_fidelity, final_norm)
            final_trace = nrep.get_matrix_from_2rdm(nrep.two_particle_rdm_optimized).trace()
            print('trace of error-mitigated 2-RDM: ', final_trace)

            # then you could calculate energy with 2-RDM
            ideal_energy = get_energy(nrep.two_particle_rdm_ideal, atom, basis, num_particles)
            noisy_energy = get_energy(nrep.two_particle_rdm_noisy, atom, basis, num_particles)
            optimized_energy = get_energy(nrep.two_particle_rdm_optimized, atom, basis, num_particles)
            print('Energy calculation with ideal 2-RDM, noisy 2-RDM, and error-mitigated 2-RDM: ',
                ideal_energy, noisy_energy, optimized_energy)

    return


if __name__ == '__main__':
    # change NRepresentabily.py, then reload it
    # to make sure that Python executor refresh the changes
    importlib.reload(sys.modules['NRepresentability'])
    from NRepresentability import NRepresentability

    test_error_mitigate()