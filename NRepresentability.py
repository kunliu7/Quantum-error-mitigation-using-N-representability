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
Created Date: Wednesday, July 28th 2021, 11:21:03 am

Author: Kun Liu

Description: 
    This program is the implementation of N-representability.

    Reference see
    Lanssens2017: Method For Making 2-Electron Response Reduced Density Matrices Approximately N-representable. 

    Available online at https://arxiv.org/pdf/1707.01022.
'''

import numpy as np
from scipy import optimize
import math

from rdm_mapping_functions import *

class NRepresentability():
    def __init__(self, num_spin_orbital: int, num_particle: tuple, 
            two_particle_rdm_ideal: np.ndarray, two_particle_rdm_noisy: np.ndarray) -> None:
        self.two_particle_rdm_noisy = two_particle_rdm_noisy
        self.two_particle_rdm_ideal = two_particle_rdm_ideal
        self.two_particle_rdm_optimized = None

        self.num_spin_orbital = num_spin_orbital
        self.num_particle = sum(num_particle) # num_particle = (1, 1): one spin-up, one spin-down
        self.num_hole = num_spin_orbital - self.num_particle

        self.prev_two_particle_rdm_optimized = None

        # big capital variables indicates constant
        # calculate ideal trace here
        # D: two particle rdm
        # Q: two hole rdm
        # G: particle-hole rdm
        self.TRACES_IDEAL = {'D': self.num_particle * (self.num_particle - 1),
                             'Q': (self.num_spin_orbital - self.num_particle) 
                                * (self.num_spin_orbital - self.num_particle - 1),
                             'G': self.num_particle * (self.num_spin_orbital - self.num_particle + 1)
                            }

        # max iteration for optimization
        self.NUM_MAX_ITER = 10000
        self.THRESHOLD_CONVERGE = 1e-6
        self.prev_cost = 0
        return
    

    def fix_two_rdm(self, two_rdm: np.ndarray, rdm_type: str) -> np.ndarray:
        '''
        Args:
            two_rdm (numpy.ndarray): The {two particle, two hole, particle-hole} RDM as a 4-index tensor.
        Returns:
            two_rdm (numpy.ndarray): The {two particle, two hole, particle-hole} RDM as a 4-index tensor.
        '''

        # transform 4-index tensor to 2-index tensor, which is a matrix of shape (dim*dim, dim*dim)
        matrix_rdm = self.get_matrix_from_2rdm(two_rdm)

        # symmetrize the 2-DM
        sym_rdm = (matrix_rdm + matrix_rdm.T) / 2

        # get the eigenvalues and eigenvectors
        # eigh: eigenvalues and eigenvectors of a real symmetric or complex Hermitian (conjugate symmetric) array.
        result = np.linalg.eigh(sym_rdm) 
        eigvals = result[0] # eigenvalue list, which is a vector
        eigvecs = np.array(result[1]) # eigenvector list, which is a matrix

        # shift the eigenvalues
        ideal_trace = self.TRACES_IDEAL[rdm_type]
        shifted_eigvals = self.shift_eigval_with_scipy(eigvals, ideal_trace)
        shifted_eigvals = np.array(shifted_eigvals)

        # retrieve the RDM
        updated_matrix_of_rdm = np.dot(np.dot(eigvecs, np.diag(shifted_eigvals)), eigvecs.T)

        # reshape from a matrix (2-index tensor) to a 2-RDM (4-index tensor)
        return self.get_2rdm_from_matrix(updated_matrix_of_rdm)
    

    def shift_eigval_with_scipy(self, eigvals: list, ideal_trace_of_2rdm: float) -> list:
        """Shift eigenvalues with SciPy. Once tried SimPy, but it is too slow.
        See paper Eq.40-43 for details.

        Args:
            eigvals (list[float]): eigenvalues
            ideal_trace_of_2rdm (float): ideal trace value

        Returns:
            list: shifted eigenvalues
        """
        def _func(x):
            temp = 0
            for eigval in eigvals:
                temp += np.heaviside(eigval - x, 0.5) * (eigval - x) 
            temp -= ideal_trace_of_2rdm 
            return temp

        # according to the appendix of the paper, Page 12, the root is unique
        # bisect method needs to specify the interval, which cannot secure a root
        # so use scipy.optimize.root instead
        # sigma0 = bisect(_func, np.min(eigvals) - 10, np.max(eigvals) + 10)
        sigma0 = optimize.root(_func, -1000).x[0]

        shifted_eigvals = [0 if eigval <= sigma0 else eigval - sigma0 for eigval in eigvals]

        return shifted_eigvals

    
    def optimize(self) -> bool:
        """See the paper FIG.1 for detail.

        Returns:
            bool: whether converge or not.
        """
        # set initial value
        counter = 0
        # the 2rdm keeps changing in the loop
        two_particle_rdm = self.two_particle_rdm_noisy

        # the optimized 2rdm of each step
        self.two_particle_rdm_optimized = self.two_particle_rdm_noisy

        # the optimized 2rdm of the last step
        self.prev_two_particle_rdm_optimized = None

        while not self.is_two_particle_rdm_converge():
            # fix D
            fixed_two_particle_rdm = self.fix_two_rdm(two_particle_rdm, rdm_type='D')

            # transfer to Q
            one_particle_rdm = map_two_pdm_to_one_pdm(fixed_two_particle_rdm, self.num_particle)
            two_hole_rdm = map_two_pdm_to_two_hole_dm(fixed_two_particle_rdm, one_particle_rdm)

            # fix Q
            fixed_two_hole_rdm = self.fix_two_rdm(two_hole_rdm, rdm_type='Q')

            # transfer Q to D
            one_hole_rdm = map_two_hole_dm_to_one_hole_dm(fixed_two_hole_rdm, self.num_hole)
            one_particle_rdm = map_one_hole_dm_to_one_pdm(one_hole_rdm)
            two_particle_rdm = map_two_hole_dm_to_two_pdm(fixed_two_hole_rdm, one_particle_rdm)
            
            # transfer D to G
            one_particle_rdm = map_two_pdm_to_one_pdm(two_particle_rdm, self.num_particle)
            particle_hole_rdm = map_two_pdm_to_particle_hole_dm(two_particle_rdm, one_particle_rdm)

            # fix G
            fixed_particle_hole_rdm = self.fix_two_rdm(particle_hole_rdm, rdm_type='G')

            # transfer to D
            one_particle_rdm = map_particle_hole_dm_to_one_pdm(fixed_particle_hole_rdm, self.num_particle, self.num_spin_orbital)
            two_particle_rdm = map_particle_hole_dm_to_two_pdm(fixed_particle_hole_rdm, one_particle_rdm)
            
            # record the current optimized RDM
            self.prev_two_particle_rdm_optimized = self.two_particle_rdm_optimized 
            self.two_particle_rdm_optimized = two_particle_rdm
            counter += 1

            if counter % 100 == 0:
                print('optimize step: ', counter)

            if counter > self.NUM_MAX_ITER:
                print('optimize over %d steps and have not converge, terminate the optimization.'
                     % (self.NUM_MAX_ITER))
                return False
        
        return True 


    def is_two_particle_rdm_converge(self) -> bool:
        """Paper on Page 6, at the bottom of the left column:
        Convergence is measured by the error on the traces
        and the magnitude of the largest negative eigenvalue.

        Returns:
            bool: Is converge or not
        """
        if self.prev_two_particle_rdm_optimized is None:
            # the SDP hasn't started yet
            return False

        matrix_of_2rdm = self.get_matrix_from_2rdm(self.two_particle_rdm_optimized)
        curr_trace = matrix_of_2rdm.trace()
        eigvals, eigvecs = np.linalg.eigh(matrix_of_2rdm)
        min_eigval = min(eigvals)
        magnitude_of_largest_neg_eigval = abs(min_eigval) if min_eigval < 0 else 0
        
        curr_cost = abs(curr_trace - self.TRACES_IDEAL['D']) \
                  + magnitude_of_largest_neg_eigval

        # fidelity = self.dist_of_2_2rdm('fidelity', self.prev_two_particle_rdm_optimized,
        #                            self.two_particle_rdm_optimized)
        
        # norm = self.dist_of_2_2rdm('norm', self.prev_two_particle_rdm_optimized,
        #                            self.two_particle_rdm_optimized)

        # as describe in the paper
        dist = abs(self.prev_cost - curr_cost)
        self.prev_cost = curr_cost

        return math.isclose(dist, 0.0, abs_tol=self.THRESHOLD_CONVERGE)
    

    def dist_of_2_2rdm(self, dist_type: str, rdm1: np.ndarray, rdm2: np.ndarray) -> float:
        dist = 0
        if dist_type == 'fidelity':
            m1 = self.get_matrix_from_2rdm(rdm1)
            m2 = self.get_matrix_from_2rdm(rdm2)
            # rdm is real
            dist = np.matmul(m1, m2).trace()
        else:
            diff = rdm1 - rdm2
            dist = np.linalg.norm(diff)
        
        return dist


    def get_matrix_from_2rdm(self, rdm: np.ndarray, transpose: bool=True) -> np.ndarray:
        """The mapping functions copied from OpenFermion, that is in 'rdm_mapping_functions.py',
        require indexes of 2-RDM being arranged in order:

        two_rdm[p, q, r, s] == `\langle a_{p}^{\dagger}a_{q}^{\dagger}a_{r}a_{s} \rangle`

        For this reason, transposing the last two dimensions of 2-RDM is needed
        when transform a 2-RDM to a matrix. Besides that, we do not need to transpose.

        Args:
            rdm (np.ndarray): 2-RDM 
            transpose (bool, optional): whether to transpose the last two dimensions of the 2-RDM. Defaults to True.

        Returns:
            np.ndarray: the matrix w.r.t the 2-RDM
        """
        dim = rdm.shape[0]
        if transpose:
            rdm = rdm.transpose([0, 1, 3, 2])
        return np.reshape(rdm, newshape=(dim * dim, dim * dim))


    def get_2rdm_from_matrix(self, matrix: np.ndarray, transpose: bool=True) -> np.ndarray:
        """The mapping functions copied from OpenFermion, that is in 'rdm_mapping_functions.py',
        require indexes of 2-RDM being arranged in order:

        two_rdm[p, q, r, s] == `\langle a_{p}^{\dagger}a_{q}^{\dagger}a_{r}a_{s} \rangle`

        For this reason, transposing the last two dimensions of 2-RDM is needed
        when transform a 2-RDM to a matrix. Besides that, we do not need to transpose.

        Args:
            matrix (): the matrix w.r.t the 2-RDM 
            transpose (bool, optional): whether transpose the last two dimensions of 2-RDM. Defaults to True.

        Returns:
            np.ndarray: 2-RDM
        """
        dim = self.num_spin_orbital
        rdm = np.reshape(matrix, newshape=(dim, dim, dim, dim))
        if transpose:
            rdm = rdm.transpose([0, 1, 3, 2])
        return rdm  
    
        
    def is_rdm_singular_val_all_non_negative(self, rdm: np.ndarray=None) -> bool:
        if rdm == None:
            rdm = self.two_particle_rdm_optimized
        dim = rdm.shape[0]
        matrix = np.reshape(rdm, (dim*dim, dim*dim))
        u, sigmas, vt = np.linalg.svd(matrix)

        for s in sigmas:
            if s < 0:
                print("contain <0 singular value")
                return False
        print("all singular value >= 0")
        return True
   

    def loss(self, dist_type: str) -> float:
        return self.dist_of_2_2rdm(dist_type, self.two_particle_rdm_ideal, self.two_particle_rdm_optimized)