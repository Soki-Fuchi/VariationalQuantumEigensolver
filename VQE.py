from cirq.ops import *
import cirq
import random
from cirq.google import XmonSimulator
from openfermion.hamiltonians import MolecularData
from openfermionpyscf import run_pyscf
from openfermion.transforms import get_fermion_operator, jordan_wigner, bravyi_kitaev
from openfermion.utils import eigenspectrum, expectation
from openfermion.transforms import get_sparse_operator
from openfermion.ops import FermionOperator
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from multiprocessing import *

class VQE:
    def __init__(self, n_qubit, n_point):
        self.n_qubit = n_qubit
        self.bond_lengths = np.linspace(0, 3, n_point+1)[1:]
        self.qubits = [cirq.LineQubit(i) for i in range(self.n_qubit)]

    def make_ansatz(self, theta_list=None, simulator=XmonSimulator()):
        if theta_list is None:
            #theta_list = self.theta_list
            theta_list = self.make_param()
        else:
            theta_list = theta_list

        circuit = self.first_cycle_circuit(theta_list, self.n_qubit)
        result = simulator.simulate(circuit)
        psi = result.final_state.reshape(len(result.final_state),1)
        return psi
    
    def make_param(self):
        #theta_list has involved dammy theta for CNOT!!! so has n*2 extra theta
        np.random.seed(seed=1)
        n_theta = self.n_qubit*6
        theta_list = np.random.randint(-3, 3, (1, n_theta)) +np.random.randn(1, n_theta)
        #self.theta_list = theta_list[0]
        theta_list = theta_list[0]
        return theta_list
    
    def first_cycle_circuit(self, theta_list=None, n_qubit=None):
        
        if theta_list is None:
            #theta_list = self.theta_list
            theta_list = self.make_param()
        else:
            theta_list = theta_list

        if n_qubit is None:
            n_qubit = self.n_qubit
        else:
            n_qubit = n_qubit
    
        circuit = cirq.Circuit()
        count = 0
        for theta in theta_list:
            if count // n_qubit == 0:
                circuit.append(RotYGate(rads=theta).on(self.qubits[count % n_qubit]))
                count += 1
            
            elif count // n_qubit == 1:
                circuit.append(RotZGate(rads=theta).on(self.qubits[count % n_qubit]))
                count += 1
                
            elif count // n_qubit == 2:
                if count % n_qubit == n_qubit-1:
                    count += 1
                    continue
                circuit.append(CNOT(self.qubits[count % n_qubit], self.qubits[count % n_qubit +1]))
                count += 1
            
            elif count // n_qubit == 3:
                circuit.append(RotYGate(rads=theta).on(self.qubits[count % n_qubit]))
                count += 1
    
            elif count // n_qubit == 4:
                circuit.append(RotZGate(rads=theta).on(self.qubits[count % n_qubit]))
                count += 1
            
            elif count // n_qubit == 5:
                if count % n_qubit == n_qubit-1:
                    count += 1
                    continue
                circuit.append(CNOT(self.qubits[count % n_qubit], self.qubits[count % n_qubit +1]))
                count += 1
        
        return circuit 

    def get_gradient(self, hamiltonian):
        grad = np.zeros_like(self.make_param())
        h = 1e-4
        for i in range(len(self.make_param())):
            tmp_diff = np.zeros_like(self.make_param())
            tmp_diff[i] = h
            psi_plus_list = np.array([psi for psi in self.make_ansatz(self.make_param() + tmp_diff)])
            psi_minus_list = np.array([psi for psi in self.make_ansatz(self.make_param() - tmp_diff)])
            grad_theta = (get_expect(hamiltonian, np.array(psi_plus_list)) - get_expect(hamiltonian, np.array(psi_minus_list)))/ (2*h)
            #grad[count][i] = np.real(grad_theta)
            grad[i] = grad_theta.real
        return grad
    
    def gradient_descent(self, hamiltonian, lr=0.01, n_step=100):
        update_list = self.make_param().copy()
        
        for _ in range(n_step):
            grad = self.get_gradient(hamiltonian)
            update_list -= lr * grad
        
        return update_list

    def opt_energy(self, bond_length):
        psi = self.make_ansatz()
        expect_val = get_expect(get_BK_ham(bond_length), psi)
        optimize_energy = expect_val.real
        return optimize_energy
    
    def get_opt_energy(self):
        opt_list = []
        for bond_length in self.bond_lengths:
            bk_ham = get_BK_ham(bond_length)
            update_param = self.gradient_descent(bk_ham)
            opt = self.opt_energy(bond_length)
            opt_list.append(opt[0][0])
        return opt_list

def get_expect(hamiltonian, psi):
    expect = np.conjugate(psi.T).dot(hamiltonian.dot(psi))
    return expect

def get_BK_ham(distance, basis="sto-3g", multiplicity=1, charge=1):
    geometry = [["He",[0,0,0]],["H", [0,0,distance]]]
    description = str(distance)
    molecule = MolecularData(geometry, basis, multiplicity, charge, description)
    molecule = run_pyscf(molecule,run_scf=1,run_fci=1)
    bk_hamiltonian = get_sparse_operator(bravyi_kitaev(get_fermion_operator(molecule.get_molecular_hamiltonian())))
    return bk_hamiltonian





if __name__ == "__main__":

    vqe = VQE(n_qubit=4, n_point=10)
    print(vqe.get_opt_energy())
