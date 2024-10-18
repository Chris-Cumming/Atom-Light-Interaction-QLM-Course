# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 13:58:24 2024

@author: xxvh83
"""

import numpy as np
from scipy import linalg
from scipy import constants
import matplotlib.pyplot as plt

'''Exercise 1 of Lecture Notes Problems'''

'''Question 1'''

print("")

print("QUESTION 1")

print("")

#Create identity and Pauli matrices

Identity2 = np.array([[1, 0], [0, 1]])
#print(Identity2)

pauli_x = np.array([[0, 1], [1, 0]])
#print(pauli_x)
pauli_y = np.array([[0, -1j], [1j, 0]])
#print(pauli_y)
pauli_z = np.array([[1, 0], [0, -1]])
#print(pauli_z)

pauli_vector = np.array([pauli_x, pauli_y, pauli_z])
#print(pauli_vector)
spin_vector = 1/2 * pauli_vector #Note hbar is missing so all results are in terms of hbar
#print(spin_vector)

#Consider 2 spin 1/2 system
#Want the operators for the combined 2 spin system, i.e 4x4 matrices instead of 2x2
#Can form 4x4 matrices using tensor product idea acting on hilbert spaces
#S_x = s_x1 + s_x2, which has unwritten tensor products with the 2x2 identity matrix

Spin_Vector_X = linalg.kron(spin_vector[0], Identity2) + linalg.kron(Identity2, spin_vector[0])
#print(Spin_Vector_X)

Spin_Vector_Y = linalg.kron(spin_vector[1], Identity2) + linalg.kron(Identity2, spin_vector[1])
#print(Spin_Vector_Y)

Spin_Vector_Z = linalg.kron(spin_vector[2], Identity2) + linalg.kron(Identity2, spin_vector[2])
#print(Spin_Vector_Z)

Spin_Vector = np.array([Spin_Vector_X, Spin_Vector_Y, Spin_Vector_Z])
#print(Spin_Vector)

Spin_Vector_Squared = Spin_Vector[0]@Spin_Vector[0] + Spin_Vector[1]@Spin_Vector[1] + Spin_Vector[2]@Spin_Vector[2] #Has units of hbar squared


print("The 4x4 matrix representation for S^2 in the uncoupled basis is:")
print(Spin_Vector_Squared)

print("The 4x4 matrix representation for Sz in the uncoupled basis is:")
print(Spin_Vector_Z)

print("")

'''Question 2'''

print("QUESTION 2")

print("")

#The operator we wish to diagonalise in order to find the coupled basis eigenstates in terms 
#of the uncoupled basis eigenstates is S^2 + Sz

total_spin_matrix = Spin_Vector_Squared + Spin_Vector_Z
#print(total_spin)

#Now determine eigenstates of matrix

w, vr = linalg.eig(total_spin_matrix)
eigenvalues = w
coupled_basis_eigenstates_kets = vr

#print("The eigenvalues of this matrix are:")
#print(eigenvalues)

#Individual coupled basis eigenstats kets is given by columns
print("The coupled basis eigenstate kets in terms of the uncoupled basis eigenstates are:")
print(coupled_basis_eigenstates_kets)

#Individual coupled basis eigenstates bras is given by columns 
coupled_basis_eigenstates_bra = coupled_basis_eigenstates_kets.conj().T
print("The coupled basis eigenstate bras in terms of the uncoupled basis eigenstates are:")
print(coupled_basis_eigenstates_bra)

#Compute eigenvalues of the eigenstates to check validity
#Need better variable names
total_spin_number = coupled_basis_eigenstates_bra @ Spin_Vector_Squared @ coupled_basis_eigenstates_kets
print("The total spin of each of the coupled basis eigenstates is:")
print(np.diag(total_spin_number)) #3 states with total = 2 and the other total = 0 (close enough due to computation finite)

spin_projection_z_number = coupled_basis_eigenstates_bra @ Spin_Vector_Z @ coupled_basis_eigenstates_kets
print("The spin component along z for each of the coupled basis eigenstates is:")
print(np.diag(total_spin_number)) #Believe this to be correct

dim_coupled_basis_eigenstates_kets = np.size(coupled_basis_eigenstates_kets, axis = 0)
#print(dim_coupled_basis_eigenstates_kets)

for i in range(dim_coupled_basis_eigenstates_kets):
    print("The ket of eigenstate", i + 1, "is:")
    print(coupled_basis_eigenstates_kets[:, i])
    print("The bra of eigenstate", i + 1, "is:")
    print(coupled_basis_eigenstates_kets[:, i].conj().T)


#We can find the Clebsch Coefficients by applying the bra of the uncoupled eigenstates onto the ket of coupled eigenstates
#First identify each of the original unnormalised uncoupled basis eigenstates
#Can't find normalisation factor since each individual eigenstates has a unique normalisation constant.
#Don't have enough constraints to determine each individually.

'''
uncoupled_basis_eigenstates_kets = np.identity(4)
print("The uncoupled basis eigenstate kets are:")
print(uncoupled_basis_eigenstates_kets)

uncoupled_basis_eigenstates_bras = uncoupled_basis_eigenstates_kets
print("The uncoupled basis eigenstate bras are:")
print(uncoupled_basis_eigenstates_bras)

'''


'''Question 3'''

print("")

print("QUESTION 3")

print("")


s_Ax = linalg.kron(spin_vector[0], Identity2)
print(s_Ax)

s_Ay = linalg.kron(spin_vector[1], Identity2)
print(s_Ay)

s_Az = linalg.kron(spin_vector[2], Identity2)
print(s_Az)

s_A_squared = s_Ax @ s_Ax + s_Ay @ s_Ay + s_Az @ s_Az
print(s_A_squared)


s_Bx = linalg.kron(Identity2, spin_vector[0])
print(s_Bx)

s_By = linalg.kron(Identity2, spin_vector[1])
print(s_By)

s_Bz = linalg.kron(Identity2, spin_vector[2])
print(s_Bz)

















