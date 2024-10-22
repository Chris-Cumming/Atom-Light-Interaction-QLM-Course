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

print("Exercise 1 Problems:")

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
spin_vector = 1/2 * pauli_vector #Note hbar is missing, i.e hbar = 1, so all results are in terms of hbar
#print(spin_vector)

#Consider 2 spin 1/2 system
#Note capital S means total of the 2 spins, whereas lower case s refers to individual spins
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


print("The 4x4 matrix representation for S^2 in the uncoupled basis, in units of hbar squared, is:")
print(Spin_Vector_Squared)

print("The 4x4 matrix representation for Sz in the uncoupled basis, in units of hbar, is:")
print(Spin_Vector_Z)

print("")

'''Question 2'''

print("QUESTION 2")

print("")

#The operator we wish to diagonalise in order to find the coupled basis eigenstates in terms 
#of the uncoupled basis eigenstates is S^2 + Sz, this is because we want simultaneous eigenstates

total_Spin_matrix = Spin_Vector_Squared + Spin_Vector_Z
#print(total_Spin_matrix)

#Now determine eigenstates of matrix

w, vr = linalg.eig(total_Spin_matrix)
eigenvalues = w
coupled_basis_eigenstates_kets = vr

#Individual coupled basis eigenstats kets is given by columns
print("The coupled basis eigenstate kets in terms of the uncoupled basis eigenstates are given by the columns of:")
print(coupled_basis_eigenstates_kets)

#Individual coupled basis eigenstates bras is given by columns 
coupled_basis_eigenstates_bra = coupled_basis_eigenstates_kets.conj().T
print("The coupled basis eigenstate bras in terms of the uncoupled basis eigenstates are give by the rows of:")
print(coupled_basis_eigenstates_bra)

'''

#Compute eigenvalues of the eigenstates to check validity
total_Spin_system = coupled_basis_eigenstates_bra @ Spin_Vector_Squared @ coupled_basis_eigenstates_kets
print("The total spin for each of the coupled basis eigenstates is:")
print(np.diag(total_Spin_system)) #3 states with total = 2 and the other total = 0 (close enough due to computation finite)

Spin_projection_z_number = coupled_basis_eigenstates_bra @ Spin_Vector_Z @ coupled_basis_eigenstates_kets
print("The spin component along z for each of the coupled basis eigenstates is:")
print(np.diag(Spin_projection_z_number)) #Believe this to be correct

'''

dim_coupled_basis_eigenstates_kets = np.size(coupled_basis_eigenstates_kets, axis = 0)
#print(dim_coupled_basis_eigenstates_kets)

for i in range(dim_coupled_basis_eigenstates_kets):
    print("The ket of eigenstate", i + 1, "is, as a column vector:")
    print(coupled_basis_eigenstates_kets[:, i])
    print("The bra of eigenstate", i + 1, "is, as a column vector:")
    print(coupled_basis_eigenstates_kets[:, i].conj().T)


#In theory, we can find the Clebsch Coefficients by applying the bra of the uncoupled eigenstates onto the ket of coupled eigenstates
#First identify each of the original un-normalised uncoupled basis eigenstates
#Can't find normalisation factor since each individual eigenstates has a unique normalisation constant.
#Don't have enough constraints to determine each individually!

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

#The diagonals of these operators should be the observables for the individual spin system in question

#Determine the 4x4 representations of the spin operators for spin A
#Required tensor product between Hilbert space A and B


s_Ax = linalg.kron(spin_vector[0], Identity2)
#print(s_Ax)

s_Ay = linalg.kron(spin_vector[1], Identity2)
#print(s_Ay)

s_Az = linalg.kron(spin_vector[2], Identity2)
print("The matrix representation of the operator S_z for system A, in units of hbar, is:")
print(s_Az)

s_A_squared = s_Ax @ s_Ax + s_Ay @ s_Ay + s_Az @ s_Az
print("The matrix representation of the operator S^2 for system A, in units of hbar squared is:")
print(s_A_squared)

#Determine the 4x4 representations of the spin operators for spin B
#Required tensor product between Hilbert space A and B


s_Bx = linalg.kron(Identity2, spin_vector[0])
#print(s_Bx)

s_By = linalg.kron(Identity2, spin_vector[1])
#print(s_By)

s_Bz = linalg.kron(Identity2, spin_vector[2])
print("The matrix representation of the operator S_z for system B, in units of hbar, is:")
print(s_Bz)

s_B_squared = s_Bx @ s_Bx + s_By @ s_By + s_Bz @ s_Bz
print("The matrix representation of the operator S^2 for system B, in units of hbar squared, is:")
print(s_B_squared)


'''Specific problems from handout'''

print("")

print("Specific problems from handout:")

print("")

'''Question 1'''

print("")

print("Question 1")

print("")

print("Notes made in Overleaf")

print("")

'''Question 2'''

print("Question 2")

print("")

#First define the J+ raising (creation) operator

def J_plus(j):
    #The operator acts on m_j states of which there are 2j + 1
    dim = np.rint(2.0*j+1).astype(int) # round 2j+1 to integer, .astype(int) is required otherwise returned as float
    jp = np.zeros((dim,dim)) #Matrix form of J+, currently empty
    for mj in range(dim-1):
        jp[mj,mj+1] = np.sqrt(j*(j+1)-(j-mj)*(j-mj-1))
    return jp

def J_minus(J_plus):
    J_minus = np.transpose(J_plus)
    return J_minus

    
input_j = 1/2 #For spin-j system
output_j_plus = J_plus(input_j)
print("The matrix representation for the J+ operator for j = ", input_j, "is given by:")
print(output_j_plus)
#The J- lowering (annihilation) operator is simply the transpose of the creation operator
output_j_minus = J_minus(output_j_plus)
print("The matrix representation for the J- operator for j = ", input_j, "is given by:")
print(output_j_minus)

'''Question 3'''

print("")

print("Question 3")

print("")

print("Done on paper as requested")















