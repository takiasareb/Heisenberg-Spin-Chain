"""
Saikat Bera
finding eigenenergies and eigenvectors of N-site Heisenberg spin half chain
"""

# open chain
print("\nProgram for calculating ground state and first excited state energy open Heisenberg spin half chain for N-sites\n")

import math
import numpy as np
from numpy import linalg as la

# program for converting decimal to binary number
# n is the decimal number, arr is its binary representation and a is the number of digits
def bin(n,a) :
    count = 0
    arr = np.zeros(a)
    while n != 0 :
        r = n%2
        arr[a - count-1] = r
        n = n//2
        count += 1
    return arr

# defining function to get eigenvalues of a matrix using QR algorithm
def qr(A):
    eigenvals = []
    n = len(A[0])
    # initializing Q
    Q = np.zeros([n,n])

    for k in range(1000):
    # Gram-Schimdt orthogonalization procedure where rows contains orthonormal vectors corresponding to column of A

        AT = np.transpose(A)       # transpose of A

        for i in range(n):
            temp = np.zeros([n])
            for j in range(i):
                temp += np.dot(AT[i], Q[j])* Q[j]
            Q[i] = AT[i] - temp
            Q[i] = Q[i]/la.norm(Q[i])

        # A = Q A QT iterative procedure
        D = np.dot(Q,np.dot(A,np.transpose(Q)))
        A = D
    for i in range(n):
        eigenvals.append(A[i][i])
    return eigenvals
    
# no of sites input
N = int(input("Enter number of lattice sites, N : ")) 
J = float(input("Enter value of exchange coupling, J : "))

# defining spin up and down state as
u = [1,0]
d = [0,1]

"""
Abbreviations :
-configuration denotes a N-site spin chain where 1 denotes up and 0 denotes down
-chain is just one configuration including up-down spin array
-chains are the array of different possible chains
-basis, corresponding to a chain, is tensor product of each spin in that chain
-bases are the array of all possible basis
"""

print("Program is running ...\n")

# initiating them with their dimensions
chain = np.zeros([N,2])
chains = np.zeros([2**N,N,2])

# findind chains with N sites
for i in range(2**N):
    config = bin(i,N)
    for j in range(N):
        if config[j] == 1:
            chain[j] = u
        else:
            chain[j] = d
    chains[i] = chain

# defining function to convert a chain state to a basis form
def chain_to_basis(chain):
    basis = chain[0]
    for k in range(1, N):
        basis = np.kron(basis, chain[k])
    return basis

# finding bases with N sites
bases = np.zeros([2**N,2**N])
for i in range(2**N):
    bases[i] = chain_to_basis(chains[i])

# Initializing Hamiltonian matrix
H = np.zeros([2**N,2**N])

# defining Sz, S+ and S- matrix taking h-cut=1
Sz = [[0.5,0],[0,-0.5]]
Sp = [[0,1],[0,0]]
Sm = [[0,0],[1,0]]

# defining functions to get resultant basis state when different component of H acts on each of the chains
def SzSz(state,j):
    result = np.zeros([N,2])
    sign = 1
    for i in range(N):
        if i == j or i == j+1:
            result[i] = abs(np.dot(Sz, state[i])/0.5)
            if result[i][0] == 0:
                sign = sign*(-1)
        else :
            result[i] = state[i]
    return sign*(0.5**2)*chain_to_basis(result)

def SpSm(state,j):
    result = np.zeros([N,2])
    for i in range(N):
        if i == j:
            result[i] = np.dot(Sp, state[i])
            result[i+1] = np.dot(Sm, state[i+1])
        elif (i!=j+1) :
            result[i] = state[i]
    return (chain_to_basis(result))

def SmSp(state,j):
    result = np.zeros([N,2])
    for i in range(N):
        if i == j:
            result[i] = np.dot(Sm, state[i])
            result[i+1] = np.dot(Sp, state[i+1])
        elif (i!=j+1) :
            result[i] = state[i]
    return (chain_to_basis(result))

# finding the resultant bases when H acted on initial bases
result_bases = np.zeros([2**N,2**N])
for i in range(2**N):
    for j in range(N-1):
        result_bases[i] += SzSz(chains[i],j) + 0.5*(SpSm(chains[i],j) + SmSp(chains[i],j))

# finding the Hamiltonian matrix
for i in range(2**N):
    for j in range(2**N):
        H[i][j] = J * np.dot(bases[i],result_bases[j])

file = open("heisenberg_open_chain_output.txt", "w")
file.write("Open chain\n")

# tridiagnalizing H using Lanczos algorithm
n = len(H[0])
v1 = np.random.random(n)           # trial vector
v1 = v1/la.norm(v1)                # normalized trial vector

m = 1                              # number of iterations
gs_energy=[0]                      # ground state energies for different number of iterations 

while m > 0:
    v0 = np.zeros(n)
    b = 0
    T = np.zeros([m,m])            # initializing the tridiagnal matrix

    for i in range(m):
        # applying formula
        w = np.dot(H,v1)
        a = np.dot(w,v1)
        v2 = w - a*v1 - b*v0
        b = la.norm(v2)
        v0 = v1
        v1 = v2/b
        T[i][i] = a
        if i < m-1:
            T[i][i+1] = T[i+1][i] = b
            
    # used qr decomposition method to find its eigenvalues
    eigenvalues = qr(T)
    sort_values = np.sort(eigenvalues)
    gs_energy.append(sort_values[0])
    
    if abs(gs_energy[m]-gs_energy[m-1]) < 1e-7:
        break
    m = m + 1
    
# calculating 1st excited energy. This will be true there the energy difference between ground state and first excited state be minimum of 1e-4
round_sort_values = sort_values.round(3)
unique_round_sort_values = np.unique(round_sort_values)
round_first_energy = unique_round_sort_values[1]
first_energy_index = np.where(round_sort_values == round_first_energy)[0][0]
first_energy = sort_values[first_energy_index]

# printing output
print("\nProgram finished.\n")
print("OUTPUT:\n")
print("Lanczos ground state enegry : ", gs_energy[-1].round(7),"\n")
print("Lanczos first excited state energy : ",first_energy.round(7),"\n")
print("The output has been stored in heisenberg_open_chain_output.txt file")

# writing Lanczos values
file.write("N = "+str(N)+" and J = "+str(J))
file.write("Lanczos ground state energy is : " + str(gs_energy[-1].round(7)) + "\n")
file.write("Lanczos first excited state energy is : " + str(first_energy.round(7)) + "\n")
file.write("Number of Lanczos iterations required is : "+str(m))
file.close()

