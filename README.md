# Heisenberg-Spin-Chain
A python program to solve heisenberg spi half chain using Lanczos algorithm.

You just input the value of N and J, (number of lattice sites and the exchange integral), the program will give you all the relevant energy eigenvalues.

To understand the background problem and the algorithm behind the code read the pdf file.

This program was meant for part of my cousework, So it is written from very basic, not using any packages, other than numpy.

Another thing is this code works good for $N \geq 3$ sites, there may be some problem if the eigen energy is found to be zero, as I used QR algorithm  for calculating eigenvalues.

You can use relevant sections of the code after understanding it.
