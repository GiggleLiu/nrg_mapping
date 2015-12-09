#!/usr/bin/python
'''
Setting File For NRG Program.
'''

#high precision used in tridiagonalization
PRECISION=4000

#the default number of w-samples used in the program for interpolating rho(w)
NW=100000

#the number of x-samples used in the program for integration over rho(epsilon(x))
NX=1000000

############## Check for discretization ###############
#number of xs to make discretized model into hybridization functions, the more the smoother.
NX_CHECK_DISC=2000
#number of ws for display
NW_CHECK_DISC=200
#smearing factor for checking, the larger the smoother.
SMEARING_CHECK_DISC=0.015

############## Check for chain mapping ###############
#number of ws for display
NW_CHECK_CHAIN=300
#smearing factor for checking, the larger the smoother.
SMEARING_CHECK_CHAIN=1.3

############## Other setting #################
DATA_FOLDER='./'
