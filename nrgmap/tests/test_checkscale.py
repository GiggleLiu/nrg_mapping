'''
Tests for nrg mapping procedures.
'''

from numpy import *
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose
from matplotlib.pyplot import *
from scipy import sparse as sps
from scipy.linalg import qr,eigvalsh,norm
import time,pdb,sys

from ..utils import *
from ..discretization import get_wlist,quick_map
from ..chain import *
from ..hybri_sc import *
from ..chainmapper import *

def get_chain(z=0.5,method='qr'):
    '''
    run this sample, visual check is quite slow!
    '''
    #generate the hybridization function.
    nband=4
    Gamma=0.5/pi
    Lambda=1.8
    
    D=[-1.,0.5]             #the energy window.
    wlist=get_wlist(w0=1e-12,Nw=10000,mesh_type='log',Gap=0,D=D)
    #rhofunc=lambda w:identity(4)+0.3*w*Gmat[0]+0.3*w**2*Gmat[2] #the case with degeneracy
    rhofunc=lambda w:identity(4)+0.3*w*Gmat[0]+0.3*w**2*Gmat[2]+0.1*kron(sz,sz)     #the case without degeneracy.

    #create the discretized model
    N=33      #the chain length
    tick_type='adaptive'
    discmodel=quick_map(rhofunc=rhofunc,wlist=wlist,N=N,z=z,Nx=200000,tick_params={'tick_type':tick_type,'Lambda':Lambda},autofix=1e-5)[1]

    #map to a chain
    chains=map2chain(discmodel,nsite=2*N,normalize_method=method)
    return chains[0]

def get_chain_sc(z=0.5,method='qr'):
    '''
    run this sample, visual check is quite slow!
    '''
    #generate the hybridization function.
    nband=2
    Gamma=0.5/pi
    Lambda=1.6
    D0=2.
    Gap=0.3
    D=sqrt(D0**2+Gap**2)
    wlist=get_wlist(w0=1e-12,Nw=10000,mesh_type='sclog',Gap=Gap,D=D)
    rhofunc=get_hybri_skew(Gap,Gamma,D=D,eta=1e-12,skew=0.3)
    rholist=array([rhofunc(w) for w in wlist])

    #create the discretized model
    N=23      #the chain length
    tick_type='adaptive'
    discmodel=quick_map(rhofunc=rhofunc,wlist=wlist,N=N,z=z,Nx=200000,tick_params={'tick_type':tick_type,'Lambda':Lambda},autofix=1e-5)[1]
    #map to a chain
    chains=map2chain(discmodel,normalize_method=method)
    return chains[0]

def test_checkscale():
    print('Test Scaling of random 4 band model.')
    ion()
    methods=['qr','mpqr','sqrtm']
    for method in methods:
        chain1=get_chain(z=0.5,method=method)
        show_scaling(chain1)
    legend(methods)
    pdb.set_trace()

def test_checkscale_sc():
    print('Test Scaling of chain, superconducting, the offdiagonal part do not obey scaling.')
    ion()
    methods=['qr','mpqr','sqrtm']
    for method in methods:
        chain1=get_chain_sc(z=0.5,method=method)
        show_scaling(chain1)
    legend(methods)
    pdb.set_trace()

if __name__=='__main__':
    test_checkscale()
    test_checkscale_sc()
