'''
Sample: Speudogap model with r = 1
'''
from builtins import input
from numpy import *
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import time,pdb

from nrgmap.hybri_sc import get_hybri_skew
from nrgmap.discretization import quick_map,check_disc
from nrgmap.chainmapper import map2chain,check_spec
from nrgmap.utils import Gmat,sz,get_wlist

def run():
    '''
    run this sample, visual check is quite slow!
    '''
    #generate the hybridization function.
    nband=4
    Gamma=0.5/pi
    Lambda=1.8
    
    D=[-1.,0.5]             #the energy window.
    wlist=get_wlist(w0=1e-12,Nw=10000,mesh_type='log',Gap=0,D=D)
    rhofunc=lambda w:identity(4)+0.3*w*Gmat[0]+0.3*w**2*Gmat[2]+0.1*kron(sz,sz)     #the case without degeneracy.

    #create the discretized model
    N=33      #the chain length
    nz=50      #the number of twisting parameter z
    z=linspace(0.5/nz,1-0.5/nz,nz)
    tick_type='adaptive'

    print('''Start mapping the hybridization function into a discrete set of baths.
%s sites for each(pos/neg) branch
%s z numbers
tick type -> %s
Lambda    -> %s
'''%(N,nz,tick_type,Lambda))
    discmodel=quick_map(rhofunc=rhofunc,wlist=wlist,N=N,z=z,Nx=200000,tick_params={'tick_type':tick_type,'Lambda':Lambda},autofix=1e-5)[1]

    #map to a chain
    print('Start mapping the discrete model to a chain.')
    chains=map2chain(discmodel,nsite=2*N,normalize_method='qr')
    print('Done')

    plot_wlist=wlist[::30]
    docheck=input('Check whether this star model recover the hybridization function?(y/n):')=='y'
    if docheck:
        plt.ion()
        check_disc(rhofunc=rhofunc,wlist=plot_wlist,discmodel=discmodel,smearing=0.7)
        print('Press `c` to continue.')
        pdb.set_trace()

    docheck=input('Check whether this chain recover the hybridization function?(y/n):')=='y'
    if docheck:
        plt.ion();plt.cla()
        check_spec(rhofunc=rhofunc,chains=chains,wlist=plot_wlist,smearing=0.7,mode='eval')
        print('Press `c` to continue.')
        pdb.set_trace()

    dosave=input('Save the chain datas?(y/n):')=='y'
    if dosave:
        for iz,chain in zip(z,chains):
            chain.save('data/4band_%s'%iz)

if __name__=='__main__':
    run()
