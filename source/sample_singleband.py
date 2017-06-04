'''
Sample: Speudogap model with r = 1
'''
from numpy import *
from matplotlib.pyplot import *
import time,pdb

from discretization import quick_map,check_disc
from utils import get_wlist
from chainmapper import map2chain,check_spec


def run():
    '''
    run this sample, visual check is quite slow!
    '''
    #generate the hybridization function.
    nband=1
    Gamma=0.5/pi
    Lambda=1.8
    D=[-1,1.5]
    rhofunc=lambda w:Gamma*abs(w)
    wlist=get_wlist(w0=1e-12,Nw=10000,mesh_type='log',Gap=0,D=D)
    rholist=rhofunc(wlist)

    #create the discretized model
    N=33      #the chain length
    nz=10      #the number of twisting parameter z
    z=linspace(0.5/nz,1-0.5/nz,nz)
    tick_type='log'

    print '''Start mapping the hybridization function into a discrete set of baths.
%s sites for each(pos/neg) branch
%s z numbers
tick type -> %s
Lambda    -> %s
'''%(N,nz,tick_type,Lambda)
    discmodel=quick_map(rhofunc=rhofunc,wlist=wlist,N=N,z=z,Nx=200000,tick_params={'tick_type':tick_type,'Lambda':Lambda},autofix=1e-5)[1]

    #map to a chain
    print 'Start mapping the discrete model to a chain.'
    chains=map2chain(discmodel)
    print 'Done'

    plot_wlist=wlist[::20]
    docheck=raw_input('Check whether this star model recover the hybridization function?(y/n):')=='y'
    if docheck:
        ion()
        check_disc(rhofunc=rhofunc,wlist=plot_wlist,discmodel=discmodel,smearing=0.5)
        print 'Press `c` to continue.'
        pdb.set_trace()

    docheck=raw_input('Check whether this chain recover the hybridization function?(y/n):')=='y'
    if docheck:
        ion();cla()
        check_spec(rhofunc=rhofunc,chains=chains,wlist=plot_wlist,smearing=0.5,mode='pauli' if nband==2 else 'eval')
        print 'Press `c` to continue.'
        pdb.set_trace()

    dosave=raw_input('Save the chain datas?(y/n):')=='y'
    if dosave:
        for iz,chain in zip(z,chains):
            chain.save('singleband_%s'%iz)

if __name__=='__main__':
    run()
