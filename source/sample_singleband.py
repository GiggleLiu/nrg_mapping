'''
Sample: Speudogap model with r = 1
'''
from numpy import *
from matplotlib.pyplot import *
import time,pdb

from discretization import quick_map,get_wlist,check_disc
from chainmapper import map2chain,check_spec
from nrg_setting import PRECISION


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
    wlist=get_wlist(w0=1e-8,Nw=10000,mesh_type='log',Gap=0,D=D)
    rholist=rhofunc(wlist)

    #create the discretized model
    N=33      #the chain length
    nz=50      #the number of twisting parameter z
    z=linspace(0.5/nz,1-0.5/nz,nz)
    tick_type='adaptive'

    print '''Start mapping the hybridization function into a discrete set of baths.
%s sites for each(pos/neg) branch
%s z numbers
tick type -> %s
Lambda    -> %s
'''%(N,nz,tick_type,Lambda)
    discmodel=quick_map(rhofunc=rhofunc,wlist=wlist,N=N,z=z,Nx=200000,tick_params={'tick_type':tick_type,'Lambda':Lambda},autofix=1e-5)[1]

    #map to a chain
    print 'Start mapping the discrete model to a chain, using precision %s-bit.'%PRECISION
    chain=map2chain(discmodel,prec=PRECISION)
    print 'Done'

    plot_wlist=wlist[::30]
    docheck=raw_input('Check whether this star model recover the hybridization function?(y/n):')=='y'
    if docheck:
        ion()
        check_disc(rhofunc=rhofunc,wlist=plot_wlist,discmodel=discmodel,smearing=0.2)
        print 'Press `c` to continue.'
        pdb.set_trace()

    docheck=raw_input('Check whether this chain recover the hybridization function?(y/n):')=='y'
    if docheck:
        ion();cla()
        check_spec(rhofunc=rhofunc,chain=chain,wlist=plot_wlist,smearing=1.,mode='pauli' if nband==2 else 'eval')
        print 'Press `c` to continue.'
        pdb.set_trace()

    dosave=raw_input('Save the chain datas?(y/n):')=='y'
    if docheck:
        chain.save('singleband')

if __name__=='__main__':
    run()
