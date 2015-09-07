#!/usr/bin/python
'''
A test for versatile mapping scheme!
'''
import pdb,time
from discretization import *
from chainmapper import ChainMapper,save_chain,load_chain
from hybri_sc import get_hybri_wideband,get_hybri
from utils import Gmat

#MPI setting
try:
    from mpi4py import MPI
    COMM=MPI.COMM_WORLD
    SIZE=COMM.Get_size()
    RANK=COMM.Get_rank()
except:
    COMM=None
    SIZE=1
    RANK=0


def test_adapt(check_mapping=True,check_trid=True,which='sc'):
    '''
    Test for mapping 2x2 rho function to discretized model.

    check_mapping:
        check and plot the hybridization after discretization if True.
    check_trid:
        check and plot the hybridization after tridiagonalization if True.
    '''
    if which=='4band':
        D=1.
        Gap=0.
        rhofunc=lambda w:identity(4)+0.3*w*Gmat[0]+0.3*w**2*Gmat[2]+0.1*kron(sz,sz) if abs(w)<=D else zeros([4,4])
        token='TEST4'
        ymin,ymax=0,2
    elif which=='sc':
        D=1.
        Gap=0.1
        rhofunc=get_hybri_wideband(Gamma=0.5/pi,D=D,Gap=Gap)
        token='TESTSC'
        ymin,ymax=-0.2,0.2
    else:
        raise Exception('Undefined Test Case!')
    mapper=DiscHandler(token,2.,N=25,D=D,Gap=Gap)  #token,Lambda,maximum scale index and band range,Gap.
    mapper.set_rhofunc(rhofunc,NW=100000)    #a function of rho(w), number of ws for interpolation(for speed).
    funcs=mapper.quick_map2(tick_type='log',NN=1000000) #tick type,number samples for integration over rho.
    (sf,ef,tf),(snf,enf,tnf)=funcs
    #check for discretization.
    if check_mapping and RANK==0:
        ylim(ymin,ymax)
        mapper.check_mapping_eval(rhofunc,ef,tf,sf,sgn=1,Nx=2000,smearing=0.01)
        mapper.check_mapping_eval(rhofunc,enf,tnf,snf,sgn=-1,Nx=2000,smearing=0.01)

    #get a discrete model
    z=linspace(0,0.98,50)+0.01
    #z=1.
    disc_model=mapper.get_discrete_model(funcs,z=z,append=False)
    cmapper=ChainMapper(prec=2500)
    chain=cmapper.map(disc_model)
    save_chain(token,chain)
    if check_trid and RANK==0:
        figure()
        ylim(ymin,ymax)
        cmapper.check_spec(chain,mapper,rhofunc)
    print 'TEST OVER! PRESS `c` TO END PROGRAM.'
    pdb.set_trace()

if __name__=='__main__':
    test_adapt(False,True,which='sc')
