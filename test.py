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
    #select test cases.
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
    elif which=='pseudogap':
        D=1.
        Gap=0.
        one=1.
        rhofunc=lambda w:one*abs(w) if abs(w)<=D else 0.*one
        token='TEST1'
        ymin,ymax=0.,1.
    else:
        raise Exception('Undefined Test Case!')
    #DiscHandler is a handler class for discretization.
    mapper=DiscHandler(token,2.,N=25,D=D,Gap=Gap)  #token,Lambda,maximum scale index and band range,Gap.

    #set the hybridization function rho(w)
    print 'Setting up hybridization function ...'
    mapper.set_rhofunc(rhofunc,Nw=100000)    #a function of rho(w), number of ws for each branch.
    print 'Done.'

    #perform mapping and get functions of epsilon(x),E(x) and T(x) for positive and negative branches.
    #epsilon(x) -> function of discretization mesh points.
    #E(x)/T(x) -> function of representative energy and hopping terms.
    funcs=mapper.quick_map(tick_type='log',Nx=1000000) #tick type,number samples for integration over x.
    (ef,Ef,Tf),(ef_neg,Ef_neg,Tf_neg)=funcs

    #check for discretization.
    if check_mapping and RANK==0:
        ylim(ymin,ymax)
        mapper.check_mapping_eval(rhofunc,Ef,Tf,ef,sgn=1,Nx=2000,smearing=0.01)
        mapper.check_mapping_eval(rhofunc,Ef_neg,Tf_neg,ef_neg,sgn=-1,Nx=2000,smearing=0.01)

    #get a discrete model
    #twisting parameters, here we take 50 zs for checking.
    z=linspace(0,0.98,50)+0.01
    #z=1.
    #extract discrete set of models with output functions of quick_map, a DiscModel instance will be returned.
    disc_model=mapper.get_discrete_model(funcs,z=z,append=False)

    #Chain Mapper is a handler to map the DiscModel instance to a Chain model.
    cmapper=ChainMapper(prec=2500)
    chain=cmapper.map(disc_model)
    pdb.set_trace()

    #save the chain, you can get the chain afterwards by load_chain method or import it to other programs.
    #data saved:
    #   data/<token>.tl.dat -> representative coupling of i-th site to the previous site(including coupling with impurity site - t0), float view for complex numbers, shape is raveled to 1 column with length: nband x nband x nz x N(chain length) x 2(complex and real).
    #   data/<token>.el.dat -> representative energies, stored same as above.
    #   data/<token>.info.dat -> shape information, (Chain length,nz,nband,nband)
    save_chain(token,chain)

    #check for tridiagonalization
    if check_trid and RANK==0:
        figure()
        ylim(ymin,ymax)
        cmapper.check_spec(chain,mapper,rhofunc,mode='pauli')
    print 'TEST OVER! PRESS `c` TO END PROGRAM.'
    pdb.set_trace()

if __name__=='__main__':
    #4band,sc,speudogap
    test_adapt(True,True,which='sc')
