#!/usr/bin/python
'''
A test for versatile mapping scheme!
'''
import pdb,time,sys
from discretization import *
from chainmapper import ChainMapper,save_chain,load_chain
from hybri_sc import get_hybri_wideband,get_hybri
from utils import Gmat
from nrg_setting import *

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

def test_adapt(check_mapping=True,check_trid=True,which='sc',nz=50):
    '''
    Test for mapping 2x2 rho function to discretized model.

    check_mapping:
        check and plot the hybridization after discretization if True.
    check_trid:
        check and plot the hybridization after tridiagonalization if True.
    nz:
        number of z for averaging.
    '''
    #select test cases.
    if which=='4band':
        D=[-1.,0.5]             #the energy window.
        Gap=0                   #the gap range, 0. for gapless, or specifing the gap range by [gapedge_neg, gapedge_pos].
        rhofunc=lambda w:identity(4)+0.3*w*Gmat[0]+0.3*w**2*Gmat[2]+0.1*kron(sz,sz)     #hybridization function
        token='TEST4'           #for storing and fetching datas.
        ymin,ymax=0,2           #the hybridization strength window.
        check_scheme='eval'     #check the eigen values.
    elif which=='sc':
        D=1.
        Gap=0.1
        rhofunc0=get_hybri_wideband(Gamma=0.5/pi,D=D,Gap=Gap)
        rhofunc=lambda w:rhofunc0(w)+(0.03*(w-2*Gap) if w>2*Gap else 0)*sz
        token='TESTSC'
        ymin,ymax=-0.2,0.2
        check_scheme='pauli'    #check for pauli components is always recommended for 2-band hybridization function.
    elif which=='pseudogap':
        D=1.
        Gap=0.
        one=1.
        rhofunc=lambda w:one*abs(w) if abs(w)<=D else 0.*one
        token='TEST1'
        ymin,ymax=0.,1.
        check_scheme='eval'
    else:
        raise Exception('Undefined Test Case!')
    #DiscHandler is a handler class for discretization.
    mapper=DiscHandler(token,2.,N=25,D=D,Gap=Gap)  #token,Lambda,maximum scale index and band range,Gap.

    #set the hybridization function rho(w)
    print 'Setting up hybridization function ...'
    mapper.set_rhofunc(rhofunc,Nw=NW)    #a function of rho(w), number of ws for each branch.
    print 'Done.'

    #perform mapping and get functions of epsilon(x),E(x) and T(x) for positive and negative branches.
    #epsilon(x) -> function of discretization mesh points.
    #E(x)/T(x) -> function of representative energy and hopping terms.
    funcs=mapper.quick_map(tick_type='log',Nx=NX) #tick type,number samples for integration over x.
    (ef,Ef,Tf),(ef_neg,Ef_neg,Tf_neg)=funcs

    #check for discretization.
    if check_mapping and RANK==0:
        ylim(ymin,ymax)
        if check_scheme=='eval':
            mapper.check_mapping_eval(rhofunc,Ef,Tf,ef,sgn=1,Nx=NX_CHECK_DISC,smearing=SMEARING_CHECK_DISC,Nw=NW_CHECK_DISC)
            mapper.check_mapping_eval(rhofunc,Ef_neg,Tf_neg,ef_neg,sgn=-1,Nx=NX_CHECK_DISC,smearing=SMEARING_CHECK_DISC,Nw=NW_CHECK_DISC)
        else:
            mapper.check_mapping_pauli(rhofunc,Ef,Tf,ef,sgn=1,Nx=NX_CHECK_DISC,smearing=SMEARING_CHECK_DISC,Nw=NW_CHECK_DISC)
            mapper.check_mapping_pauli(rhofunc,Ef_neg,Tf_neg,ef_neg,sgn=-1,Nx=NX_CHECK_DISC,smearing=SMEARING_CHECK_DISC,Nw=NW_CHECK_DISC)

    #get a discrete model
    #twisting parameters, here we take 50 zs for checking.
    z=linspace(1./nz/2,1-1./nz/2,nz)
    #z=1.
    #extract discrete set of models with output functions of quick_map, a DiscModel instance will be returned.
    disc_model=mapper.get_discrete_model(funcs,z=z,append=False)

    #Chain Mapper is a handler to map the DiscModel instance to a Chain model.
    cmapper=ChainMapper(prec=PRECISION)
    chain=cmapper.map(disc_model)

    #save the chain, you can get the chain afterwards by load_chain method or import it to other programs.
    #data saved:
    #   <DATA_FOLDER>/<token>.tl.dat -> representative coupling of i-th site to the previous site(including coupling with impurity site - t0), float view for complex numbers, shape is raveled to 1 column with length: nband x nband x nz x N(chain length) x 2(complex and real).
    #   <DATA_FOLDER>/<token>.el.dat -> representative energies, stored same as above.
    #   <DATA_FOLDER>/<token>.info.dat -> shape information, (Chain length,nz,nband,nband)
    save_chain(token,chain)

    #check for tridiagonalization
    if check_trid and RANK==0:
        figure()
        ylim(ymin,ymax)
        cmapper.check_spec(chain,mapper,rhofunc,mode=check_scheme,Nw=NW_CHECK_CHAIN,smearing=SMEARING_CHECK_CHAIN)
    print 'TEST OVER! PRESS `c` TO END PROGRAM.'
    pdb.set_trace()

if __name__=='__main__':
    cases=['pseudogap','sc','4band']
    unvalid=True
    hintstr='''
Welcome to versatile mapping scheme for NRG, Please select a TEST CASE(Ctrl+D to quit):
    1. Traditional SpeudoGap Model(1 band).
    2. S-wave Superconducting Model(2 band).
    3. An artificial 4-band model.
    Choice(1/2/3):'''
    unvalid=True
    while unvalid:
        try:
            n=int(raw_input(hintstr))
        except EOFError:
            print 'Quiting'
            sys.exit()
        except:
            print 'Unvalid String, input again!'
            continue
        if n>=1 and n<=3:
            unvalid=False
        else:
            print 'Choice out of range, input again!'
    test_adapt(True,True,which=cases[n-1],nz=50)
