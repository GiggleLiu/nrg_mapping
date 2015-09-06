#!/usr/bin/python
'''
A test for versatile mapping scheme!
'''
import pdb,time
from discretization import *
from chainmapper import ChainMapper
from hybri_sc import get_hybri_wideband,get_hybri

def test_adapt2(check_mapping=True,check_trid=True):
    '''
    Test for mapping 2x2 rho function to discretized model.

    check_mapping:
        check and plot the hybridization after discretization if True.
    check_trid:
        check and plot the hybridization after tridiagonalization if True.
    '''
    #rhofunc=lambda w:identity(2)+0.5*w*sx+0.7*w**2*sy if (w<=D[1] and w>=D[0]) else zeros([2,2])
    rhofunc=get_hybri_wideband(Gamma=0.5/pi,D=1.,Gap=0.1)
    mapper=DiscHandler('TEST',2.,N=30,D=1.,Gap=0.1)  #token,Lambda,maximum scale index and band range,Gap.
    mapper.set_rhofunc(rhofunc,NW=50000)    #a function of rho(w), number of ws for interpolation(for speed).
    funcs=mapper.quick_map2(tick_type='log',NN=100000) #tick type,number samples for integration over rho.
    (sf,ef,tf),(snf,enf,tnf)=funcs
    #check for discretization.
    if check_mapping:
        ylim(-0.2,0.2)
        mapper.check_mapping2(rhofunc,ef,tf,sf,sgn=1,Nx=2000,smearing=0.01)
        mapper.check_mapping2(rhofunc,enf,tnf,snf,sgn=-1,Nx=2000,smearing=0.01)

    #get a discrete model
    z=linspace(0,0.98,50)+0.01
    #z=linspace(0,0.8,5)+0.1
    disc_model=mapper.get_discrete_model(funcs,z=z,append=False)
    cmapper=ChainMapper(prec=6000)
    chain=cmapper.map(disc_model)
    if check_trid:
        figure()
        ylim(-0.2,0.2)
        cmapper.check_spec(chain,mapper,rhofunc)
    print 'TEST OVER! PRESS `c` TO END PROGRAM.'
    pdb.set_trace()

if __name__=='__main__':
    test_adapt2(True,True)
