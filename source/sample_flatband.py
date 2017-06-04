'''
Sample: Speudogap model with r = 1
'''
from numpy import *
from matplotlib.pyplot import *
import time,pdb

from quickmap import quickmap
from utils import get_wlist
from chainmapper import check_spec


def run():
    '''
    run this sample, visual check is quite slow!
    '''
    rhofunc=lambda w:0.5/pi
    wlist=get_wlist(w0=1e-10,Nw=10000,mesh_type='log',Gap=0,D=1.)

    #create the discretized model
    chain=quickmap(wlist,rhofunc,Lambda=2.0,nsite=30,nz=1,tick_type='log')[0]

    plot_wlist=wlist
    docheck=raw_input('Check whether this chain recover the hybridization function?(y/n):')=='y'
    if docheck:
        ion();cla()
        check_spec(rhofunc=rhofunc,chains=[chain],wlist=plot_wlist,smearing=0.4)
        print 'Integrate should be %s, if being too small, oversmeared!'%(1./pi)
        print 'Press `c` to continue.'
        ylim(0,0.2)
        pdb.set_trace()

    dosave=raw_input('Save the chain datas?(y/n):')=='y'
    if dosave:
        for iz,chain in zip(z,chains):
            chain.save('flatband_%s'%iz)

if __name__=='__main__':
    run()
