from numpy import *
import matplotlib.pyplot as plt
import sys,pdb
sys.path.insert(0,'../')

from discretization import check_disc
from discmodel import load_discmodel
from chain import load_chain
from chainmapper import check_spec
from utils import get_wlist

from make_samples import rhofunc_singleband,rhofunc_4band,rhofunc_sc

def check(file_prefix):
    discmodel=load_discmodel(file_prefix)
    chains=[]
    for z in discmodel.z:
        chains.append(load_chain(file_prefix+'_%s'%z))
    if chains[0].nband==2:
        rhofunc=rhofunc_sc
        D=sqrt(2**2+0.3**2)
        plot_wlist=get_wlist(w0=1e-12,Nw=100,mesh_type='sclog',Gap=0.3,D=[-D,D])
    elif chains[0].nband==1:
        rhofunc=rhofunc_singleband
        plot_wlist=get_wlist(w0=1e-12,Nw=100,mesh_type='log',Gap=0,D=[-1,1.5])
    else:
        rhofunc=rhofunc_4band
        plot_wlist=get_wlist(w0=1e-12,Nw=100,mesh_type='log',Gap=0,D=[-1,0.5])
    plt.ion()
    print 'Checking Star model.'
    check_disc(rhofunc=rhofunc,wlist=plot_wlist,discmodel=discmodel,smearing=0.7)
    if chains[0].nband==2:plt.ylim(0,0.5)
    print 'Press `c` to continue.'
    pdb.set_trace()

    print 'Checking Wilson Chain model.'
    plt.cla()
    check_spec(rhofunc=rhofunc,chains=chains,wlist=plot_wlist,smearing=0.7,mode='eval')
    if chains[0].nband==2:plt.ylim(0,0.5)
    print 'Press `c` to continue.'
    pdb.set_trace()

if __name__=='__main__':
    check('../data/sc')
    check('../data/4band')
    check('../data/singleband')
