'''
Tests for nrg mapping procedures.
'''

from numpy import *
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose
from matplotlib.pyplot import *
import time,pdb

from hybri_sc import *
from discretization import *
from utils import *
from ticklib import *

def test_get_wlist():
    '''test for get_wlist'''
    D=[-1,2]
    Gap=[-0.1,0.4]
    w0=1e-8
    Nw=500
    for mesh_type in ['log','sclog','linear']:
        print 'Testing for %s wlist.'%mesh_type
        wlist=get_wlist(w0,Nw,mesh_type,D=D,Gap=Gap)
        assert_(all(diff(wlist)>0))
        assert_almost_equal(wlist[0],D[0])
        assert_almost_equal(wlist[-1],D[-1])
        assert_(all(wlist[wlist>0]>=Gap[1]) and all(wlist[wlist<0]<=Gap[0]))
        assert_(len(wlist)==2*Nw)
    pdb.set_trace()

def test_hybri_sc():
    '''test hybridization function for superconductor.'''
    D0=2.
    Gap=0.3
    Gamma=0.5/pi
    D=sqrt(D0**2+Gap**2)
    wlist=get_wlist(w0=1e-8,Nw=2000,mesh_type='sclog',Gap=Gap,D=D0+Gap)
    dfunc1=get_hybri(Gap,Gamma,D0=D0,mu=0.,eta=1e-10)
    dfunc2=get_hybri_wideband(Gap,Gamma,D=D,mu=0.,eta=1e-10)
    ion()
    dl1=array([dfunc1(w) for w in wlist])
    dl2=array([dfunc2(w) for w in wlist])
    plot_pauli_components(wlist,dl1)
    plot_pauli_components(wlist,dl2,method='scatter')
    ylim(-0.1,0.2)
    assert_allclose(dl1,dl2,atol=1e-5)
    pdb.set_trace()

class MapTest():
    '''test hybridization function for superconductor.'''
    def __init__(self):
        self.D0=2.
        self.Gap=0.3
        self.Gamma=0.5/pi
        self.D=sqrt(self.D0**2+self.Gap**2)
        self.wlist=get_wlist(w0=1e-8,Nw=2000,mesh_type='sclog',Gap=self.Gap,D=sqrt(self.D0**2+self.Gap**2))
        dfunc=get_hybri_wideband(self.Gap,self.Gamma,D=self.D,mu=0.,eta=1e-10)
        self.rhofunc=lambda w:dfunc(w)/pi
        self.rholist=array([self.rhofunc(w) for w in self.wlist])

    def test_tick(self):
        '''test for ticks.'''
        tick_types=['log','sclog','adaptive','linear','adaptive_linear','ed']
        Lambda=2.0
        N=12
        wlist=self.wlist
        pmask=wlist>0
        ion()
        rholist=self.rholist[pmask]
        rholist=sqrt((rholist*swapaxes(rholist,1,2)).sum(axis=(1,2)))
        colors=['r','g','b','k','y','c']
        for i,tick_type in enumerate(tick_types):
            offset_y=i
            ticker=get_ticker(tick_type,D=self.D,N=N,Lambda=Lambda,Gap=self.Gap,wlist=wlist[pmask],rholist=rholist)
            plt=scatter(ticker(arange(2,2+N+1)),offset_y*ones(N+1),edgecolor='none',color=colors[i],label=tick_type)
        legend(loc=3)
        pdb.set_trace()

    def test_map2b(self):
        '''test for 2-band mapping'''
        funcs=quick_map(rhofunc=self.rhofunc,wlist=self.wlist,Gap=self.Gap,tick_type='adaptive',Lambda=2.,Nx=200000,branches=[0,1],tick_params=None,autofix=1e-5)
        check_disc_pauli(rhofunc,Efunc,Tfunc,scalefunc,sgn,N,D,Gap=Gap,Nx=1000,Nw=200,smearing=0.02,save_token='')

#test_get_wlist()
#test_hybri_sc()
MapTest().test_tick()
