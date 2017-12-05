'''
Tests for nrg mapping procedures.
'''

from numpy import *
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose
from matplotlib.pyplot import *
from scipy import sparse as sps
from scipy.linalg import qr,eigvalsh,norm
import time,pdb,sys

from ..hybri_sc import *
from ..discretization import *
from ..utils import *
from ..ticklib import *
from ..chainmapper import *
from ..discmodel import *
from ..chain import *
from ..tridiagonalize import tridiagonalize_qr,tridiagonalize,tridiagonalize_sqrtm,construct_tridmat

def test_get_wlist():
    '''test for get_wlist'''
    D=[-1,2]
    Gap=[-0.1,0.4]
    w0=1e-8
    Nw=500
    for mesh_type in ['log','sclog','linear']:
        print('Testing for %s wlist.'%mesh_type)
        wlist=get_wlist(w0,Nw,mesh_type,D=D,Gap=Gap)
        assert_(all(diff(wlist)>0))
        assert_almost_equal(wlist[0],D[0])
        assert_almost_equal(wlist[-1],D[-1])
        assert_(all(wlist[wlist>0]>=Gap[1]) and all(wlist[wlist<0]<=Gap[0]))
        assert_(len(wlist)==Nw)

def test_hybri_sc():
    '''test hybridization function for superconductor.'''
    D0=2.
    Gap=0.3
    Gamma=0.5/pi
    skew=0.3
    D=sqrt(D0**2+Gap**2)
    wlist=get_wlist(w0=1e-8,Nw=2000,mesh_type='sclog',Gap=Gap,D=D0+Gap)
    dfunc1=get_hybri(Gap,Gamma,D0=D0,mu=0.,eta=1e-10)
    dfunc2=get_hybri_wideband(Gap,Gamma,D=D,mu=0.,eta=1e-10)
    dfunc3=get_hybri_skew(Gap,Gamma,D=D,eta=1e-10,skew=skew)
    dfuncs=[dfunc1,dfunc2,dfunc3]
    ion()
    fig=figure(figsize=(4,9))
    dls=[array([df(w) for w in wlist]) for df in dfuncs]
    titles=['normal','wideband','skew']
    for i in range(3):
        subplot('31%s'%(i+1))
        title(titles[i])
        plot_pauli_components(wlist,dls[i],lw=2)
        ylim(-0.1,0.2)
    assert_allclose(dls[0],dls[1],atol=1e-5)
    tight_layout()

class MapTest():
    '''test hybridization function for superconductor.'''
    def __init__(self,nband):
        self.nband=nband
        self.Gamma=0.5/pi
        self.Lambda=1.7
        if nband==2:
            D0=2.
            self.Gap=0.3
            D=sqrt(D0**2+self.Gap**2)
            self.D=[-D,D]
            self.wlist=get_wlist(w0=1e-8,Nw=10000,mesh_type='sclog',Gap=self.Gap,D=D)
            self.rhofunc=get_hybri_skew(self.Gap,self.Gamma,D=D,eta=1e-15,skew=0.3)
        elif nband==1:
            self.D=[-1,1.5]
            self.Gap=0
            self.rhofunc=lambda w:self.Gamma*abs(w)
            self.wlist=get_wlist(w0=1e-8,Nw=10000,mesh_type='log',Gap=self.Gap,D=self.D)
        elif nband==4:
            self.D=[-1.,0.5]             #the energy window.
            self.Gap=0
            self.wlist=get_wlist(w0=1e-8,Nw=10000,mesh_type='log',Gap=self.Gap,D=self.D)
            #self.rhofunc=lambda w:identity(4)+0.3*w*Gmat[0]+0.3*w**2*Gmat[2]+0.1*kron(sz,sz)     #the case without degeneracy.
            self.rhofunc=lambda w:identity(4)+0.3*w*Gmat[0]+0.3*w**2*Gmat[2] #the case with degeneracy
        self.rholist=array([self.rhofunc(w) for w in self.wlist])

        #create the model
        self.N=25
        nz=5
        self.z=linspace(0.5/nz,1-0.5/nz,nz)
        self.discmodel=quick_map(rhofunc=self.rhofunc,wlist=self.wlist,N=self.N,z=self.z,Nx=200000,tick_params={'tick_type':'adaptive','Gap':self.Gap,'Lambda':self.Lambda},autofix=1e-5)[1]
        assert_(self.discmodel.N_pos==self.N and self.discmodel.N_neg==self.N and self.discmodel.N==2*self.N)
        if nband==1:
            assert_(self.discmodel.Elist_pos.shape==(self.N,nz))
        else:
            assert_(self.discmodel.Elist_pos.shape==(self.N,nz,nband,nband))

        #map to a chain
        self.chains=map2chain(self.discmodel)

    def test_saveload(self):
        '''save and load data.'''
        for iz,chain in zip(self.z,self.chains):
            file_prefix='test_%s'%iz
            self.discmodel.save(file_prefix)
            model=load_discmodel(file_prefix)
            assert_allclose(model.Elist,self.discmodel.Elist)
            assert_allclose(model.Tlist,self.discmodel.Tlist)
            assert_allclose(model.z,self.discmodel.z)

            chain.save(file_prefix)
            chain2=load_chain(file_prefix)
            assert_allclose(chain2.elist,chain.elist)
            assert_allclose(chain2.tlist,chain.tlist)

    @dec.slow
    def test_map(self):
        '''test for mapping'''
        plot_wlist=self.wlist[::50]
        if self.nband==2:
            check_disc(rhofunc=self.rhofunc,wlist=plot_wlist,discmodel=self.discmodel,smearing=1,mode='pauli')
            print('***The superconducting model needs some special gradients to cope the smearing factor here,\
                    \nwhich is not included for general purpose,\
                    \nso, don\'t be disappointed by the poor match here, they are artifacts.***')
            ylim(-0.1,0.2)
        elif self.nband==1 or self.nband==4:
            check_disc(rhofunc=self.rhofunc,wlist=plot_wlist,discmodel=self.discmodel,smearing=0.2 if self.nband==1 else 0.4)

    @dec.slow
    def test_chain(self):
        '''test for tridiagonalization.'''
        plot_wlist=self.wlist[::20]
        chains=self.chains
        assert_(chains[0].nsite==self.N)
        nband=self.nband
        if nband==2:
            smearing=1
        elif nband==4:
            smearing=0.4
        else:
            smearing=0.2
        check_spec(rhofunc=self.rhofunc,chains=chains,wlist=plot_wlist,smearing=smearing,mode='pauli' if self.nband==2 else 'eval')
        if self.nband==2:
            ylim(-0.1,0.2)


def test_all():
    ion()
    test_get_wlist()
    test_hybri_sc()
    for i in [1,2,4]:
        t0=time.time()
        ti=MapTest(i)
        t1=time.time()
        print('Elapse, %s'%(t1-t0))
        ti.test_saveload()
        ti.test_map()
        ti.test_chain()

if __name__=='__main__':
    test_all()
