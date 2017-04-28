from numpy import *
import sys
sys.path.insert(0,'../')

from utils import Gmat,sz,get_wlist
from hybri_sc import get_hybri_skew

rhofunc_4band=lambda w:identity(4)+0.3*w*Gmat[0]+0.3*w**2*Gmat[2]+0.1*kron(sz,sz)     #the case without degeneracy.
rhofunc_sc=get_hybri_skew(Gap=0.3,Gamma=0.5/pi,D=sqrt(2**2+0.3**2),eta=1e-12,skew=0.3)
rhofunc_singleband=lambda w:0.5/pi*abs(w)

def make_4band_data():
    Gamma=0.5/pi
    D=[-1.,0.5]             #the energy window.
    wlist=get_wlist(w0=1e-12,Nw=10000,mesh_type='log',Gap=0,D=D)
    filename='hybridization_func_4band.complex.hybri'
    dump_rhofunc(rhofunc_4band,wlist,filename)

def make_sc_data():
    D=sqrt(2**2+0.3**2)
    wlist=get_wlist(w0=1e-12,Nw=10000,mesh_type='sclog',Gap=0.3,D=D)
    filename='hybridization_func_sc.complex.hybri'
    dump_rhofunc(rhofunc_sc,wlist,filename)

def make_singleband_data():
    #generate the hybridization function.
    D=[-1,1.5]
    wlist=get_wlist(w0=1e-12,Nw=10000,mesh_type='log',Gap=0,D=D)

    filename='hybridization_func_singleband.hybri'
    dump_rhofunc(rhofunc_singleband,wlist,filename)
 
def dump_rhofunc(rhofunc,wlist,filename):
    data=concatenate([wlist[:,newaxis],array([rhofunc(w) for w in wlist]).reshape([len(wlist),-1]).view('float64')],axis=1)
    savetxt(filename,data)

if __name__=='__main__':
    make_4band_data()
    make_singleband_data()
    make_sc_data()
