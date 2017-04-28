'''
Tests for tickers.
'''

from numpy import *
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose
from matplotlib.pyplot import *
from scipy import sparse as sps
from scipy.linalg import qr,eigvalsh,norm
import time,pdb,sys
sys.path.insert(0,'../')

from ticklib import *
from discretization import *


def random_function(k=5):
    '''
    Generate a random 1d function.

    Parameters:
        :k: int, the order of function, as the `fluctuation`.

    Return:
        function,
    '''
    return poly1d(random.random(k)*10-5)

def test_tick():
    '''test for ticks.'''
    tick_types=['log','sclog','adaptive','linear','adaptive_linear','ed']
    Lambda=1.7
    N=20
    wlist=get_wlist(w0=1e-8,Nw=2000,mesh_type='log',Gap=0,D=[-0.5,1])
    pmask=wlist>0
    ion()
    rholist=abs(random_function()(wlist))
    if ndim(rholist)>1:
        rholist=sqrt((rholist*swapaxes(rholist,1,2)).sum(axis=(1,2)))
    colors=['r','g','b','k','y','c']
    plts=[]
    for i,tick_type in enumerate(tick_types):
        offset_y=i
        ticker=get_ticker(tick_type,D=wlist[-1],N=N,Lambda=Lambda,Gap=0,wlist=wlist[pmask],rholist=rholist[pmask])
        plt=scatter(ticker(arange(2,2+N+1)),offset_y*ones(N+1),edgecolor='none',color=colors[i],label=tick_type)
        plts.append(plt)
        #for negative branch
        ticker_=get_ticker(tick_type,D=-wlist[0],N=N,Lambda=Lambda,Gap=0,wlist=-wlist[~pmask][::-1],rholist=rholist[~pmask][::-1])
        plt=scatter(-ticker_(arange(2,2+N+1)),offset_y*ones(N+1),edgecolor='none',color=colors[i],label=tick_type)
        #consistancy check
        assert_allclose(ticker(arange(1,N+2)),[ticker(i) for i in xrange(1,N+2)])
    legend(plts,tick_types,loc=2)
    plot(wlist,rholist)
    pdb.set_trace()

if __name__=='__main__':
    test_tick()


