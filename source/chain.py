'''
The Wilson chain class.
'''
from numpy import *
import warnings

from nrg_setting import DATA_FOLDER

__all__=['Chain','load_chain']

class Chain(object):
    '''
    NRG chain class.
    
    t0:
        the coupling term of the first site and the impurity.
    elist/tlist:
        a list of on-site energies and coupling terms.
    '''
    def __init__(self,t0,elist,tlist):
        self.t0=complex128(t0)
        self.elist=complex128(elist)
        self.tlist=complex128(tlist)

    @property
    def is_scalar(self):
        '''True if this is a model mapped from scalar hybridization function.'''
        return ndim(self.t0)<=2

    @property
    def nsite(self):
        '''The number of sites.'''
        return len(self.elist)

    @property
    def nband(self):
        '''number of bands.'''
        if self.is_scalar:
            return 1
        else:
            return self.tlist.shape[-1]

    def to_scalar(self):
        '''transform 1-band non-scalar model to scalar one by remove redundant dimensions.'''
        if self.nband!=1:
            warnings.warn('Warning! Parse multi-band model to scalar model!')
        return Chain(self.t0[...,0,0],self.elist[...,0,0],self.tlist[...,0,0])

    def save(self,token):
        '''
        save a Chain instance to files.

        token:
            a string as a prefix to store datas of a chain.
        '''
        token='%s/'%DATA_FOLDER+token
        tlist=concatenate([self.t0[newaxis,...],self.tlist],axis=0)
        savetxt(token+'.info.dat',array(self.elist.shape))
        savetxt(token+'.el.dat',self.elist.ravel().view('float64'))
        savetxt(token+'.tl.dat',tlist.ravel().view('float64'))

def load_chain(token):
    '''
    load a Chain instance from files.

    token:
        a string as a prefix to store datas of a chain.
    *return*:
        a Chain instance.
    '''
    token='%s/'%DATA_FOLDER+token
    shape=loadtxt(token+'.info.dat')
    elist=loadtxt(token+'.el.dat').view('complex128').reshape(shape)
    tlist=loadtxt(token+'.tl.dat').view('complex128').reshape(shape)
    return Chain(tlist[0],elist,tlist[1:])

