'''
The Wilson chain class.
'''
from numpy import *
from scipy import sparse as sps
import warnings

from tridiagonalize import construct_tridmat

__all__=['Chain','load_chain']

class Chain(object):
    '''
    Wilson chain class.
    
    Attributes:
        :t0: number/2D array, the coupling term of the first site and the impurity.
        :elist/tlist: 2D array/4D array, A list of on-site energies and coupling terms.
        :is_scalar: bool, True is it is a single band scalar chain.(readonly)
        :nband: integer, the number of bath sites.(readonly)
        :nsite: integer, the number of bands.(readonly)
    '''
    def __init__(self,t0,elist,tlist):
        self.t0=complex128(t0)
        self.elist=complex128(elist)
        self.tlist=complex128(tlist)

    @property
    def is_scalar(self):
        '''True if this is a model mapped from scalar hybridization function.'''
        return ndim(self.t0)<2

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
            return
        return Chain(self.t0[...,0,0],self.elist[...,0,0],self.tlist[...,0,0])

    def save(self,file_prefix):
        '''
        save a Chain instance to files.

        Parameters:
            :file_prefix: A string as a prefix to store datas of a chain.

        **Note:**
        The data is stored in 3 files,

        1. <file_prefix>.info.dat, the true shape of elist.
            e.g. for multi-band, it's (# of sites, degree of freedom, degree of freedom).
                 for single-band, it's (# of sites).
        2. <file_prefix>.el.dat, an array of on-site energies, in the format(2-band,2-site as an example)
            | E0[0,0].real E0[0,0].imag
            | E0[0,1].real E0[0,1].imag
            | E0[1,0].real E0[1,0].imag
            | E0[0,1].real E0[0,1].imag
            | E1[0,0].real E1[0,0].imag
            |          ... ...
        3. <file_prefix>.tl.dat, an array of hopping terms with the first element t0 the coupling strength with impurity.
            Its data format is similar to `el.dat`.
        '''
        tlist=concatenate([self.t0[newaxis,...],self.tlist],axis=0)
        savetxt(file_prefix+'.info.dat',array(self.elist.shape))
        savetxt(file_prefix+'.el.dat',self.elist.ravel().view('float64'))
        savetxt(file_prefix+'.tl.dat',tlist.ravel().view('float64'))

    def get_H0(self,e0):
        '''
        Hamiltonian without interaction terms.

        Return:
            sparse matrix, the hamiltonian.
        '''
        tlist=self.tlist
        elist=self.elist
        N=self.nsite+1
        B=ndarray([N,N],dtype='O')
        elist=concatenate([e0[newaxis,...],elist],axis=0)
        tlist=concatenate([self.t0[newaxis,...],tlist],axis=0)
        tlistH=swapaxes(tlist,1,2).conj()
        offset=[-1,0,1]
        return construct_tridmat([tlist,elist,tlistH],offset).toarray()

def load_chain(file_prefix):
    '''
    load a Chain instance from files.

    Parameters:
        :file_prefix: A string as a prefix to store datas of a chain.

    Return:
        A <Chain> instance.
    '''
    shape=loadtxt(file_prefix+'.info.dat').astype('int32')
    elist=loadtxt(file_prefix+'.el.dat').view('complex128').reshape(shape)
    tlist=loadtxt(file_prefix+'.tl.dat').view('complex128').reshape(shape)
    return Chain(tlist[0],elist,tlist[1:])

