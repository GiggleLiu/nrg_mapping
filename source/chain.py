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
        :elist/tlist: 2D array/4D array, A list of on-site energies and coupling terms.
        :nband: integer, the number of bath sites.(readonly)
        :nsite: integer, the number of bands.(readonly)
    '''
    def __init__(self,elist,tlist):
        if ndim(elist)!=3 or ndim(tlist)!=3:
            raise ValueError()
        self.elist=complex128(elist)
        self.tlist=complex128(tlist)

    @property
    def nsite(self):
        '''The number of sites.'''
        return len(self.elist)

    @property
    def nband(self):
        '''number of bands.'''
        return self.tlist.shape[-1]

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
        savetxt(file_prefix+'.info.dat',array(self.elist.shape))
        savetxt(file_prefix+'.el.dat',self.elist.ravel().view('float64'))
        savetxt(file_prefix+'.tl.dat',self.tlist.ravel().view('float64'))

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
    return Chain(elist,tlist)

