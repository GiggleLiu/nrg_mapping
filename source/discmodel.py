'''
The Discretized model, or the sun model.
'''

from numpy import *
from scipy import sparse as sps
import pdb

__all__=['DiscModel','load_discmodel']

class DiscModel(object):
    '''
    Discrete model class.

    Construct:
        DiscModel((Elist_neg,Elist_pos),(Tlist_neg,Tlist_pos),z=1.)

        z could be one or an array of float >0 but <=1.

    Attributes:
        :Elist_neg/Elist_pos/Elist(readonly)/Tlist_neg/Tlist_pos/Tlist(readonly): An array of on-site energies and hopping terms.

            * The shape of Elist_neg/Tlist_neg is (N_neg,nz,nband,nband)
            * The shape of Elist_pos/Tlist_pos is (N_pos,nz,nband,nband)
            * The shape of Elist/Tlist is (N_pos+N_neg,nz,nband,nband)
        :z: The twisting parameters.
        :nz: The number of z-numbers(readonly)
        :nband: The number of bands(readonly)
        :N_neg/N_pos/N: The number of intervals for negative/positive/total band(readonly)
        :is_scalar: Is a single band scalar model if True(readonly)
    '''
    def __init__(self,Elists,Tlists,z=1.):
        if ndim(z)==0:
            self.z=array([z])
        elif ndim(z)==1:
            self.z=array(z)
        else:
            raise Exception('z must be an 1D array or a Number !')
        if any(z)>1. or any(z<=0.):
            raise Exception('z must greater than 0 and small or equal to 1!')
        assert(shape(Elists[0])==shape(Tlists[1]))
        assert(shape(Elists[0])==shape(Tlists[1]))
        assert(ndim(Elists[0])==2 or ndim(Elists[0])==4)  #when ndim=2, the last 2 dimensions are ignored(scalar).

        self.Tlist_neg,self.Tlist_pos=Tlists
        self.Elist_neg,self.Elist_pos=Elists

    @property
    def nz(self):
        '''number of twisting parameters.'''
        return len(self.z)

    @property
    def N_pos(self):
        '''The total number of particles in positive branch'''
        return len(self.Tlist_neg)

    @property
    def N_neg(self):
        '''The total number of particles in negative branch'''
        return len(self.Tlist_pos)

    @property
    def N(self):
        '''The total number of particles'''
        return self.N_neg+self.N_pos

    @property
    def nband(self):
        '''number of bands.'''
        if self.is_scalar:
            return 1
        return self.Elist_pos.shape[-1]

    @property
    def is_scalar(self):
        '''is a scalar model(no matrix representation of on-site energies and hopping) or not.'''
        return ndim(self.Elist_pos)==2

    @property
    def Elist(self):
        '''The list of on-site energies.'''
        return concatenate([self.Elist_neg[::-1],self.Elist_pos],axis=0)

    @property
    def Tlist(self):
        '''The list of on-site energies.'''
        return concatenate([self.Tlist_neg[::-1],self.Tlist_pos],axis=0)

    def get_H0(self,e0,iz):
        '''Get the hamiltonian.'''
        N=self.N+1
        mat=ndarray((N,N),dtype='O')
        elist,tlist=self.Elist[:,iz],self.Tlist[:,iz]

        #fill datas
        mat[0,0]=e0
        for i in xrange(1,N):
            ti=tlist[i-1]
            mat[i,i]=elist[i-1]
            mat[0,i]=ti.T.conj()
            mat[i,0]=ti

        return sps.bmat(mat).toarray()

    def save(self,file_prefix):
        '''
        Save data.

        Parameters:
            :file_prefix: The target filename prefix.

        **Note:**
        For the scalar model mapped from SingleBandDiscHandler, with z=[0.3,0.7] and 2 sites for each-branch.
        The data file `negfile`(`posfile`) looks like:
            | E1(z=0.7)
            | E1(z=0.7)
            | E2(z=0.7)
            | E2(z=0.7)
            | T1(z=0.7)
            | T1(z=0.7)
            | T2(z=0.7)
            | T2(z=0.7)

        However, for the multi-band model, the parameters are allowed to take imaginary parts,
        Now, the data file for a two band model looks like:
            | E1[0,0].real, E1[0,0].imag, E1[0,1].real, E1[0,1].imag, E1[1,0].real, E1[1,0].imag, E1[1,1].real, E1[1,1].imag  #z=0.3
            | ...

        It will take 8 columns to store each matrix element.
        '''
        N_neg,N_pos=self.N_neg,self.N_pos
        nband=self.nband
        nz=self.nz
        zfile='%s.z.dat'%(file_prefix)
        negfile='%s.neg.dat'%(file_prefix)
        posfile='%s.pos.dat'%(file_prefix)
        if self.is_scalar:
            negdata=concatenate([self.Elist_neg,self.Tlist_neg]).real.reshape([2*N_neg*nz])
            posdata=concatenate([self.Elist_pos,self.Tlist_pos]).real.reshape([2*N_pos*nz])
        else:
            negdata=complex128(concatenate([self.Elist_neg,self.Tlist_neg])).view('float64').reshape([2*N_neg*nz,nband**2*2])
            posdata=complex128(concatenate([self.Elist_pos,self.Tlist_pos])).view('float64').reshape([2*N_pos*nz,nband**2*2])
        savetxt(negfile,negdata)
        savetxt(posfile,posdata)
        savetxt(zfile,self.z)

def load_discmodel(file_prefix):
    '''
    Load specific data.

    Parameters:
        :file_prefix: The target filename prefix.

    Return:
        A <DiscModel> instance.
    '''
    zfile='%s.z.dat'%(file_prefix)
    negfile='%s.neg.dat'%(file_prefix)
    posfile='%s.pos.dat'%(file_prefix)
    z=loadtxt(zfile)
    nz=len(atleast_1d(z))
    negdata=loadtxt(negfile)
    posdata=loadtxt(posfile)
    if ndim(negdata)==1:
        negdata=negdata.reshape([-1,nz])
        posdata=posdata.reshape([-1,nz])
    else:
        #the matrix version contains complex numbers.
        nband=sqrt(negdata.shape[1]/2).astype('int32')
        negdata=negdata.view('complex128').reshape([-1,nz,nband,nband])
        posdata=posdata.view('complex128').reshape([-1,nz,nband,nband])
    Elist_neg,Tlist_neg=split(negdata,2)
    Elist_pos,Tlist_pos=split(posdata,2)
    return DiscModel(Elists=(Elist_neg,Elist_pos),Tlists=(Tlist_neg,Tlist_pos),z=z)

