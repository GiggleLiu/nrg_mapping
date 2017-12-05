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
from ..tridiagonalize import *

class TridTest(object):
    '''test tridiagonalization.'''
    def gen_randomH(self,n,p,matrix_type='array'):
        '''get matrix with n-nlock and blocksize p.'''
        if p is None:
            p=1
        N=n*p
        H0=random.random([N,N])+1j*random.random([N,N])
        H0=H0+H0.T.conj()
        if matrix_type=='array':
            return H0
        elif matrix_type=='sparse':
            return sps.csr_matrix(H0)
        else:
            raise ValueError()

    def gen_randv0(self,n,p,matrix_type='array'):
        '''generate a random orthorgonal vetor.'''
        if p is not None:
            v0=random.random([n*p,p])
            v0=qr(v0,mode='economic')[0]  #The orthogonality of starting vector v0 will be re-inforced in the program.
        else:
            v0=random.random(n)
            v0=v0/norm(v0)
        if matrix_type=='array':
            return v0
        elif matrix_type=='sparse':
            return sps.csr_matrix(v0)
        else:
            raise ValueError()

    def test1(self,n,p,matrix_type,method):
        '''
        check the quality of tridiagonalization.
        '''
        v0=self.gen_randv0(n,p,matrix_type='array')
        H0=self.gen_randomH(n,p,matrix_type=matrix_type)
        if p is not None:
            if method=='qr':
                data,offset=tridiagonalize_qr(H0,q=v0)
            if method=='mpqr':
                data,offset=tridiagonalize_mpqr(H0,q=v0)
            else:
                data,offset=tridiagonalize_sqrtm(H0,q=v0)
        else:
            data,offset=tridiagonalize_mp(H0,q=v0)
        B=construct_tridmat(data,offset).toarray()
        if sps.issparse(H0):
            H0=H0.toarray()
        H0=complex128(H0)
        e1=eigvalsh(H0)
        e2=eigvalsh(B)
        assert_allclose(e1,e2)

    def test_all(self):
        '''test all the cases.'''
        nl=[1,15]
        pl=[None,1,2,3]
        matrix_types=['array','sparse']
        for n in nl:
            for p in pl:
                for matrix_type in matrix_types:
                    for method in ['qr','sqrtm','mpqr']:
                        print('Testing n=%s,p=%s,matrix_type=%s,method=%s'%(n,p,matrix_type,method))
                        self.test1(n=n,p=p,matrix_type=matrix_type,method=method)

if __name__=='__main__':
    TridTest().test_all()
