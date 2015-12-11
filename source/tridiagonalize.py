from numpy import *
from scipy import linalg
from scipy.sparse import diags
from scipy.sparse.linalg import eigs
from matplotlib.pyplot import *
import scipy.sparse as sps
import gmpy2
import time,pdb

import utils

__all__=['construct_tridmat','tridiagonalize','tridiagonalize2','tridiagonalize_qr']

linalg_np={'eigh':linalg.eigh,'eigh_pauliv':utils.eigh_pauliv_npy,
        'conj':conj,'qr':(lambda A:linalg.qr(A,mode='economic')),
        'inv':linalg.inv,'sqrtm':linalg.sqrtm,'tocomplex':complex128,
        }
linalg_mp={'eigh':utils.eigh2,'eigh_pauliv':utils.eigh_pauliv_mpc,
        'conj':utils.mpconj,'qr':utils.mpqr,
        'inv':utils.invh2,'sqrtm':utils.sqrth2,'tocomplex':vectorize(gmpy2.mpc),
        }

def tridiagonalize(A, q, m=None,prec=5000,getbasis=False):
    """
    Use m steps of the lanczos algorithm starting with q to generate
    the tridiagonal form of this matrix(The traditional scalar version).

    A:
        A sparse hermitian matrix.
    q:
        The starting vector.
    m:
        The steps to run.
    prec:
        The precision in bit, `None` for double precision.
    getbasis:
        Return basis vectors if True.

    Return
    ----------------------
    Tridiagonal part elements (data,offset),
    data -> (lower part, middle part, upper part)
    offset -> (-1, 0, 1) to indicate the value of (j-i) of specific data with i,j the matrix element indices.

    To construct the matrix, set the block-matrix elements with block indices j-i == offset[k] to data[k].
    This is exactly what `construct_tridmat` function do.

    Note: The initial vector q will be renormalized to guarant the correctness of result,
    """
    if sps.issparse(A): A=A.toarray()
    if m==None:
        m=A.shape[0]

    #set the precision, use gmpy2 if set.
    if prec is not None:
        gmpy2.get_context().precision=prec
        csqrt=vectorize(gmpy2.sqrt)
        cconj=linalg_mp['conj']
        A=lin['tocomplex'](A)
    else:
        csqrt=sqrt
        cconj=conj

    #initialize states
    qq=q/csqrt(dot(q,q))
    Q=qq[...,newaxis]
    alpha=[]
    beta=[]
    
    #run steps
    for i in range(m):
        Q_=Q[:,i]
        z = A.dot(Q_)
        alpha_i = dot(cconj(z), Q_)
        tmp = dot(cconj(Q.T), z)
        tmp = dot(Q, tmp)
        z = z - tmp
        beta_i = csqrt(dot(cconj(z),z))
        z=z/beta_i

        alpha.append(alpha_i)
        beta.append(beta_i.item())
        Q=append(Q,z[...,newaxis],axis=-1)
    Bl=array(beta[0:m-1])
    if A.shape[0]==1:
        data = array([zeros(0),alpha,zeros(0)])
    else:
        data = array([cconj(Bl), alpha, Bl])
    offsets = array([-1, 0, 1])
    if not getbasis:
        return data,offsets
    else:
        return data,offsets,Q

def tridiagonalize2(A,q,m=None,prec=5000,getbasis=False):
    """
    Use block lanczos algorithm to generate the tridiagonal part of matrix.
    This is the symmetric version of block-tridiagonalization in contrast to `qr` version.
    However, only matrices with blocksize p = 2 are currently supported.

    Parameters
    --------------------------
    A:
        A sparse Hermitian matrix.
    q:
        The starting columnwise orthogonal vector q with shape (n*p,p) with p the block size and n the number of blocks.
    m:
        the steps to run.
    prec:
        The precision in bit, `None` for double precision.
    getbasis:
        Return basis vectors if True.

    Return
    ---------------------
    Tridiagonal part elements (data,offset),
    data -> (lower part, middle part, upper part)
    offset -> (-1, 0, 1) to indicate the value of (j-i) of specific data with i,j the matrix element indices.

    To construct the matrix, set the block-matrix elements with block indices j-i == offset[k] to data[k].
    This is exactly what `construct_tridmat` function do.

    Note: The orthogonality of initial vector q will be re-inforced to guarant the convergent result,
    meanwhile, the orthogonality of starting vector is also checked.
    """
    n=q.shape[1]
    assert(n==2)  #only block dimension 2 is implemented!
    if sps.issparse(A): A=A.toarray()
    if m==None:
        m=A.shape[0]/n

    #set the precision, use gmpy2 if set.
    if prec is not None:
        gmpy2.get_context().precision=prec
        lin=linalg_mp
        A=lin['tocomplex'](A)
    else:
        lin=linalg_np

    #reinforce the orthogonality.
    Q=lin['qr'](q)[0]
    #Q=dot(q,lin['inv'](lin['sqrtm'](dot(lin['conj'](q.T),q))))
    #check for othogonality of `q vector`.
    if not allclose(complex128(Q),complex128(q)):
        raise Exception('Error','Othogoanlity check for start vector q failed.')

    #initialize states
    alpha=[]
    beta=[]
    
    #run steps
    for i in range(m):
        qi_1=Q[:,(i-1)*n:(i+1)*n]
        qi=Q[:,i*n:i*n+n]

        z=A.dot(qi)
        ai=dot(lin['conj'](z.T),qi)
        tmp=dot(lin['conj'](qi_1.T),z)
        tmp=dot(qi_1,tmp)
        z=z-tmp
        bi=lin['sqrtm'](dot(lin['conj'](z.T),z))
        z=dot(z,lin['inv'](bi))

        alpha.append(ai)
        beta.append(bi)
        Q=append(Q,z,axis=-1)
        if i!=m-1 and sum(abs(bi))<1e-20:
            print 'Warning! bad krylov space!'

    Bl=array(beta[:m-1])
    if q.shape[0]/n==1:
        BTl=zeros(0)
    else:
        BTl=lin['conj'](swapaxes(Bl,1,2))
    data = array([BTl, alpha, Bl])
    offsets = array([-1, 0, 1])
    if not getbasis:
        return data,offsets
    else:
        return data,offsets,Q

def tridiagonalize_qr(A,q,m=None,prec=5000):
    """
    Use m steps of the lanczos algorithm starting with q - the block QR decomposition version.

    Parameters
    ---------------------
    A:
        A sparse Hermitian matrix.
    q:
        The starting columnwise orthogonal vector q with shape (n*p,p) with p the block size and n the number of blocks.
    m:
        The number of iteractions.
    prec:
        The precision in bit, `None` for double precision.

    getbasis:
        Return basis vectors if True.

    Return
    ---------------------
    Tridiagonal part elements (data,offset),
    data -> (lower part, middle part, upper part)
    offset -> (-1, 0, 1) to indicate the value of (j-i) of specific data with i,j the matrix element indices.

    To construct the matrix, set the block-matrix elements with block indices j-i == offset[k] to data[k].
    This is exactly what `construct_tridmat` function do.

    Note: The orthogonality of initial vector q will be re-inforced to guarant the convergent result,
    meanwhile, the orthogonality of starting vector is also checked.
    """
    n=q.shape[1]
    if sps.issparse(A): A=A.toarray()
    if m==None:
        m=A.shape[0]/n
    #set the precision, use gmpy2 if set.
    if prec is not None:
        gmpy2.get_context().precision=prec
        lin=linalg_mp
        A=lin['tocomplex'](A)
    else:
        lin=linalg_np

    #reenforce the orthogonality.
    Q=lin['qr'](q)[0]
    #check for othogonality of original `q vector`.
    if not allclose(complex128(Q),complex128(q)):
        raise Exception('Error','Othogoanlity check for starting vector q failed.')

    Al=[]
    Bl=[zeros([n,n])]
    Ql=[zeros(Q.shape),Q]
    #run steps
    for i in range(m):
        U_i=A.dot(Ql[-1])-dot(Ql[-2],lin['conj'](Bl[-1].T))
        A_i=dot(lin['conj'](Ql[-1].T),U_i)
        R_i=U_i-dot(Ql[-1],A_i)
        Q_i,B_i=lin['qr'](R_i)
        Al.append(A_i)
        Bl.append(B_i)
        Ql.append(Q_i)
        if i!=m-1 and sum(abs(B_i))<1e-20:
            print 'Warning! bad krylov space!'

    Bl=array(Bl[1:m])
    if q.shape[0]/n==1:
        BTl=zeros(0)
    else:
        BTl=lin['conj'](swapaxes(Bl,1,2))
    data = array([Bl, Al, BTl])
    offset = array([-1, 0, 1])
    return data,offset

def construct_tridmat(data,offset):
    '''
    Construct tridiagonal matrix.

    Parameters
    ---------------------
    data:
        The datas of lower, middle, upper tridiagonal part.
    offset:
        The offsets indicating the position of datas.

    Return
    ---------------------
    2D sparse matrix, use res.toarray() to get a dense array.
    '''
    n=len(data[1])
    if ndim(data[1])==1:
        p=1
    else:
        p=len(data[1][0])
    N=n*p
    B=ndarray([n,n],dtype='O')
    #fill datas
    for i in xrange(n):
        for j in xrange(n):
            for k in xrange(3):
                if j-i==offset[k]:
                    B[i,j]=complex128(data[offset[k]+1][min(i,j)])
    B=sps.bmat(B)
    return B

