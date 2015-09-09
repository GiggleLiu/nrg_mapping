#!/usr/bin/python
'''
Tridiagonalization methods for both scalar(tridiagonalize) and block(tridiagonalize_qr) versions.
Some test functions are also included.
'''
from numpy import *
from scipy.linalg import *
from scipy.sparse import diags,issparse
from scipy.sparse.linalg import eigs
from utils import mpconj,qr2
from matplotlib.pyplot import *
import scipy.sparse as sps
import gmpy2
import time,pdb

def tridiagonalize(A, q, m=None,prec=None,getbasis=False):
    """
    Use m steps of the lanczos algorithm starting with q to generate
    eigenvalues for the sparse symmetric matrix A.

    A:
        a sparse symmetric matrix.
    q:
        the starting vector.
    m:
        the steps to run.
    getbasis:
        return basis vectors if True.
    *return*:
        (data,offset,vectorbase)

        Use scipy.sparse.diags(res[0],res[1]) to generate a sparse matrix 
    """
    if m==None:
        m=len(A)

    #set the precision, use gmpy2 if set.
    if prec!=None:
        gmpy2.get_context().precision=prec
        csqrt=vectorize(gmpy2.sqrt)
    else:
        csqrt=sqrt

    #initialize states
    qq=q/csqrt(dot(q,q))
    Q=qq[...,newaxis]
    alpha=[]
    beta=[]
    
    #run steps
    for i in range(m):
        Q_=Q[:,i]
        z = A.dot(Q_)
        alpha_i = dot(z, Q_)
        tmp = dot(Q.T, z)
        tmp = dot(Q, tmp)
        z = z - tmp
        beta_i = csqrt(dot(z,z))
        z=z/beta_i

        alpha.append(alpha_i)
        beta.append(beta_i)
        Q=append(Q,z[...,newaxis],axis=-1)
    data = array([beta[0:m-1], alpha, beta[0:m-1]])
    offsets = array([-1, 0, 1])
    if not getbasis:
        return data,offsets
    else:
        return data,offsets,Q

def tridiagonalize_qr(A,q,m=None,prec=None):
    """
    Use m steps of the lanczos algorithm starting with q - the block version with QR decomposition.

    *Note: we need to specify a two-column starting vectors here (q0,q1) with q0,q1 orthogonal to each other.*

    A:
        a sparse symmetric matrix.
    q:
        the starting othogonal vector q=(q0,q1).
    m:
        the steps to run.
    *return*:
        (data,offset), the trdiagonal matrix can be generated by scipy.sparse.diags(data,offset).
    """
    n=q.shape[1]
    if m==None:
        m=len(A)/n
    #set the precision, use gmpy2 if set.
    if prec!=None:
        gmpy2.get_context().precision=prec
        cconj=mpconj
        cqr=qr2
    else:
        cconj=conj
        cqr=lambda A:qr(A,mode='economic')
    #check for othogonality of `q vector`.
    deviation=sum(abs(dot(cconj(q.T),q)-identity(n)))
    if deviation>1e-10:
        raise Exception('Error','Othogoanlity check for start vector q failed(deviation %s).'%float64(deviation))

    Al=[]
    Bl=[zeros([n,n])]
    Ql=[zeros(q.shape),q]
    #run steps
    for i in range(m):
        U_i=A.dot(Ql[-1])-dot(Ql[-2],cconj(Bl[-1].T))
        A_i=dot(cconj(Ql[-1].T),U_i)
        R_i=U_i-dot(Ql[-1],A_i)
        Q_i,B_i=cqr(R_i)
        Al.append(A_i)
        Bl.append(B_i)
        Ql.append(Q_i)
        if sum(abs(B_i))<1e-20:
            print 'Maximum krylov space(%s) reached!'%i
            if i!=m-1:
                print 'Unreliable results will accur!'

    Bl=Bl[1:m]
    BTl=[cconj(b.T) for b in Bl]
    data = array([BTl, Al, Bl])
    offset = array([-1, 0, 1])
    return data,offset

def check_tridiagonalize(H0,trid):
    '''
    check the quality of tridiagonalization.

    H0:
        the original hamiltonian.
    trid:
        tridiagonalization result, a tuple of (data,offset).
    '''
    ion()
    data,offset=trid
    N=len(data[1])
    B=ndarray([N,N],dtype='O')
    #fill datas
    for i in xrange(N):
        for j in xrange(N):
            for k in xrange(3):
                if i-j==offset[k]:
                    B[i,j]=complex128(data[offset[k]+1][min(i,j)])
    B=sps.bmat(B).toarray()
    e1=eigvalsh(H0)
    e2=eigvalsh(B)
    plot(sort(e1),lw=3);plot(sort(e2),lw=3)
    legend(['Original','Mapped'])
    pdb.set_trace()

