from numpy import *
from scipy.linalg import qr,inv,sqrtm,eigh,norm
from scipy.sparse import diags
from scipy.sparse.linalg import eigs
from matplotlib.pyplot import *
import scipy.sparse as sps
import time,pdb

__all__=['icgs','construct_tridmat','tridiagonalize','tridiagonalize_sqrtm','tridiagonalize_qr','tridiagonalize_mpqr','tridiagonalize_mp']

def icgs(u,Q,M=None,return_norm=False,maxiter=3):
    '''
    Iterative Classical M-orthogonal Gram-Schmidt orthogonalization.

    Parameters:
        :u: vector, the column vector to be orthogonalized.
        :Q: matrix, the search space.
        :M: matrix/None, the matrix, if provided, perform M-orthogonal.
        :return_norm: bool, return the norm of u.
        :maxiter: int, the maximum number of iteractions.

    Return:
        vector, orthogonalized vector u.
    '''
    assert(ndim(u)==2)
    uH,QH=u.T.conj(),Q.T.conj()
    alpha=0.5
    it=1
    Mu=M.dot(u) if M is not None else u
    r_pre=norm(uH.dot(Mu))
    for it in xrange(maxiter):
        u=u-Q.dot(QH.dot(Mu))
        Mu=M.dot(u) if M is not None else u
        r1=norm(uH.dot(Mu))
        if r1>alpha*r_pre:
            break
        r_pre=r1
    if r1<=alpha*r_pre:
        warnings.warn('loss of orthogonality @icgs.')
    return (u,r1) if return_norm else u

def tridiagonalize(A, q, m=None,getbasis=False):
    """
    Use m steps of the lanczos algorithm starting with q to generate
    the tridiagonal form of this matrix(The traditional scalar version).

    Parameters:
        :A: A sparse hermitian matrix.
        :q: The starting vector.
        :m: The steps to run.
        :getbasis: Return basis vectors if True.

    Return:
        Tridiagonal part elements (data,offset),
        | data -> (lower part, middle part, upper part)
        | offset -> (-1, 0, 1) to indicate the value of (j-i) of specific data with i,j the matrix element indices.

    To construct the matrix, set the block-matrix elements with block indices j-i == offset[k] to data[k].
    This is exactly what `construct_tridmat` function do.

    **Note:** The initial vector q will be renormalized to guarant the correctness of result,
    """
    if sps.issparse(A): A=A.toarray()
    if m==None:
        m=A.shape[0]

    #initialize states
    qq=q/sqrt(dot(q,q))
    Q=qq[...,newaxis]
    alpha=[]
    beta=[]
    
    #run steps
    for i in range(m):
        Q_=Q[:,i]
        z = A.dot(Q_)
        alpha_i = dot(conj(z), Q_)
        tmp = dot(conj(Q.T), z)
        tmp = dot(Q, tmp)
        z = z - tmp
        beta_i = sqrt(dot(conj(z),z))
        alpha.append(alpha_i)
        beta.append(beta_i.item())
        if i==m-1: break
        z=z/beta_i
        Q_i=icgs(z[:,newaxis],Q)
        Q=append(Q,Q_i,axis=-1)
    Bl=array(beta[0:m-1])
    if A.shape[0]==1:
        data = array([zeros(0),alpha,zeros(0)])
    else:
        data = array([conj(Bl), alpha, Bl])
    offsets = array([-1, 0, 1])
    if not getbasis:
        return data,offsets
    else:
        return data,offsets,Q

def tridiagonalize_sqrtm(A,q,m=None,getbasis=False):
    """
    Use block lanczos algorithm to generate the tridiagonal part of matrix.
    This is the symmetric version of block-tridiagonalization in contrast to `qr` version.
    However, only matrices with blocksize p = 2 are currently supported.

    Parameters:
        :A: A sparse Hermitian matrix.
        :q: The starting columnwise orthogonal vector q with shape (n*p,p) with p the block size and n the number of blocks.
        :m: the steps to run.
        :getbasis: Return basis vectors if True.

    Return:
        Tridiagonal part elements (data,offset),
        | data -> (lower part, middle part, upper part)
        | offset -> (-1, 0, 1) to indicate the value of (j-i) of specific data with i,j the matrix element indices.

        To construct the matrix, set the block-matrix elements with block indices j-i == offset[k] to data[k].
        This is exactly what `construct_tridmat` function do.

        **Note:** The orthogonality of initial vector q will be re-inforced to guarant the convergent result,
        meanwhile, the orthogonality of starting vector is also checked.
    """
    n=q.shape[1]
    if sps.issparse(A): A=A.toarray()
    if m==None:
        m=A.shape[0]/n

    #check for othogonality of `q vector`.
    if not allclose(q.T.conj().dot(q),identity(q.shape[1])):
        raise Exception('Error','Othogoanlity check for start vector q failed.')
    #reinforce the orthogonality.
    Q=qr(q,mode='economic')[0]

    #initialize states
    alpha=[]
    beta=[]
    
    #run steps
    for i in range(m):
        qi_1=Q[:,(i-1)*n:(i+1)*n]
        qi=Q[:,i*n:i*n+n]

        z=A.dot(qi)
        ai=dot(z.T.conj(),qi)
        tmp=dot(qi_1.T.conj(),z)
        tmp=dot(qi_1,tmp)
        z=z-tmp
        bi=sqrtm(dot(z.T.conj(),z))
        alpha.append(ai)
        beta.append(bi)

        if i==m-1: break
        z=dot(z,inv(bi))
        Q_i=icgs(z,Q)
        Q=append(Q,Q_i,axis=-1)
        if sum(abs(bi))<1e-20:
            print 'Warning! bad krylov space!'

    Bl=array(beta[:m-1])
    if q.shape[0]/n==1:
        BTl=zeros(0)
    else:
        BTl=swapaxes(Bl,1,2).conj()
    data = array([BTl, alpha, Bl])
    offsets = array([-1, 0, 1])
    if not getbasis:
        return data,offsets
    else:
        return data,offsets,Q

def tridiagonalize_qr(A,q,m=None):
    """
    Use m steps of the lanczos algorithm starting with q - the block QR decomposition version.

    Parameters:
        :A: A sparse Hermitian matrix.
        :q: The starting columnwise orthogonal vector q with shape (n*p,p) with p the block size and n the number of blocks.
        :m: The number of iteractions.

    Return:
        Tridiagonal part elements (data,offset),
        | data -> (lower part, middle part, upper part)
        | offset -> (-1, 0, 1) to indicate the value of (j-i) of specific data with i,j the matrix element indices.

        To construct the matrix, set the block-matrix elements with block indices j-i == offset[k] to data[k].
        This is exactly what `construct_tridmat` function do.

        **Note:** The orthogonality of initial vector q will be re-inforced to guarant the convergent result,
        meanwhile, the orthogonality of starting vector is also checked.
    """
    n=q.shape[1]
    if sps.issparse(A): A=A.toarray()
    if m==None:
        m=A.shape[0]/n

    #check for othogonality of original `q vector`.
    if not allclose(q.T.conj().dot(q),identity(n)):
        raise Exception('Error','Othogoanlity check for starting vector q failed.')

    Al=[]
    Bl=[zeros([n,n])]
    Ql=concatenate([zeros(q.shape),q],axis=-1)
    #run steps
    for i in range(m):
        U_i=A.dot(Ql[:,-n:])-dot(Ql[:,-2*n:-n],Bl[-1].T.conj())
        A_i=dot(Ql[:,-n:].T.conj(),U_i)
        R_i=U_i-dot(Ql[:,-n:],A_i)
        Q_i,B_i=qr(R_i,mode='economic')
        Al.append(A_i)
        Bl.append(B_i)

        #reinforce orthorgonality, Q_i orth- Q
        Q_i=icgs(Q_i,Ql[:,n:],return_norm=False)
        Ql=concatenate([Ql,Q_i],axis=1)
        if i!=m-1 and sum(abs(B_i))<1e-20:
            print 'Warning! bad krylov space!'

    Bl=array(Bl[1:m])
    if q.shape[0]/n==1:
        BTl=zeros(0)
    else:
        BTl=swapaxes(Bl,1,2).conj()
    data = array([Bl, Al, BTl])
    offset = array([-1, 0, 1])
    return data,offset

def construct_tridmat(data,offset):
    '''
    Construct tridiagonal matrix.

    Parameters:
        :data: The datas of lower, middle, upper tridiagonal part.
        :offset: The offsets indicating the position of datas.

    Return:
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

def tridiagonalize_mp(A, q, m=None,prec=5000,getbasis=False):
    """
    Use m steps of the lanczos algorithm starting with q to generate
    the tridiagonal form of this matrix(The traditional scalar version).

    Parameters:
        :A: A sparse hermitian matrix.
        :q: The starting vector.
        :m: The steps to run.
        :prec: The precision in bit, `None` for double precision.
        :getbasis: Return basis vectors if True.

    Return:
        Tridiagonal part elements (data,offset),
        | data -> (lower part, middle part, upper part)
        | offset -> (-1, 0, 1) to indicate the value of (j-i) of specific data with i,j the matrix element indices.

    To construct the matrix, set the block-matrix elements with block indices j-i == offset[k] to data[k].
    This is exactly what `construct_tridmat` function do.

    **Note:** The initial vector q will be renormalized to guarant the correctness of result,
    """
    import gmpy2
    from mplib import mpqr,mpconj
    gmpy2.get_context().precision=prec
    lin={'conj':mpconj,'tocomplex':vectorize(gmpy2.mpc),'sqrt':vectorize(gmpy2.sqrt)}
    if sps.issparse(A): A=A.toarray()
    if m==None:
        m=A.shape[0]

    #initialize states
    qq=q/lin['sqrt'](dot(q,q))
    Q=qq[...,newaxis]
    alpha=[]
    beta=[]
    
    #run steps
    for i in range(m):
        Q_=Q[:,i]
        z = A.dot(Q_)
        alpha_i = dot(lin['conj'](z), Q_)
        tmp = dot(lin['conj'](Q.T), z)
        tmp = dot(Q, tmp)
        z = z - tmp
        beta_i = lin['sqrt'](dot(lin['conj'](z),z))
        z=z/beta_i

        alpha.append(alpha_i)
        beta.append(beta_i.item())
        Q=append(Q,z[...,newaxis],axis=-1)
    Bl=array(beta[0:m-1])
    if A.shape[0]==1:
        data = array([zeros(0),alpha,zeros(0)])
    else:
        data = array([lin['conj'](Bl), alpha, Bl])
    offsets = array([-1, 0, 1])
    if not getbasis:
        return data,offsets
    else:
        return data,offsets,Q



def tridiagonalize_mpqr(A,q,m=None,prec=5000):
    """
    High precision version of block tridiagonalization.
    Use m steps of the lanczos algorithm starting with q - the block QR decomposition version.

    Parameters:
        :A: matrix, Hermitian matrix.
        :q: 1d array, starting columnwise orthogonal vector q with shape (n*p,p) with p the block size and n the number of blocks.
        :m: int, number of iteractions.
        :prec: int, precision in bits

    Return:
        Tridiagonal part elements (data,offset),
        | data -> (lower part, middle part, upper part)
        | offset -> (-1, 0, 1) to indicate the value of (j-i) of specific data with i,j the matrix element indices.

        To construct the matrix, set the block-matrix elements with block indices j-i == offset[k] to data[k].
        This is exactly what `construct_tridmat` function do.

        **Note:** The orthogonality of initial vector q will be re-inforced to guarant the convergent result,
        meanwhile, the orthogonality of starting vector is also checked.
    """
    #setup environment
    import gmpy2
    from mplib import mpqr,mpconj
    gmpy2.get_context().precision=prec
    lin={'conj':mpconj,'qr':mpqr,'tocomplex':vectorize(gmpy2.mpc)}

    n=q.shape[1]
    if sps.issparse(A): A=A.toarray()
    if m==None:
        m=A.shape[0]/n
    A=lin['tocomplex'](A)

    #check and reinforce the orthogonality.
    if not allclose(complex128(q).T.conj().dot(complex128(q)),identity(q.shape[1])):
        raise Exception('Error','Othogoanlity check for start vector q failed.')
    Q=lin['qr'](q)[0]

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


