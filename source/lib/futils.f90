SUBROUTINE zinv(N,A)
    IMPLICIT NONE
    INTEGER N
    COMPLEX*16 A(N,N), WORK(2*N)
    INTEGER INFO, IPIV(N), LWORK
    !f2py intent(inout) :: A
    !f2py intent(in) :: N
    IF(N<=0)STOP 'INVALID DIM @ ZINVERT'
    IF(N==1)THEN; A=1/A; RETURN; END IF
    CALL ZGETRF(N, N, A, N, IPIV, INFO)
    IF(INFO/=0)PRINT*, 'LU@INVERSE DETECTED A SINGULARITY'
    LWORK=2*N
    CALL ZGETRI(N, A, N, IPIV, WORK, LWORK, INFO)
    IF(INFO/=0)STOP 'ZINVERT FAILED'
    RETURN
END SUBROUTINE zinv

!``` Python Code
!GL=array([sum([sum([dot(Ti.T.conj(),dot(H2G(Ei,w=w,geta=smearing*1./nz*max(1e-3,abs(w))),Ti))\
!        for Ei,Ti in zip(Elist[:,i],Tlist[:,i])],axis=0) for i in xrange(nz)],axis=0)/nz for w in wlist])
!AL=1j*(GL-transpose(GL,axes=(0,2,1)).conj())/(pi*2.)
subroutine hybri_sun(tlist,elist,wlist,hybri,smearing,nsite,nz,nband,nw)
    implicit none
    integer,intent(in) :: nsite,nw,nz,nband
    complex*16,intent(in) :: tlist(nsite,nz,nband,nband),elist(nsite,nz,nband,nband)
    real*8,intent(in) :: wlist(nw),smearing
    complex*16,intent(out) :: hybri(nw,nband,nband)
    complex*16 :: ti(nband,nband),gi(nband,nband),sigma(nband,nband)
    integer :: isite,iz,iw,iband
    real*8 :: identity(nband,nband),wi
    real*8,parameter :: pi=2*asin(1D0)
    complex*16,parameter :: one=dcmplx(0D0,1D0)

    !initialize identity matrix
    identity=0D0
    do iband=1,nband
        identity(iband,iband)=1D0
    enddo 

    do iw=1,nw
        wi=wlist(iw)
        sigma=0
        do iz=1,nz
            do isite=1,nsite
                ti=tlist(isite,iz,:,:)
                gi=dcmplx(wi,smearing*max(1D-4,abs(wi)))*identity-elist(isite,iz,:,:)
                call zinv(nband,gi)
                sigma=sigma+matmul(matmul(transpose(conjg(ti)),gi),ti)
            enddo
        enddo
        hybri(iw,:,:)=one*(sigma-transpose(conjg(sigma)))/2/pi
    enddo
    hybri=hybri/nz
end subroutine hybri_sun

!``` Python Code
!dlv=[]
!for iz in xrange(nz):
!    print 'Recovering spectrum for %s-th z number.'%iz
!    chain=chains[iz]
!    el=chain.elist
!    tl=concatenate([chain.t0[newaxis,...],chain.tlist],axis=0)
!    dl=[]
!    for w in wlist:
!        sigma=0
!        for e,t in zip(el[::-1],tl[::-1]):
!            geta=abs(w)+1e-10
!            g0=H2G(w=w,h=e+sigma,geta=smearing*geta/nz)
!            tH=transpose(conj(t))
!            sigma=dot(tH,dot(g0,t))
!        dl.append(1j*(sigma-sigma.T.conj())/2./pi)
!    dlv.append(dl)
!dlv=mean(dlv,axis=0)
subroutine hybri_chain(tlist,elist,wlist,hybri,smearing,nsite,nz,nband,nw)
    implicit none
    integer,intent(in) :: nsite,nw,nz,nband
    complex*16,intent(in) :: tlist(nz,nsite,nband,nband),elist(nz,nsite,nband,nband)
    real*8,intent(in) :: wlist(nw),smearing
    complex*16,intent(out) :: hybri(nw,nband,nband)
    complex*16 :: ti(nband,nband),gi(nband,nband),sigma(nband,nband)
    integer :: isite,iz,iw,iband
    real*8 :: identity(nband,nband),wi
    real*8,parameter :: pi=2*asin(1D0)
    complex*16,parameter :: one=dcmplx(0D0,1D0)

    !initialize identity matrix
    identity=0D0
    do iband=1,nband
        identity(iband,iband)=1D0
    enddo 
    hybri=0

    do iw=1,nw
        wi=wlist(iw)
        do iz=1,nz
            sigma=0
            do isite=nsite,1,-1
                ti=tlist(iz,isite,:,:)
                gi=dcmplx(wi,maxval(abs(ti))*smearing)*identity-elist(iz,isite,:,:)-sigma
                !gi=dcmplx(wi,smearing*max(1D-4,abs(wi)))*identity-elist(iz,isite,:,:)-sigma
                call zinv(nband,gi)
                sigma=matmul(matmul(transpose(conjg(ti)),gi),ti)
            enddo
            hybri(iw,:,:)=hybri(iw,:,:)+one*(sigma-conjg(transpose(sigma)))/2/pi
        enddo
    enddo
    hybri=hybri/nz
end subroutine hybri_chain
