'''
Quick map from hybridization function to chain..
'''

from numpy import *

from discretization import quick_map
from chainmapper import map2chain
from scipy.interpolate import interp1d
from configobj import ConfigObj
from validate import Validator
import sys,pdb,os

__all__=['quickmap','run_config']

def quickmap(wlist,rhofunc,Lambda=2.0,nsite=25,nz=1,tick_type='adaptive'):
    '''
    Quick map from hybridization function to chain..

    Parameters:
        :wlist: 1D array, frequency space holding this hybridization function.
        :rhofunc: function, hybridization function.
        :nsite: integer, number of discrete sites for each z number.
        :Lambda: scaling factor.
        :nz: int, number of twisting parameters.
        :tick_type: 'adaptive' or 'log'

    Return:
        list, the <Chain> instances for each z.
    '''
    if nz==1:
        z=1.
    else:
        z=linspace(0.5/nz,1-0.5/nz,nz)
    print 'Start mapping the hybridization function to sun model.'
    discmodel=quick_map(rhofunc=rhofunc,wlist=wlist,N=nsite,z=z,Nx=200000,tick_params={'tick_type':tick_type,'Lambda':Lambda},autofix=1e-5)[1]
    print 'Start mapping the sun model to a Wilson chain.'
    chains=map2chain(discmodel,nsite=nsite,normalize_method='qr')
    print 'Done'
    return chains

def run_config(config_file):
    '''Run according to the configuration file `config.py`.'''
    specfile=os.path.join(os.path.dirname(__file__),'config-spec.ini')
    config=ConfigObj(config_file,configspec=specfile,stringify=True)
    validator = Validator()
    result = config.validate(validator,preserve_errors=True)
    if result!=True:
        raise ValueError('Configuration Error! %s'%result)
    hybri_config=config['hybri_func']
    disc_config=config['discretization']
    trid_config=config['tridiagonalization']
    file_prefix=config['global']['data_folder'].strip('/')+'/'+config['global']['data_token']

    #load hybridization function and do elementwise interpolation
    hybri_file=hybri_config['hybridization_func_file']
    is_complex=hybri_file.split('.')[-2]=='complex'
    wlist,rholist=_load_hybri(hybri_file,is_complex)
    rhofunc=interp1d(wlist,rholist,axis=0)

    #
    gap_neg=hybri_config['gap_neg']
    gap_pos=hybri_config['gap_pos']
    min_scale=hybri_config['min_scale']
    posmask=wlist>0
    wpos,wneg=wlist[posmask],wlist[~posmask]
    if wpos.min()>gap_pos+min_scale:
        wpos=append([gap_pos+min_scale],wpos)
    if wneg.max()<-gap_neg-min_scale:
        wneg=append(wneg,[-gap_neg-min_scale])
    wlist=append(wneg,wpos)
    
    #discretize to star model
    zs=array(disc_config['twisting_parameters'])
    discmodel=quick_map(rhofunc=rhofunc,wlist=wlist,N=disc_config['num_sites_star'],z=zs,Nx=disc_config['num_x'],
            tick_params={'tick_type':disc_config['energy_scaling'],'Lambda':disc_config['Lambda'],'Gap':[-gap_neg,gap_pos]},
            autofix=hybri_config['zero_rounding'])[1]
    discmodel.save(file_prefix=file_prefix)

    #tridiagonalize to a wilson chain.
    num_sites_wilson=trid_config['num_sites_wilson']
    if num_sites_wilson==-1: num_sites_wilson=disc_config['num_sites_star']
    chains=map2chain(discmodel,nsite=num_sites_wilson,normalize_method=trid_config['vector_normalization_method'])
    for z,chain in zip(zs,chains):
        chain.save(file_prefix=file_prefix+'_%s'%z)

    print '''%s
Mapping complete!
> Data for Wilson chains are stored in files `%s_[%s].[el,tl,info].dat`.
> To use them in python, check @chain.load_chain.
> To use them in other programming language,
  please read the description in method @chain.Chain.save.
%s'''%('='*40,file_prefix,','.join([str(z) for z in zs]),'='*40)


def _load_hybri(hybri_file,is_complex):
    data=loadtxt(hybri_file)
    wlist=data[:,0]
    rholist=data[:,1:]
    if is_complex:
        hdim=int(sqrt(rholist.shape[1]/2))
        rholist=rholist.reshape([-1,2]).view('complex128')
    else:
        hdim=int(sqrt(rholist.shape[1]))
    if hdim>1:
        rholist=rholist.reshape([len(wlist),hdim,hdim])
    else:
        rholist=rholist[:,0]
    #check and modify wlist
    if any(diff(wlist)<0):
        raise ValueError('omega not monotonious!')
    return wlist,rholist

if __name__=='__main__':
    if len(sys.argv)>1:
        config_file=sys.argv[1]
    else:
        config_file='config-sample.ini'
    run_config(config_file)
