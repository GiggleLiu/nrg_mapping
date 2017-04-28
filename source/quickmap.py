'''
Quick map from hybridization function to chain..
'''

from numpy import *

from discretization import quick_map
from chainmapper import map2chain
from scipy.interpolate import interp1d
from configobj import ConfigObj
from validate import Validator
import sys,pdb

__all__=['quickmap']

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

def run_config():
    '''Run according to the configuration file `config.py`.'''
    if len(sys.argv)>1:
        config_file=sys.argv[1]
    else:
        config_file='config-sample.ini'
    config=ConfigObj(config_file,configspec='config-spec.ini',stringify=True)
    validator = Validator()
    result = config.validate(validator,preserve_errors=True)
    if result!=True:
        raise ValueError('Configuration Error! %s'%result)
    hybri_config=config['hybri_func']
    disc_config=config['discretization']
    trid_config=config['tridiagonalization']
    file_prefix=config['global']['data_folder'].strip('/')+'/'+config['global']['data_token']

    #load hybridization function and do elementwise interpolation
    data=loadtxt(hybri_config['hybridization_func_file'])
    wlist=data[:,0]
    rholist=data[:,1:]
    if rholist.shape[1]>1:
        hdim=int(sqrt(rholist.shape[1]))
        rholist=rholist.reshape([rholist.shape[0],hdim,hdim])
    else:
        rholist=rholist[:,0]
    #check and modify wlist
    if any(diff(wlist)<0):
        raise ValueError('omega not monotonious!')
    rhofunc=interp1d(wlist,rholist,axis=0)

    #
    gap_neg=hybri_config['gap_neg']
    gap_pos=hybri_config['gap_pos']
    posmask=wlist>0
    wpos,wneg=wlist[posmask],wlist[~posmask]
    if wpos.min()>gap_pos+1e-15:
        wpos=append([gap_pos+1e-15],wpos)
    if wneg.max()<-gap_neg-1e-15:
        wneg=append(wneg,[-gap_neg-1e-15])
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
%s'''%('='*40,','.join([str(z) for z in zs]),file_prefix,'='*40)

if __name__=='__main__':
    run_config()
