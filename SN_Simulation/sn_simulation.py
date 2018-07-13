import yaml
import argparse
from SN_Telescope import Telescope
from astropy.cosmology import w0waCDM
from SN_Object import SN_Object
from SN_Utils import Generate_Sample
from importlib import import_module
from Observations import Observations
import numpy as np
import time

parser = argparse.ArgumentParser(description='Run a SN simulation from a configuration file')
parser.add_argument('config_filename', help='Configuration file in YAML format.')

def run(config_filename):
    # YAML input file.
    config = yaml.load(open(config_filename))
    print(config)
   
    # load cosmology
    cosmo_par=config['Cosmology']
    cosmology=w0waCDM(H0=cosmo_par['H0'],Om0=cosmo_par['Omega_m'],Ode0=cosmo_par['Omega_l'],w0=cosmo_par['w0'],wa=cosmo_par['wa']) 

    #load telescope
    tel_par=config['Instrument']
    telescope=Telescope(name=tel_par['name'],throughput_dir=tel_par['throughput_dir'],atmos_dir=tel_par['atmos_dir'],atmos=tel_par['atmos'],aerosol=tel_par['aerosol'],airmass=tel_par['airmass'])
    print(telescope.m5('ugrizy'))

     # load Observations

    obs_file=config['Observations']['dirname']+'/'+config['Observations']['filename']
    obs=Observations(filename=obs_file).seasons[config['Observations']['season']]
    idx = [i for i,val in enumerate(obs['band']) if val[-1]!= 'u']
    obs=obs[idx]
    print('observations',obs,obs_file)

    #load SN_parameters and make a table of SN to simulate...
    sn_parameters=config['SN parameters']
   
    gen=Generate_Sample(sn_parameters,cosmo_par)
    gen_params=gen(obs)
    #gen.Plot_Parameters(gen_params)
    

    # load all parameters
    for val in gen_params:
        sn_par=sn_parameters
        for name in ['z','X1','Color','DayMax']:
            sn_par[name]=val[name]
        sn_object=SN_Object(config['Simulator'],sn_par,cosmology,telescope)

        for simu_name in [config['Simulator']['name']]:
            module = import_module(simu_name)
            simu=module.SN(sn_object,config['Simulator'])
            # simulation
            #remove the u band
            simu(obs)
    
def main(args):
    print('running')
    time_ref=time.time()
    run(args.config_filename)
    print('Time',time.time()-time_ref)
    
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
