import yaml
import argparse
from Telescope import Telescope
from astropy.cosmology import w0waCDM
from SN_Object import SN_Object
from importlib import import_module
from Observations import Observations
import numpy as np
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

    # load all parameters
    sn_object=SN_Object(config['Simulator'],config['SN parameters'],cosmology,telescope)
    print('booo',sn_object.name)

    # load Observations

    obs_file=config['Observations']['dirname']+'/'+config['Observations']['filename']
    obs=Observations(filename=obs_file).seasons[config['Observations']['season']]
    print('observbations',obs,obs_file)
    
    for simu_name in [config['Simulator']['name']]:
        module = import_module(simu_name)
        simu=module.SN(sn_object,config['Simulator'])
        # simulation
        #remove the u band
        idx = [i for i,val in enumerate(obs['band']) if val[-1]!= 'u']
        simu(obs[idx])
    
def main(args):
    print('running')
    run(args.config_filename)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
