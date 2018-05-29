import yaml
import argparse
#from SN_Cosmo import SN_Cosmo
#from SN_Sim import SN_Sim
#from SN_Ana import SN_Ana
from Telescope import Telescope
from astropy.cosmology import w0waCDM
from params import Parameter
from importlib import import_module

parser = argparse.ArgumentParser(description='Run a SN simulation from a configuration file')
parser.add_argument('config_filename', help='Configuration file in YAML format.')

def run(config_filename):
    # YAML input file.
    config = yaml.load(open(config_filename))
    print(config)
    """
    print(config['SN parameters']['c'])
    if config['SN parameters']['c'] == 'random':
        print('yes')
    """
    #exec(config['Simulator']+'('+config['Simulator']+','+str(config['SN parameters'])+','+str(config['Cosmology'])+')')

    # load cosmology
    cosmo_par=config['Cosmology']
    cosmology=w0waCDM(H0=cosmo_par['H0'],Om0=cosmo_par['Omega_m'],Ode0=cosmo_par['Omega_l'],w0=cosmo_par['w0'],wa=cosmo_par['wa']) 

    #load telescope
    tel_par=config['Instrument']
    telescope=Telescope(name=tel_par['name'],throughput_dir=tel_par['throughput_dir'],atmos_dir=tel_par['atmos_dir'],atmos=tel_par['atmos'],aerosol=tel_par['aerosol'],airmass=tel_par['airmass'])
    print(telescope.m5('ugrizy'))

    # load all parameters
    param=Parameter(config['Simulator'],config['SN parameters'],cosmology,telescope)
    print('booo',param.name)
    
    simu_name=config['Simulator']
    module = import_module(simu_name)
    simu=module.SN(param)
    simu()
    """
    func = getattr(module, param)
    func()
    """
    """
    cmd=config['Simulator']+'('+str(param)+')'
    print(cmd)
    exec(config['Simulator']+'('+str(param)+')')
    """
def main(args):
    print('running')
    run(args.config_filename)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
