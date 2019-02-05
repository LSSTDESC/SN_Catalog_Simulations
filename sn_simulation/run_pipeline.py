import yaml
import argparse
from SN_Simulation import SN_Simulation
import time
import numpy as np

parser = argparse.ArgumentParser(
    description='Run a SN simulation from a configuration file')
parser.add_argument('config_filename',
                    help='Configuration file in YAML format.')


def run(config_filename):
    # YAML input file.
    config = yaml.load(open(config_filename))
    print(config)

    # load cosmology
    cosmo_par = config['Cosmology']
    # load telescope
    tel_par = config['Instrument']

    # this is for output

    save_status = config['Output']['save']
    outdir = config['Output']['directory']
    prodid = config['ProductionID']
    # sn parameters
    sn_parameters = config['SN parameters']

    simu_config = config['Simulator']
    display_lc = config['Display']

    names = dict(zip(['band', 'mjd', 'rawSeeing', 'sky', 'exptime',
                      'moonPhase', 'Ra', 'Dec', 'Nexp', 'fiveSigmaDepth',
                      'seeing', 'airmass', 'night', 'season', 'pixarea',
                      'pixRa', 'pixDec'],
                     ['band', 'mjd', 'seeingFwhm500', 'sky', 'exptime',
                      'moonPhase', 'Ra', 'Dec', 'numExposures',
                      'fiveSigmaDepth', 'seeingFwhmEff', 'airmass',
                      'night', 'season', 'pixarea',
                     'pixRa', 'pixDec']))

    simu = SN_Simulation(cosmo_par, tel_par, sn_parameters,
                    save_status, outdir, prodid,
                    simu_config, display_lc, names=names,nproc=config['Multiprocessing']['nproc'])

    # load input file (.npy)

    input_name = config['Observations']['dirname'] + \
        '/'+config['Observations']['filename']
    print('loading', input_name)
    input_data = np.load(input_name)

    print(input_data.dtype)

    toprocess = np.unique(input_data[['fieldname', 'fieldid']])
    print('Number of fields to simulate', len(toprocess))
    for (fieldname, fieldid) in toprocess:
        idx = (input_data['fieldname'] == fieldname) & (
            input_data['fieldid'] == fieldid)
        print('Simulating',fieldname,fieldid)
        simu(input_data[idx],fieldname,fieldid)
        
    simu.Finish()


def main(args):
    print('running')
    time_ref = time.time()
    run(args.config_filename)
    print('Time', time.time()-time_ref)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
