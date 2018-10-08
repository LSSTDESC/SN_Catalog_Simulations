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
from astropy.table import vstack, Table, Column
import os
import h5py

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
    cosmology = w0waCDM(H0=cosmo_par['H0'],
                        Om0=cosmo_par['Omega_m'],
                        Ode0=cosmo_par['Omega_l'],
                        w0=cosmo_par['w0'], wa=cosmo_par['wa'])

    # load telescope
    tel_par = config['Instrument']
    telescope = Telescope(name=tel_par['name'],
                          throughput_dir=tel_par['throughput_dir'],
                          atmos_dir=tel_par['atmos_dir'],
                          atmos=tel_par['atmos'],
                          aerosol=tel_par['aerosol'],
                          airmass=tel_par['airmass'])
    # print(telescope.m5('ugrizy'))

    # load Observations

    obs_file = config['Observations']['dirname'] + \
        '/'+config['Observations']['filename']
    all_obs = Observations(filename=obs_file)

    save_status = config['Output']['save']
    fieldname = config['Observations']['fieldname']
    fieldid = config['Observations']['fieldid']
    season_obs = config['Observations']['season']
    # Output files - check dir ok and open files
    if save_status:
        # Check whether output directory exists
        outdir = config['Output']['directory']
        if not os.path.exists(outdir):
            print('Creating output directory', outdir)
            os.makedirs(outdir)
        # Two files to be opened (fieldname and fieldid
        # given in the input yaml file)
        # One containing a summary of the simulation:
        # astropy table with (SNID,Ra,Dec,X1,Color,z) parameters
        # -> name: Simu_fieldname_fieldid_season.hdf5
        # A second containing the Light curves (list of astropy tables)
        # -> name : LC_fieldname_fieldid_season.hdf5
        simu_out = outdir+'/Simu_'+fieldname+'_' + \
            str(fieldid)+'_'+str(season_obs)+'.hdf5'
        lc_out = outdir+'/LC_'+fieldname+'_' + \
            str(fieldid)+'_'+str(season_obs)+'.hdf5'
        # and these files will be removed now (before processing)
        # if they exist (to avoid confusions)
        if os.path.exists(simu_out):
            os.remove(simu_out)
        if os.path.exists(lc_out):
            os.remove(lc_out)

    # load SN_parameters and make a table of SN to simulate...
    # This simulation table is obs dependent because of DayMax choice
    sn_parameters = config['SN parameters']
    sn_meta = []
    for season in range(len(all_obs.seasons)):
        if season != season_obs and season_obs != -1:
            continue
        obs = all_obs.seasons[season]
        # remove the u band
        idx = [i for i, val in enumerate(obs['band']) if val[-1] != 'u']
        obs = obs[idx]
        # print('obs', sn_parameters)
        gen = Generate_Sample(sn_parameters, cosmo_par)
        gen_params = gen(obs)
        # gen.Plot_Parameters(gen_params)

        # load all parameters

        for i, val in enumerate(gen_params[:]):
            index_hdf5 = i+10000*season
            sn_par = sn_parameters.copy()
            for name in ['z', 'X1', 'Color', 'DayMax']:
                sn_par[name] = val[name]
            SNID = sn_par['Id']+index_hdf5
            sn_object = SN_Object(
                config['Simulator'], sn_par, cosmology, telescope, SNID)

            for simu_name in [config['Simulator']['name']]:
                module = import_module(simu_name)
                simu = module.SN(sn_object, config['Simulator'])
                # simulation
                obs_table = simu(obs, config['Display'])
                if save_status:
                    # write this table in the lc_out

                    obs_table.write(
                        lc_out, path='lc_'+str(index_hdf5),
                        append=True, compression=True)
                    # append the parameters in tab_out
                    m_lc = obs_table.meta
                    sn_meta.append((m_lc['SNID'], m_lc['Ra'],
                                    m_lc['Dec'], m_lc['DayMax'],
                                    m_lc['X1'], m_lc['Color'],
                                    m_lc['z'], index_hdf5, season))

    if len(sn_meta) > 0:
        print(sn_meta)
        Table(rows=sn_meta,
              names=['SNID', 'Ra', 'Dec', 'DayMax', 'X1', 'Color',
                     'z', 'id_hdf5', 'season'],
              dtype=('i4', 'f8', 'f8', 'f8', 'f8', 'f8',
                     'f8', 'i4', 'i4')).write(
                         simu_out, 'summary', compression=True)


def main(args):
    print('running')
    time_ref = time.time()
    run(args.config_filename)
    print('Time', time.time()-time_ref)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
