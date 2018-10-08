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

class Simu_All:

    def __init__(self, cosmo_par, tel_par,sn_parameters,
                 save_status,outdir,prodid,
                 simu_config, display_lc,names):

        self.cosmo_par = cosmo_par
        self.sn_parameters = sn_parameters
        self.simu_config = simu_config
        self.display_lc = display_lc
        self.gen_par = Generate_Sample(sn_parameters, cosmo_par)
        self.index_hdf5 = 100
        self.save_status = save_status
        self.names = names
        self.cosmology = w0waCDM(H0=cosmo_par['H0'],
                                 Om0=cosmo_par['Omega_m'],
                                 Ode0=cosmo_par['Omega_l'],
                                 w0=cosmo_par['w0'], wa=cosmo_par['wa'])

        self.telescope = Telescope(name=tel_par['name'],
                                    throughput_dir=tel_par['throughput_dir'],
                                    atmos_dir=tel_par['atmos_dir'],
                                    atmos=tel_par['atmos'],
                                    aerosol=tel_par['aerosol'],
                                    airmass=tel_par['airmass'])

        if self.save_status:
            self.Prepare_Save(outdir,prodid)
           
    def Prepare_Save(self,outdir,prodid):

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
        self.simu_out = outdir+'/Simu_'+prodid+'.hdf5'
        self.lc_out = outdir+'/LC_'+prodid+'.hdf5'
        self.sn_meta=[]
        # and these files will be removed now (before processing)
        # if they exist (to avoid confusions)
        if os.path.exists(self.simu_out):
            os.remove(self.simu_out)
        if os.path.exists(self.lc_out):
            os.remove(self.lc_out)

    def __call__(self, tab):

        all_obs = Observations(data=tab,names=self.names)
        for season in range(len(all_obs.seasons)):
            obs = all_obs.seasons[season]
            # remove the u band
            idx = [i for i, val in enumerate(obs['band']) if val[-1] != 'u']
            self.Process_Season(obs[idx],season)
           
        
    def Process_Season(self, obs,season):
        
        gen_params = self.gen_par(obs)
        for i, val in enumerate(gen_params[:]):
            self.index_hdf5 += 1
            sn_par = self.sn_parameters.copy()
            for name in ['z', 'X1', 'Color', 'DayMax']:
                sn_par[name] = val[name]
            SNID = sn_par['Id']+self.index_hdf5
            sn_object = SN_Object(
                self.simu_config, sn_par, self.cosmology, self.telescope, SNID)

            module = import_module(self.simu_config['name'])
            simu = module.SN(sn_object, self.simu_config)
            # simulation
            obs_table = simu(obs, self.display_lc)
            if self.save_status:
                # write this table in the lc_out

                obs_table.write(
                    self.lc_out, path='lc_'+str(self.index_hdf5),
                    append=True, compression=True)
                # append the parameters in tab_out
                m_lc = obs_table.meta
                self.sn_meta.append((m_lc['SNID'], m_lc['Ra'],
                                     m_lc['Dec'], m_lc['DayMax'],
                                     m_lc['X1'], m_lc['Color'],
                                     m_lc['z'], self.index_hdf5, season))


    def Finish(self):
        if len(self.sn_meta) > 0:
            Table(rows=sn_meta,
                  names=['SNID', 'Ra', 'Dec', 'DayMax', 'X1',
                         'Color','z', 'id_hdf5', 'season'],
                  dtype=('i4', 'f8', 'f8', 'f8', 'f8', 'f8',
                         'f8', 'i4', 'i4')).write(
                             self.simu_out, 'summary', compression=True)

            
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

    names=dict(zip(['band','mjd','rawSeeing','sky','exptime','moonPhase','Ra','Dec','Nexp','fiveSigmaDepth','seeing','airmass','night','season'],['band','mjd','seeingFwhm500','sky','exptime','moonPhase','Ra','Dec','numExposures','fiveSigmaDepth','seeingFwhmEff','airmass','night','season']))
    
    simu = Simu_All(cosmo_par, tel_par,sn_parameters,
                    save_status,outdir,prodid,
                    simu_config, display_lc,names=names)
    
    # load input file (.npy)

    input_name = config['Observations']['dirname'] + \
                 '/'+config['Observations']['filename']
    print('loading',input_name)
    input_data = np.load(input_name)
    
    print(input_data.dtype)

 

    for (fieldname,fieldid) in np.unique(input_data[['fieldname','fieldid']]):
        idx = (input_data['fieldname'] == fieldname) & (input_data['fieldid'] == fieldid) 
        simu(input_data[idx])

    simu.Finish()

def main(args):
    print('running')
    time_ref = time.time()
    run(args.config_filename)
    print('Time', time.time()-time_ref)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
