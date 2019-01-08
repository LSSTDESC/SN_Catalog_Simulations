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
import multiprocessing

class SN_Simulation:
    """ Main class for simulation
    Input (init)
    ---------
    - cosmo_par: cosmology parameters
    - tel_par: telescope parameters
    - sn_parameters: SN parameters
    - save_status: to save (True) or not (False)
    generated quantities
    - outdir: output directory
    - prodid: production id
    - simu_config: Simulator configuration
    - display_lc: to display (True) or not (False)
    the light curves during production
    - set of names: names of some variable used in Observation data
    - nproc: number of multiprocess

    Returns
    ---------
    - call :
    LC (hdf5)
    - Finish:
    Summary of production (hdf5)
    """

    def __init__(self, cosmo_par, tel_par, sn_parameters,
                 save_status, outdir, prodid,
                 simu_config, display_lc,time_display,area,
                  mjdCol='mjd', RaCol='pixRa', DecCol='pixDec',
                 filterCol='band', exptimeCol='exptime',
                 m5Col='fiveSigmaDepth', seasonCol='season',
                 nproc=1):

        # self.cosmo_par = cosmo_par
        self.sn_parameters = sn_parameters
        self.simu_config = simu_config
        self.display_lc = display_lc
        self.time_display = time_display
        self.index_hdf5 = 100
        self.save_status = save_status
        self.mjdCol = mjdCol
        self.RaCol = RaCol
        self.DecCol = DecCol
        self.filterCol = filterCol
        self.exptimeCol = exptimeCol
        self.m5Col = m5Col
        self.seasonCol = seasonCol
        self.nproc = nproc
        self.area= area
        self.gen_par = Generate_Sample(sn_parameters, cosmo_par,mjdCol=self.mjdCol, area = self.area)
        
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
            self.Prepare_Save(outdir, prodid)

    def Prepare_Save(self, outdir, prodid):

        if not os.path.exists(outdir):
            print('Creating output directory', outdir)
            os.makedirs(outdir)
        # Two files to be opened (fieldname and fieldid
        # given in the input yaml file)
        # One containing a summary of the simulation:
        # astropy table with (SNID,Ra,Dec,X1,Color,z) parameters
        # -> name: Simu_prodid.hdf5
        # A second containing the Light curves (list of astropy tables)
        # -> name : LC_prodid.hdf5
        self.simu_out = outdir+'/Simu_'+prodid+'.hdf5'
        self.lc_out = outdir+'/LC_'+prodid+'.hdf5'
        self.sn_meta = []
        # and these files will be removed now (before processing)
        # if they exist (to avoid confusions)
        if os.path.exists(self.simu_out):
            os.remove(self.simu_out)
        if os.path.exists(self.lc_out):
            os.remove(self.lc_out)
            
    def __call__(self, obs,fieldname,fieldid,season):

        #obs = Observations(data=tab, names=self.names)
        self.fieldname = fieldname
        self.fieldid = fieldid
        if season == -1:
            seasons = np.unique(obs[self.seasonCol])
        else:
            seasons = [season]
        print('number of seasons',len(seasons))
        if self.simu_config['name'] != 'SN_Fast':
            for seas in seasons:
                time_ref = time.time()
                idxa = obs[self.seasonCol] == seas
                obs_season = obs[idxa]
                # remove the u band
                idx = [i for i, val in enumerate(obs_season[self.filterCol]) if val[-1] != 'u']
                if len(obs_season[idx]) > 0:
                    self.Process_Season(obs_season[idx], seas)
        else:
            self.Process_Fast(obs)
        print('End of simulation',time.time()-time_ref)
    """
    def __call__(self, tab,fieldname,fieldid):

        all_obs = Observations(data=tab, names=self.names)
        self.fieldname = fieldname
        self.fieldid = fieldid
        print('number of seasons',len(all_obs.seasons))
        for season in range(len(all_obs.seasons)):
            time_ref = time.time()
            obs = all_obs.seasons[season]
            # remove the u band
            idx = [i for i, val in enumerate(obs[self.filterCol]) if val[-1] != 'u']
            if len(obs[idx]) > 0:
                self.Process_Season(obs[idx], season)
            print('End of simulation',time.time()-time_ref)
    """        
    def Process_Season(self, obs, season):

        gen_params = self.gen_par(obs)
        if gen_params is None:
            return
        nlc= len(gen_params)
        batch = range(0,nlc,self.nproc)
        batch = np.append(batch,nlc)

        for i in range(len(batch)-1):
            result_queue = multiprocessing.Queue()
        
            ida = batch[i]
            idb = batch[i+1]

            for j in range(ida,idb):
                self.index_hdf5 += 1
                p=multiprocessing.Process(name='Subprocess-'+str(j),target=self.Process_Season_Single,args=(obs,season,gen_params[j],self.index_hdf5,j,result_queue))
                p.start()

            resultdict = {}
            for j in range(ida,idb):
                resultdict.update(result_queue.get())
        
            for p in multiprocessing.active_children():
                p.join()

            for j in range(ida,idb):
                
                if self.save_status:
                    metadata = resultdict[j][1]
                    n_lc_points = 0
                    if resultdict[j][0] is not None:
                        n_lc_points = len(resultdict[j][0])
                        resultdict[j][0].write(self.lc_out,
                                            path='lc_'+str(metadata['index_hdf5']),
                                            append=True,
                                            compression=True)
                    self.sn_meta.append((metadata['SNID'], metadata['Ra'],
                                         metadata['Dec'], metadata['DayMax'],
                                         metadata['X1'], metadata['Color'],
                                         metadata['z'], metadata['index_hdf5'], season,
                                         self.fieldname, self.fieldid,
                                         n_lc_points,metadata['survey_area']))
            """
            for i, val in enumerate(gen_params[:]):
            self.index_hdf5 += 1
            self.Process_Season_Single(obs,season,val)
            """
    def Process_Season_Single(self,obs,season,gen_params,index_hdf5,j=-1,output_q=None):
        sn_par = self.sn_parameters.copy()
        for name in ['z', 'X1', 'Color', 'DayMax']:
            sn_par[name] = gen_params[name]
        SNID = sn_par['Id']+index_hdf5
        sn_object = SN_Object(self.simu_config['name'],
                              sn_par,
                              self.cosmology,
                              self.telescope, SNID,self.area,
                              mjdCol=self.mjdCol, RaCol=self.RaCol,
                              DecCol= self.DecCol, 
                              filterCol=self.filterCol, exptimeCol=self.exptimeCol,
                              m5Col=self.m5Col)

        module = import_module(self.simu_config['name'])
        simu = module.SN(sn_object, self.simu_config)
        # simulation
        lc_table, metadata = simu(obs, index_hdf5,self.display_lc,self.time_display)

        if output_q is not None:
            output_q.put({j : (lc_table,metadata)})

           
    def Finish(self):
        if len(self.sn_meta) > 0:
            Table(rows=self.sn_meta,
                  names=['SNID', 'Ra', 'Dec', 'DayMax', 'X1',
                         'Color', 'z', 'id_hdf5', 'season',
                         'fieldname','fieldid',
                         'n_lc_points','survey_area'],
                  dtype=('i4', 'f8', 'f8', 'f8', 'f8', 'f8',
                         'f8', 'i4', 'i4','S3','i8','i8','f8')).write(
                             self.simu_out, 'summary', compression=True)

    def Process_Fast(self, obs):
        
        sn_par = self.sn_parameters.copy()
        """
        for name in ['z', 'X1', 'Color', 'DayMax']:
            sn_par[name] = gen_params[name]
        """
        gen_params = self.gen_par(obs)
        SNID = sn_par['Id']+self.index_hdf5
        sn_object = SN_Object(self.simu_config['name'],
                              gen_params,
                              self.cosmology,
                              self.telescope, SNID,self.area,
                              mjdCol=self.mjdCol, RaCol=self.RaCol,
                              DecCol= self.DecCol, 
                              filterCol=self.filterCol, exptimeCol=self.exptimeCol,
                              m5Col=self.m5Col)
        
        module = import_module(self.simu_config['name'])
        simu = module.SN(sn_object, self.simu_config)
        simu(obs, self.index_hdf5,self.display_lc,self.time_display,gen_params)
        
