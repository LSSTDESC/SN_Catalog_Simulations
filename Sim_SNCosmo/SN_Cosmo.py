import sncosmo
from SN_Object import SN_Object
import numpy as np
from lsst.sims.photUtils import Bandpass,Sed
from lsst.sims.photUtils import SignalToNoise
from lsst.sims.photUtils import PhotometricParameters
from astropy.table import vstack,Table,Column
import astropy.units as u
import matplotlib.animation as manimation
import pylab as plt

class SN(SN_Object):
    """ SN class - inherits from SN_Object
          Input parameters (as given in the input yaml file): 
          - SN parameters (X1, Color, DayMax, z, ...)
          - simulation parameters 

         Output: 
         - astropy table with the simulated light curve:
               - columns : band, flux, fluxerr, snr_m5,flux_e,zp,zpsys,time
               - metadata : SNID,Ra,Dec,DayMax,X1,Color,z

    """
    def __init__(self,param,simu_param):
        super().__init__(param.name,param.sn_parameters,param.cosmology,param.telescope,param.SNID)
    
        model=simu_param['model']
        version=str(simu_param['version'])
        #print('alors',model,version)

        if model == 'salt2-extended':
            model_min=300.
            model_max=180000.
            wave_min=3000.
            wave_max=11501.

        if model=='salt2':
            model_min=3400.
            model_max=11501.
            wave_min=model_min
            wave_max=model_max

        self.wave= np.arange(wave_min,wave_max,1.)
        
        source=sncosmo.get_source(model,version=version)

        dust = sncosmo.OD94Dust()

        self.SN=sncosmo.Model(source=source,effects=[dust, dust],
                         effect_names=['host', 'mw'],
                         effect_frames=['rest', 'obs'])
        
        self.SN.set(z=self.sn_parameters['z'])
        self.SN.set(t0=self.sn_parameters['DayMax'])
        self.SN.set(c=self.sn_parameters['Color'])
        self.SN.set(x1=self.sn_parameters['X1'])

        self.SN.set_source_peakabsmag(self.sn_parameters['absmag'], self.sn_parameters['band'], self.sn_parameters['magsys'])

        self.X0=self.SN.get('x0')

    def __call__(self,obs,display=False):
        """ Simulation of the light curve
              Input : a set of observations
              Output : astropy table with:
                            columns: band, flux, fluxerr, snr_m5,flux_e,zp,zpsys,time
                            metadata : SNID,Ra,Dec,DayMax,X1,Color,z
       """
        
        print('Simulating SNID',self.SNID)
        obs=self.cutoff(obs,self.sn_parameters['DayMax'],self.sn_parameters['z'],self.sn_parameters['min_rf_phase'],self.sn_parameters['max_rf_phase'])
        
        fluxes=10.*self.SN.flux(obs['mjd'],self.wave)
        
        wavelength=self.wave/10.
        
        wavelength=np.repeat(wavelength[np.newaxis,:], len(fluxes), 0)
        SED_time = Sed(wavelen=wavelength, flambda=fluxes)

        fluxes=[]
        transes=[]
        nvals=range(len(SED_time.wavelen))
        seds=[Sed(wavelen=SED_time.wavelen[i],flambda=SED_time.flambda[i]) for i in nvals]
        transes=[self.telescope.atmosphere[obs['band'][i][-1]] for i in nvals]
        fluxes=np.asarray([seds[i].calcFlux(bandpass=transes[i]) for i in nvals])
        photParams=[PhotometricParameters(nexp=obs['exptime'][i]/15.) for i in nvals]
        mag_SN=-2.5 * np.log10(fluxes / 3631.0)  #fluxes are in Jy
        calc=[SignalToNoise.calcSNR_m5(mag_SN[i],transes[i],obs['m5sigmadepth'][i],photParams[i]) for i in nvals]
        snr_m5_opsim=[calc[i][0] for i in nvals]
        #gamma_opsim=[calc[i][1] for i in nvals]
        e_per_sec = [seds[i].calcADU(bandpass=transes[i], photParams=photParams[i])/obs['exptime'][i]*photParams[i].gain for i in nvals]
        #table_obs=Table(obs)
        table_obs=Table()
        #table_obs.remove_column('band')
        table_obs.add_column(Column(fluxes, name='flux'))
        table_obs.add_column(Column(fluxes/snr_m5_opsim, name='fluxerr'))
        table_obs.add_column(Column(snr_m5_opsim, name='snr_m5'))
        table_obs.add_column(Column(e_per_sec, name='flux_e'))
        table_obs.add_column(Column([np.string_('LSST::'+obs['band'][i][-1]) for i in range(len(obs['band']))], name='band'))
        #table_obs.add_column(Column([obs['band'][i][-1] for i in range(len(obs['band']))], name='band'))
        table_obs.add_column(Column([2.5*np.log10(3631)]*len(obs),name='zp'))
        table_obs.add_column(Column([np.string_('ab')]*len(obs),name='zpsys'))
        table_obs.add_column(Column(obs['mjd'],name='time'))
        
        idx = table_obs['flux'] >= 0.
        table_obs=table_obs[idx]
        ra=np.asscalar(np.unique(obs['Ra']))
        dec=np.asscalar(np.unique(obs['Dec']))
        table_obs.meta=dict(zip(['SNID','Ra','Dec','DayMax','X1','Color','z'],[self.SNID,ra,dec,self.sn_parameters['DayMax'],self.sn_parameters['X1'],self.sn_parameters['Color'],self.sn_parameters['z']]))
        #print(table_obs.colnames)
        if display:
            self.Plot_LC(table_obs['time','band','flux','fluxerr','zp','zpsys'])

        return table_obs
    def Plot_LC(self,table):
        import pylab as plt
        print('What to plot',table)
        for band in 'ugrizy':                                                                                                            
            if self.telescope.airmass > 0:                                                                                                      
                bandpass=sncosmo.Bandpass(self.telescope.atmosphere[band].wavelen,self.telescope.atmosphere[band].sb, name='LSST::'+band,wave_unit=u.nm)                                                                                                                       
            else:                                                                                                                          
                bandpass=sncosmo.Bandpass(self.telescope.system[band].wavelen,self.telescope.system[band].sb, name='LSST::'+band,wave_unit=u.nm)                                                                                                                               
            sncosmo.registry.register(bandpass, force=True)
            
        model = sncosmo.Model('salt2')
        model.set(z=self.sn_parameters['z'], c=self.sn_parameters['Color'], t0=self.sn_parameters['DayMax'], x0=self.X0,x1=self.sn_parameters['X1'])
        sncosmo.plot_lc(data=table, model=model)
        plt.draw()
        plt.pause(1.)
        plt.close()
