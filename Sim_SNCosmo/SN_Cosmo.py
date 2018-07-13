import sncosmo
from SN_Object import SN_Object
import numpy as np
from lsst.sims.photUtils import Bandpass,Sed
from lsst.sims.photUtils import SignalToNoise
from lsst.sims.photUtils import PhotometricParameters
from astropy.table import vstack,Table,Column

class SN(SN_Object):
    def __init__(self,param,simu_param):
        super().__init__(param.name,param.sn_parameters,param.cosmology,param.telescope)
    
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

    def __call__(self,obs):
       
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
        table_obs=Table(obs)
        table_obs.add_column(Column(fluxes, name='flux'))
        table_obs.add_column(Column(snr_m5_opsim, name='snr_m5'))
        table_obs.add_column(Column(e_per_sec, name='flux_e'))
        idx = table_obs['flux'] >= 0.
        table_obs=table_obs[idx]
        print(table_obs)
        
   
