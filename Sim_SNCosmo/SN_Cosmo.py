import sncosmo
from params import Parameter
import numpy as np
from lsst.sims.photUtils import Bandpass,Sed
from lsst.sims.photUtils import SignalToNoise
from lsst.sims.photUtils import PhotometricParameters
from astropy.table import vstack,Table,Column

class SN(Parameter):
    def __init__(self,param,simu_param):
        super().__init__(param.name,param.sn_parameters,param.cosmology,param.telescope)
    
        model=simu_param['model']
        version=str(simu_param['version'])
        print('alors',model,version)

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
        seds=[Sed(wavelen=SED_time.wavelen[i],flambda=SED_time.flambda[i]) for i in range(len(SED_time.wavelen))]
        transes=[self.telescope.atmosphere[obs['band'][i][-1]] for i in range(len(SED_time.wavelen))]
        fluxes=[seds[i].calcFlux(bandpass=transes[i]) for i in range(len(SED_time.wavelen))]
        
        table_obs=Table(obs)
        snr_m5=[]
        e_per_sec_list=[]
        for i in range(len(SED_time.wavelen)):
            photParams=PhotometricParameters(nexp=table_obs['exptime'][i]/15.)
            flux_SN=fluxes[i]
            if flux_SN > 0:
                trans=self.telescope.atmosphere[table_obs['band'][i][-1]]
                mag_SN=-2.5 * np.log10(flux_SN / 3631.0)  
                snr_m5_opsim,gamma_opsim=SignalToNoise.calcSNR_m5(mag_SN,trans,table_obs['m5sigmadepth'][i],photParams)
                err_flux_SN=flux_SN/snr_m5_opsim
                e_per_sec = seds[i].calcADU(bandpass=trans, photParams=photParams) #number of ADU counts for expTime
                e_per_sec/=table_obs['exptime'][i]/photParams.gain
                snr_m5.append(snr_m5_opsim)
                e_per_sec_list.append(e_per_sec)
            else:
                snr_m5.append(1)
                e_per_sec_list.append(1)

        #print('passed')
        
        table_obs.add_column(Column(fluxes, name='flux'))
        table_obs.add_column(Column(snr_m5, name='snr_m5'))
        table_obs.add_column(Column(e_per_sec_list, name='flux_e'))
        idx = table_obs['flux'] >= 0.
        table_obs=table_obs[idx]

        print(table_obs)
        
    def cutoff(self,obs,T0,z,min_rf_phase,max_rf_phase):
        blue_cutoff=300.
        red_cutoff=800.
        
        mean_restframe_wavelength = np.asarray([self.telescope.mean_wavelength[obser['band'][-1]]/ (1. + z) for obser in obs])

        p=(obs['mjd']-T0)/(1.+z)
        
        idx = (p >= min_rf_phase)&(p<=max_rf_phase)&(mean_restframe_wavelength>blue_cutoff) & (mean_restframe_wavelength<red_cutoff)
        return obs[idx]
