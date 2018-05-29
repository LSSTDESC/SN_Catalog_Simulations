import sncosmo
from params import Parameter
import numpy as np

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
        print('in call',self.name,len(obs))

        obs=self.cutoff(obs,self.sn_parameters['DayMax'],self.sn_parameters['z'],self.sn_parameters['min_rf_phase'],self.sn_parameters['max_rf_phase'])
        print('after cut ',self.name,len(obs))
        fluxes=10.*self.SN.flux(obs['mjd'],self.wave)
        
        wavelength=self.wave/10.
        
        wavelength=np.repeat(wavelength[np.newaxis,:], len(fluxes), 0)
        #SED_time = Sed(wavelen=wavelength, flambda=fluxes)

    def cutoff(self,obs,T0,z,min_rf_phase,max_rf_phase):
        blue_cutoff=300.
        red_cutoff=800.
        print('akakakakaka',len(obs))
        for obser in obs:
            print('ererrr',obser['band'][-1])
        mean_restframe_wavelength = np.asarray([self.telescope.mean_wavelength[obser['band'][-1]]/ (1. + z) for obser in obs])

        p=(obs['mjd']-T0)/(1.+z)
        print('hello phase',p,T0,z,obs['mjd'],mean_restframe_wavelength)
        idx = (p >= min_rf_phase)&(p<=max_rf_phase)&(mean_restframe_wavelength>blue_cutoff) & (mean_restframe_wavelength<red_cutoff)
        return obs[idx]
