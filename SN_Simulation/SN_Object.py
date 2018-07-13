from SN_Telescope import Telescope
import numpy as np

class SN_Object:
    def __init__(self, name, sn_parameters, cosmology, Telescope):
        #print('there we go',name)
        self._name=name
        self._sn_parameters=sn_parameters
        self._cosmology=cosmology
        self._telescope=Telescope

    @property
    def name(self):
        return self._name
    
    @property
    def sn_parameters(self):
        return self._sn_parameters
    
    @property
    def cosmology(self):
        return self._cosmology
    
    @property
    def telescope(self):
        return self._telescope

    def cutoff(self,obs,T0,z,min_rf_phase,max_rf_phase):
        blue_cutoff=300.
        red_cutoff=800.
        
        mean_restframe_wavelength = np.asarray([self.telescope.mean_wavelength[obser['band'][-1]]/ (1. + z) for obser in obs])

        p=(obs['mjd']-T0)/(1.+z)
        
        idx = (p >= min_rf_phase)&(p<=max_rf_phase)&(mean_restframe_wavelength>blue_cutoff) & (mean_restframe_wavelength<red_cutoff)
        return obs[idx]
