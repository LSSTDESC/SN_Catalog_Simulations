from SN_Telescope import Telescope
import numpy as np


class SN_Object:
    """ class SN object
    handles sn name, parameters,
    cosmology, snid, telescope...
    necessary parameters for simulation
    SN classes inherit from SN_Object
    """
    def __init__(self, name, sn_parameters, cosmology, Telescope, snid):
        self._name = name
        self._sn_parameters = sn_parameters
        self._cosmology = cosmology
        self._telescope = Telescope
        self._SNID = snid

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

    @property
    def SNID(self):
        return self._SNID

    def cutoff(self, obs, T0, z, min_rf_phase, max_rf_phase):
        """ select observations depending on phases
        Input
        ---------
        obs: observations (recarray)
        T0: DayMax of the supernova
        z: redshift
        min_rf_phase: min phase rest-frame
        max_rf_phase: max phase rest-frame

        Returns
        ---------
        recarray of obs passing the selection
        """

        blue_cutoff = 300.
        red_cutoff = 800.

        mean_restframe_wavelength = np.asarray(
            [self.telescope.mean_wavelength[obser['band'][-1]] /
             (1. + z) for obser in obs])

        p = (obs['mjd']-T0)/(1.+z)

        idx = (p >= min_rf_phase) & (p <= max_rf_phase)
        idx &= (mean_restframe_wavelength > blue_cutoff)
        idx &= (mean_restframe_wavelength < red_cutoff)
        return obs[idx]
