from SN_Telescope import Telescope
import numpy as np
import astropy.units as u
from astropy.table import Table
from collections import OrderedDict as odict

class SN_Object:
    """ class SN object
    handles sn name, parameters,
    cosmology, snid, telescope...
    necessary parameters for simulation
    SN classes inherit from SN_Object
    """
    def __init__(self, name, sn_parameters, cosmology,
                 Telescope, snid,area,
                 mjdCol='mjd', RaCol = 'pixRa', DecCol = 'pixDec',
                 filterCol='band', exptimeCol = 'exptime',
                 m5Col = 'fiveSigmaDepth', seasonCol='season'):
        
        self._name = name
        self._sn_parameters = sn_parameters
        self._cosmology = cosmology
        self._telescope = Telescope
        self._SNID = snid

        self.mjdCol = mjdCol
        self.RaCol = RaCol
        self.DecCol = DecCol
        self.filterCol = filterCol
        self.exptimeCol = exptimeCol
        self.m5Col = m5Col
        self.seasonCol = seasonCol
        self.area = area

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
            [self.telescope.mean_wavelength[obser[self.filterCol][-1]] /
             (1. + z) for obser in obs])

        p = (obs[self.mjdCol]-T0)/(1.+z)

        idx = (p >= min_rf_phase) & (p <= max_rf_phase)
        idx &= (mean_restframe_wavelength > blue_cutoff)
        idx &= (mean_restframe_wavelength < red_cutoff)
        return obs[idx]

    def Plot_LC(self, table, time_display,z,DayMax,season):
        import pylab as plt
        import sncosmo
        print('What will be plotted')
        print(table)
        print(self.sn_parameters)
        prefix = 'LSST::'
        print(table.dtype,len(table))
        _photdata_aliases = odict([
            ('time', set(['time', 'date', 'jd', 'mjd', 'mjdobs', 'mjd_obs'])),
            ('band', set(['band', 'bandpass', 'filter', 'flt'])),
            ('flux', set(['flux', 'f'])),
            ('fluxerr', set(['fluxerr', 'fe', 'fluxerror', 'flux_error', 'flux_err'])),
            ('zp', set(['zp', 'zpt', 'zeropoint', 'zero_point'])),
            ('zpsys', set(['zpsys', 'zpmagsys', 'magsys']))
        ])
        for band in 'grizy':
            name_filter = prefix+band
            if self.telescope.airmass > 0:
                bandpass = sncosmo.Bandpass(
                    self.telescope.atmosphere[band].wavelen,
                    self.telescope.atmosphere[band].sb,
                    name=name_filter,
                    wave_unit=u.nm)
            else:
                bandpass = sncosmo.Bandpass(
                    self.telescope.system[band].wavelen,
                    self.telescope.system[band].sb,
                    name=name_filter,
                    wave_unit=u.nm)
            # print('registering',name_filter)
            sncosmo.registry.register(bandpass, force=True)

        model = sncosmo.Model('salt2')
        model.set(z=z,
                  c=np.unique(self.sn_parameters['Color']),
                  t0=DayMax,
                  #x0=self.X0,
                  x1=np.unique(self.sn_parameters['X1']))
        """
        print('tests',isinstance(table, np.ndarray),isinstance(table,Table),isinstance(table,dict))
        array_tab = np.asarray(table)
        print(array_tab.dtype)
        colnames = array_tab.dtype.names
        # Create mapping from lowercased column names to originals
        lower_to_orig = dict([(colname.lower(), colname) for colname in colnames])
        
        # Set of lowercase column names
        lower_colnames = set(lower_to_orig.keys())
        orig_colnames_to_use = []
        for aliases in _photdata_aliases.values():
            i = lower_colnames & aliases
            if len(i) != 1:
                raise ValueError('Data must include exactly one column from {0} '
                                 '(case independent)'.format(', '.join(aliases)))
            orig_colnames_to_use.append(lower_to_orig[i.pop()])

        
        new_data = table[orig_colnames_to_use].copy()
        print('bbbb',orig_colnames_to_use,_photdata_aliases.keys(),new_data.dtype.names)
        new_data.dtype.names = _photdata_aliases.keys()
        """
        print(table['band'])
        for vat in table['band']:
            print('oo',vat,'oo')
        sncosmo.plot_lc(data=table, model=model)
        
        plt.draw()
        plt.pause(time_display)
        plt.close()
        
