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
import os
from SN_Throughputs import Throughputs
from scipy import interpolate, integrate
import h5py

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
        self.model=model
        self.version=version

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
        lumidist=self.cosmology.luminosity_distance(self.sn_parameters['z']).value*1.e3
        X0= self.X0_norm()/ lumidist** 2
        #print('before alpha beta',X0)
        alpha=0.13
        beta=3.
        X0 *= np.power(10., 0.4*(alpha*self.sn_parameters['X1'] -beta*self.sn_parameters['Color']))

        self.X0=X0
        self.SN.set(x0=X0)
        """
        self.SN.set_source_peakabsmag(self.sn_parameters['absmag'], self.sn_parameters['band'], self.sn_parameters['magsys'])

        self.X0=self.SN.get('x0')
        """
        
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
        table_obs.add_column(Column(['LSST::'+obs['band'][i][-1]  for i in range(len(obs['band']))], name='band',dtype=h5py.special_dtype(vlen=str)))
        #table_obs.add_column(Column([obs['band'][i][-1] for i in range(len(obs['band']))], name='band'))
        table_obs.add_column(Column([2.5*np.log10(3631)]*len(obs),name='zp'))
        table_obs.add_column(Column(['ab']*len(obs),name='zpsys',dtype=h5py.special_dtype(vlen=str)))
        table_obs.add_column(Column(obs['mjd'],name='time'))
        
        idx = table_obs['flux'] >= 0.
        table_obs=table_obs[idx]
        ra=np.asscalar(np.unique(obs['Ra']))
        dec=np.asscalar(np.unique(obs['Dec']))
        table_obs.meta=dict(zip(['SNID','Ra','Dec','DayMax','X1','Color','z'],[self.SNID,ra,dec,self.sn_parameters['DayMax'],self.sn_parameters['X1'],self.sn_parameters['Color'],self.sn_parameters['z']]))
        #print(table_obs.dtype,table_obs['band'])
        if display:
            self.Plot_LC(table_obs['time','band','flux','fluxerr','zp','zpsys'])

        return table_obs
    def Plot_LC(self,table):
        import pylab as plt
        print('What to plot',table)
        prefix='LSST::'
        print(table.dtype)
        for band in 'ugrizy':
            name_filter=prefix+band
            if self.telescope.airmass > 0:                                                                                                      
                bandpass=sncosmo.Bandpass(self.telescope.atmosphere[band].wavelen,self.telescope.atmosphere[band].sb, name=name_filter,wave_unit=u.nm)                                                                                                                       
            else:                                                                                                                          
                bandpass=sncosmo.Bandpass(self.telescope.system[band].wavelen,self.telescope.system[band].sb, name=name_filter,wave_unit=u.nm)
            #print('registering',name_filter)
            sncosmo.registry.register(bandpass, force=True)
            
        model = sncosmo.Model('salt2')
        model.set(z=self.sn_parameters['z'], c=self.sn_parameters['Color'], t0=self.sn_parameters['DayMax'], x0=self.X0,x1=self.sn_parameters['X1'])
        sncosmo.plot_lc(data=table, model=model)
        plt.draw()
        plt.pause(1.)
        plt.close()

    def X0_norm(self):

        from lsst.sims.photUtils import Sed

        name='STANDARD'
        band='B'
        thedir='../SN_Utils/SALT2_Files'
        #thedir='.'

        os.environ[name] = thedir+'/Instruments/Landolt'

        trans_standard=Throughputs(through_dir='STANDARD',telescope_files=[],filter_files=['sb_-41A.dat'],atmos=False,aerosol=False,filterlist=('A'),wave_min=3559,wave_max=5559)
     
        mag, spectrum_file =self.Get_Mag(thedir+'/MagSys/VegaBD17-2008-11-28.dat',np.string_(name),np.string_(band))

        #print('alors mag',mag, thedir+'/'+spectrum_file)
        #sed=Sed()

        sourcewavelen,sourcefnu=self.readSED_fnu(filename=thedir+'/'+spectrum_file)
        CLIGHT_A_s  = 2.99792458e18         # [A/s]
        HPLANCK = 6.62606896e-27
       
        #sedb=Sed(wavelen=sed.wavelen,flambda=sed.wavelen*sed.fnu/(CLIGHT_A_s * HPLANCK))
        sedb=Sed(wavelen=sourcewavelen,flambda=sourcewavelen*sourcefnu/(CLIGHT_A_s * HPLANCK))
        
        flux=self.calcInteg(bandpass=trans_standard.system['A'],signal=sedb.flambda,wavelen=sedb.wavelen)

        zp=2.5*np.log10(flux)+mag
        flux_at_10pc = np.power(10., -0.4 * (self.sn_parameters['absmag']-zp))
        
        source=sncosmo.get_source(self.model,version=self.version)
        SN=sncosmo.Model(source=source)

        SN.set(z=0.)
        SN.set(t0=0)
        SN.set(c=self.sn_parameters['Color'])
        SN.set(x1=self.sn_parameters['X1'])
        SN.set(x0=1)

        fluxes=10.*SN.flux(0.,self.wave)
        
        wavelength=self.wave/10.
        SED_time = Sed(wavelen=wavelength, flambda=fluxes)

        expTime=30.
        photParams = PhotometricParameters(nexp=expTime/15.)
        trans=Bandpass(wavelen=trans_standard.system['A'].wavelen/10., sb=trans_standard.system['A'].sb)
        e_per_sec = SED_time.calcADU(bandpass=trans, photParams=photParams) #number of ADU counts for expTime
                    #e_per_sec = sed.calcADU(bandpass=self.transmission.lsst_atmos[filtre], photParams=photParams)
        e_per_sec/=expTime/photParams.gain*photParams.effarea
        #print 'hello',e_per_sec
        """
        SN.set(c=self.param['Color'])
        SN.set(x1=self.param['X1'])
        """

        
        #print 'My zp',zp,flux
        return flux_at_10pc * 1.E-4 /e_per_sec

    def Get_Mag(self,filename,name,band):

        #print('opening',filename)
        sfile=open(filename,'rb')
        spectrum_file='unknown'
        for line in sfile.readlines():
            if np.string_('SPECTRUM') in line:
                spectrum_file=line.decode().split(' ')[1].strip()
            if name in line and band in line:
                return float(line.decode().split(' ')[2]),spectrum_file

        sfile.close()

    def calcInteg(self, bandpass, signal,wavelen):
        
        fa = interpolate.interp1d(bandpass.wavelen,bandpass.sb)
        fb = interpolate.interp1d(wavelen,signal)
        
        min_wave=np.max([np.min(bandpass.wavelen),np.min(wavelen)])
        max_wave=np.min([np.max(bandpass.wavelen),np.max(wavelen)])
        
        wavelength_integration_step=5
        waves=np.arange(min_wave,max_wave,wavelength_integration_step)
        
        integrand=fa(waves) *fb(waves)
        
        range_inf=min_wave
        range_sup=max_wave
        n_steps = int((range_sup-range_inf) / wavelength_integration_step)

        x = np.core.function_base.linspace(range_inf, range_sup, n_steps)
        
        return integrate.simps(integrand,x=waves) 

    def readSED_fnu(self, filename, name=None):
        """
        Read a file containing [lambda Fnu] (lambda in nm) (Fnu in Jansky).

        Extracted from sims/photUtils/Sed.py which does not seem to work
        """
        # Try to open the data file.
        try:
            if filename.endswith('.gz'):
                f = gzip.open(filename, 'rt')
            else:
                f = open(filename, 'r')
        # if the above fails, look for the file with and without the gz
        except IOError:
            try:
                if filename.endswith(".gz"):
                    f = open(filename[:-3], 'r')
                else:
                    f = gzip.open(filename+".gz", 'rt')
            except IOError:
                raise IOError("The throughput file %s does not exist" % (filename))
        # Read source SED from file - lambda, fnu should be first two columns in the file.
        # lambda should be in nm and fnu should be in Jansky.
        sourcewavelen = []
        sourcefnu = []
        for line in f:
            if line.startswith("#"):
                continue
            values = line.split()
            sourcewavelen.append(float(values[0]))
            sourcefnu.append(float(values[1]))
        f.close()
        # Convert to numpy arrays.
        sourcewavelen = np.array(sourcewavelen)
        sourcefnu = np.array(sourcefnu)
        return sourcewavelen,sourcefnu
