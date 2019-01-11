import numpy as np
from Observations import *
from scipy import interpolate
from astropy.table import Table,Column,vstack,hstack
import glob
import h5py
import pylab as plt
from scipy.spatial import distance
import time
import multiprocessing
from optparse import OptionParser
import os
from scipy.interpolate import griddata
from SN_Object import SN_Object
import numpy.lib.recfunctions as rf
import scipy.linalg.lapack as lapack
#from scipy.sparse import coo_matrix, block_diag,csc_matrix
#from scipy.linalg import pinvh

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
    def __init__(self, param, simu_param):
        super().__init__(param.name, param.sn_parameters,
                         param.cosmology, param.telescope, param.SNID,param.area,
                         mjdCol=param.mjdCol, RaCol=param.RaCol, DecCol=param.DecCol,
                         filterCol=param.filterCol, exptimeCol=param.exptimeCol,
                         m5Col=param.m5Col, seasonCol=param.seasonCol)
        
        X1=self.sn_parameters['X1']
        Color=self.sn_parameters['Color']
        zvals=[self.sn_parameters['z']]
        print('loading ref',np.unique(zvals),param)
        lc_ref_tot=Load_Reference(simu_param['Reference File'],np.unique(zvals)).tab

        # This cutoffs are used to select observations:
        # phase = (mjd - DayMax)/(1.+z)
        # selection: min_rf_phase < phase < max_rf_phase
        # and        blue_cutoff < mean_rest_frame < red_cutoff
        # where mean_rest_frame = telescope.mean_wavelength/(1.+z)
        self.blue_cutoff = 300.
        self.red_cutoff = 800.

        # Load references needed for the following
        self.lc_ref={}
        self.gamma_ref={}
        self.m5_ref={}
        self.mag_to_flux_e_sec = {}

        #print(lc_ref_tot.dtype)
        self.param_Fisher = ['X0','X1','Color']
        bands = np.unique(lc_ref_tot['band'])
        mag_range = np.arange(14.,32.,0.1)
        for band in bands:
            
            idx = lc_ref_tot['band']==band
            lc_sel=lc_ref_tot[idx]
            #print('hello',lc_sel.dtype)
            self.lc_ref[band] = lc_sel
            self.gamma_ref[band]=lc_sel['gamma'][0]
            self.m5_ref[band]=np.unique(lc_sel['m5'])[0]
            fluxes_e_sec = self.telescope.mag_to_flux_e_sec(mag_range,[band]*len(mag_range),[30]*len(mag_range))
            self.mag_to_flux_e_sec[band] = interpolate.interp1d(mag_range,fluxes_e_sec[:,1],fill_value = 0., bounds_error = False)

    
    def __call__(self, obs, index_hdf5,display=False, time_display=0.,gen_par=None):
        """ Simulation of the light curve

        Input
        ---------
        a set of observations


        Returns
        ---------
        astropy table with:
        columns: band, flux, fluxerr, snr_m5,flux_e,zp,zpsys,time
        metadata : SNID,Ra,Dec,DayMax,X1,Color,z
        """
        assert (len(np.unique(obs[self.RaCol])) == 1)
        assert (len(np.unique(obs[self.DecCol])) == 1)
        ra = np.asscalar(np.unique(obs[self.RaCol]))
        dec = np.asscalar(np.unique(obs[self.DecCol]))
        area = self.area
        self.index_hdf5 = index_hdf5
        
        # print('Simulating SNID', self.SNID)
        print('sn params',self.sn_parameters.dtype,self.sn_parameters['DayMax'],self.sn_parameters['z'],gen_par.dtype)
       
        if len(obs) == 0:
            return ra,dec,None

        result_queue = multiprocessing.Queue()
        #bands = 'grizy'
        bands = 'i'
        
        for j,band in enumerate(bands):
            idx = obs[self.filterCol] == band
            p=multiprocessing.Process(name='Subprocess-'+str(j),target=self.Process_band,args=(obs[idx],band,gen_par,j,result_queue))
            p.start()
             
        resultdict = {}
        for j,band in enumerate(bands):
            resultdict.update(result_queue.get())
		
        for p in multiprocessing.active_children():
            p.join()
                
       
        #tab_tot = vstack([resultdict[j] for j,band in enumerate(bands) if resultdict[j] is not None])
        #tab_tot = Table()
        tab_tot = [resultdict[j] for j,band in enumerate(bands) if resultdict[j] is not None]

        # There is a trick here
        # Usually one would just use vstack to get one astropy Table
        # But vstack seems to trigger problems with special h5py dype
        # namely h5py.special_dtype(vlen=str)))
        # so the idea is to convert the Tables to list,
        # to add the lists,
        # and then recreate an astropy table.
        
        r = []
        print('len tab_tot',len(tab_tot))
        if len(tab_tot) == 0:
            return ra,dec,None
        for val in tab_tot:
            valb = np.asarray(val)
            r+=np.ndarray.tolist(valb)
            dtype = val.dtype
            names = val.colnames
        print('resultat',dtype,names)
        dtype = [dtype.fields[name][0] for name in names]
        tab_tot = Table(rows=r, names=names, dtype=dtype)

        #tab_tot = Table(r, names=tuple(names))
        #tab_tot = np.rec.fromrecords(r, names = names)#dtype = [dtype.fields[name][0] for name in names])
        """
        print('finally',tab_tot)
        bands_gr = tab_tot.group_by('band')
        print(bands_gr.groups.keys)
        for key, group in zip(bands_gr.groups.keys, bands_gr.groups):
            print(group)
        tab_tot = vstack([group for key, group in zip(bands_gr.groups.keys, bands_gr.groups)])
        time_ref = time.time()
        """
        #self.Save(tab_tot)
        time_ref = time.time()
        LC_Summary = self.Ana_LC(tab_tot)
        print('after Ana',time.time()-time_ref)
        
        self.Plot_Sigma_c(LC_Summary)
        print(LC_Summary)
        
        if display:
            season = 1
            zref = 0.5
            idx = (tab_tot['season'] == season)&(tab_tot['z']==zref)
            sel_tab = tab_tot[idx]
            for DayMax in np.unique(sel_tab['DayMax']):
                idxb = sel_tab['DayMax'] == DayMax
                self.Plot_LC(sel_tab[idxb]['time','band','flux', 'fluxerr', 'zp', 'zpsys'], time_display,zref,DayMax,season)
        return ra, dec, tab_tot

    def Process_band(self,sel_obs,band,gen_par,j=-1,output_q=None):
        
        method = 'nearest'
        if len(sel_obs) == 0:
            if output_q is not None:
                output_q.put({j : None})
            else:
                return None

        # Get the fluxes (from griddata reference)
        xi = sel_obs[self.mjdCol]-gen_par['DayMax'][:,np.newaxis]
        yi = gen_par['z'][:,np.newaxis]
        x = self.lc_ref[band]['time']
        y = self.lc_ref[band]['z']
        z = self.lc_ref[band]['flux']
        zb=self.lc_ref[band]['fluxerr']
        fluxes_obs = griddata((x,y),z,(xi, yi), method=method)
        fluxes_obs_err=griddata((x,y),zb,(xi, yi), method=method)
        p = xi/(1.+yi) #phases of LC points

        # Estimate elements to compute Fisher matrices

        dFlux = {}
        for val in self.param_Fisher:
            z_c = self.lc_ref[band]['d'+val]
            dFlux[val] = griddata((x,y),z_c,(xi, yi), method=method)
            
        Derivative_for_Fisher = {}
        for ia,vala in enumerate(self.param_Fisher):
            for jb,valb in enumerate(self.param_Fisher):
                if jb >= ia:
                    Derivative_for_Fisher[vala+valb] = dFlux[vala]*dFlux[valb]/fluxes_obs_err**2

        #for key, vals in Derivative_for_Fisher.items():
        #    print('Fisher',band,key,vals.shape)

        # remove LC points outside the restframe phase range
        min_rf_phase = gen_par['min_rf_phase'][:,np.newaxis]
        max_rf_phase = gen_par['max_rf_phase'][:,np.newaxis]
        flag = (p >= min_rf_phase) & (p <= max_rf_phase)
        
        # remove LC points outside the (blue-red) range 
        mean_restframe_wavelength = np.array([self.telescope.mean_wavelength[band]]*len(sel_obs))
        mean_restframe_wavelength = np.tile(mean_restframe_wavelength,(len(gen_par),1))/(1.+gen_par['z'][:,np.newaxis])
        flag &= (mean_restframe_wavelength > self.blue_cutoff)&(mean_restframe_wavelength < self.red_cutoff)

        flag_idx = np.argwhere(flag)

        # Correct fluxes_err (m5 in generation probably different from m5 obs)
        gamma_obs = self.telescope.gamma(sel_obs[self.m5Col],[band]*len(sel_obs),sel_obs[self.exptimeCol])
        mag_obs = -2.5*np.log10(fluxes_obs/3631.)
        m5 = np.asarray([self.m5_ref[band]]*len(sel_obs))
        gammaref = np.asarray([self.gamma_ref[band]]*len(sel_obs))
        srand_ref = self.srand(np.tile(gammaref,(len(mag_obs),1)),mag_obs,np.tile(m5,(len(mag_obs),1)))
        srand_obs = self.srand(np.tile(gamma_obs,(len(mag_obs),1)),mag_obs,np.tile(sel_obs[self.m5Col],(len(mag_obs),1)))
        correct_m5 = srand_ref/srand_obs
        fluxes_obs_err = fluxes_obs_err/correct_m5
        
        #now apply the flag to select LC points
        fluxes = np.ma.array(fluxes_obs,mask=~flag)
        fluxes_err = np.ma.array(fluxes_obs_err,mask=~flag)
        snr_m5 = np.ma.array(fluxes_obs/fluxes_obs_err,mask=~flag)
        obs_time = np.ma.array(np.tile(sel_obs[self.mjdCol],(len(mag_obs),1)),mask=~flag)
        seasons = np.ma.array(np.tile(sel_obs[self.seasonCol],(len(mag_obs),1)),mask=~flag)
        z_vals = gen_par['z'][flag_idx[:,0]]
        DayMax_vals = gen_par['DayMax'][flag_idx[:,0]]
        mag_obs = np.ma.array(mag_obs,mask=~flag)
        Fisher_Mat = {}
        for key,vals in Derivative_for_Fisher.items():
            Fisher_Mat[key] = np.ma.array(vals,mask=~flag)
    
        #print(fluxes)
        
        # Results are stored in an astropy Table
        tab = Table()
        tab.add_column(Column(fluxes[~fluxes.mask],name='flux'))
        tab.add_column(Column(fluxes_err[~fluxes_err.mask],name='fluxerr'))
        
        tab.add_column(Column(snr_m5[~snr_m5.mask], name='snr_m5')) 
        tab.add_column(Column(mag_obs[~mag_obs.mask],name='mag'))
        tab.add_column(Column((2.5/np.log(10.))/snr_m5[~snr_m5.mask], name='magerr'))
        tab.add_column(Column(obs_time[~obs_time.mask],name='time'))
        
        
        tab.add_column(
            Column(['LSST::'+band]*len(tab), name='band',
                       dtype=h5py.special_dtype(vlen=str)))
        
        tab.add_column(Column([2.5*np.log10(3631)]*len(tab),
                                    name='zp'))
        
        tab.add_column(
            Column(['ab']*len(tab), name='zpsys',
                   dtype=h5py.special_dtype(vlen=str)))
        
        
        tab.add_column(Column(seasons[~seasons.mask],name='season'))
        tab.add_column(Column(z_vals,name='z'))
        tab.add_column(Column(DayMax_vals,name='DayMax'))
       
        
        tab.add_column(Column(self.mag_to_flux_e_sec[band](tab['mag'])),name='flux_e_sec')
        for key,vals in Fisher_Mat.items():
            tab.add_column(Column(vals[~vals.mask],name='F_'+key))
        
        # remove lc points with a null or negative flux
        idx = tab['flux'] >= 0.
        lc = tab[idx]
        lc = tab
        if len(lc) == 0:
            lc = None
        if output_q is not None:
            output_q.put({j : lc})
        else:
            return lc

    def Ana_LC(self, tab):
        
        r = []
        restab = Table()
        #for season in np.unique(tab['season']):
        for season in np.unique(tab['season']):
            idxa = tab['season'] == season
            sela = tab[idxa]
            for z in np.unique(sela['z']):
                time_refa = time.time()
                idxb = np.abs(sela['z']-z)<1.e-5
                selb = sela[idxb]
                DayMax = np.unique(selb['DayMax'])
                print('alors',season,z,len(DayMax))
                diff = selb['DayMax']-DayMax[:,np.newaxis]
                flag = np.abs(diff)<1.e-5
                #flag_idx = np.argwhere(flag)
                #print(diff*flag)
                resu = np.ma.array(np.tile(selb,(len(DayMax),1)),mask=~flag)
                parts = {}
                print('there bef',time.time()-time_refa)
                time_refa = time.time()
                for ia,vala in enumerate(self.param_Fisher):
                    for jb,valb in enumerate(self.param_Fisher):
                        if jb >= ia:
                            #print('F_'+vala+valb,np.sum(resu['F_'+vala+valb],axis = 1))
                            parts[ia,jb] = np.sum(resu['F_'+vala+valb],axis = 1)
                    
                print('there one',time.time()-time_refa)
                time_refa = time.time()
                size = len(resu)
                #size = 3
                Fisher_Big = np.zeros((3*size,3*size))
                restab_loc = Table()
                restab_loc.add_column(Column([season]*size,name='season'))
                restab_loc.add_column(Column([z]*size,name='z'))
                restab_loc.add_column(Column(DayMax[:size],name='DayMax'))
                print('there two',time.time()-time_refa)
                time_refa = time.time()
                for iv in range(size):
                    #Fisher_Matrix = np.zeros((3,3))
                    for ia,vala in enumerate(self.param_Fisher):
                        for jb,valb in enumerate(self.param_Fisher):
                            if jb >= ia:
                                #Fisher_Matrix[ia,jb] = parts[ia,jb][iv]
                                Fisher_Big[ia+3*iv][jb+3*iv] = parts[ia,jb][iv]
                    #Fisher_Matrix = np.maximum(Fisher_Matrix, Fisher_Matrix.transpose())
                    #print('ici',Fisher_Matrix)
                
                Fisher_Big = np.maximum(Fisher_Big, Fisher_Big.transpose())
                #plt.imshow(Fisher_Big)
                #plt.show()
                print('Summ',Fisher_Big.shape,Fisher_Big[:10][:])
                print('there three',time.time()-time_refa)
                #from IPython import embed
                time_refa = time.time()
                #embed()
                Big_Diag = np.linalg.inv(Fisher_Big)
                #np.linalg.cholesky(Fisher_Big)
                #M,invM = self.invert_matrix(Fisher_Big)
                #Big_Diag = np.diag(invM)
                #print('Big',Big_Diag)
                print('there four',time.time()-time_refa)
                time_refa = time.time()
                #print('Diag',np.diag(np.linalg.inv(Fisher_Big)))
                for ia,vala in enumerate(self.param_Fisher):
                    indices = range(ia,len(Big_Diag),3)
                    #print('test',ia,vala,np.take(Big_Diag,indices))
                    restab_loc.add_column(Column(np.sqrt(np.take(Big_Diag,indices)),name='sigma_'+vala))
                    
                restab = vstack([restab,restab_loc])
                print('there five',time.time()-time_refa)
                time_refa = time.time()
                """
                for ii,DayMax in enumerate(np.unique(selb['DayMax'])):
                    idxc = np.abs(selb['DayMax']-DayMax)<1.e-5
                    selc = selb[idxc]
                    Fisher_Matrix = np.zeros((3,3))
                    for ia,vala in enumerate(self.param_Fisher):
                        for jb,valb in enumerate(self.param_Fisher):
                            if jb >= ia:
                                Fisher_Matrix[ia,jb] = np.sum(selc['F_'+vala+valb])
                    Fisher_Matrix = np.maximum(Fisher_Matrix, Fisher_Matrix.transpose())
                    Sigma = np.linalg.inv(Fisher_Matrix)
                    r.append([season,z,DayMax]+[np.sqrt(Sigma[ia][ia]) for ia,vala in enumerate(self.param_Fisher)])
                    print('la',Fisher_Matrix)
                    if ii > 3:
                        break
                """
                #break
            #break
        return restab
        #return np.rec.fromrecords(r, names = ['season','z','DayMax']+['sigma_'+str(val) for val in self.param_Fisher])

    def Plot_Sigma_c(self, tab):

        import pylab as plt
        season = 3

        idx = tab['season'] == season
        sel = tab[idx]
        print('number of LC',len(np.unique(sel['DayMax'])))
        for DayMax in np.unique(sel['DayMax']):
            idxb = np.abs(sel['DayMax']-DayMax) < 1.e-5
            selb = sel[idxb]
            plt.plot(selb['z'],selb['sigma_Color'],'k.')
        plt.show()

    def dpotrf(self,M):
        return lapack.dpotrf(M, lower=True, overwrite_a=True)[0]


    def dtrtri(self,M):
        return lapack.dtrtri(M, lower=True)[0]
        
    def matmul(self,a, b, out):
        return np.matmul(a, b, out=out)


    def invert_matrix(self,M,dtype="float32"):
        """Invert a positive definite matrix using cholesky decomposition.
        WARNING : This DOES NOT check if the matrix is positive definite and can lead to wrong results if a non positive definite matrix is given.
        
        Arguments:
        - `M`: Matrix to invert
        """
        M = self.dpotrf(M)  # L
        invL = np.ascontiguousarray(self.dtrtri(M), dtype=dtype)  # invL
        if np.__version__=='1.13.3':
            self.matmul(invL.T, invL, out=invL)  # invC
        else:
            invL = np.matmul(invL.T, invL)
        return M, invL




    def Add_Gamma(self,obs):
        
        gamma=self.telescope.gamma(obs[self.m5Col],[seas[self.filterCol][-1] for seas in obs],obs[self.exptimeCol])
        obs=rf.append_fields(obs,'gamma',gamma)
    
        return obs

    def Proc_Extend(self,param,obs_c,j=-1,output_q=None):
    
        obs_tot = None
       
        time_ref=time.time()
        
        obs_c=rf.append_fields(obs_c,'DayMax',[param['DayMax']]*len(obs_c))
        obs_c=rf.append_fields(obs_c,'z',[param['z']]*len(obs_c))
        obs_c=rf.append_fields(obs_c,'X1',[param['X1']]*len(obs_c))
        obs_c=rf.append_fields(obs_c,'Color',[param['Color']]*len(obs_c))
       
        if output_q is not None:
            output_q.put({j : lc})
        else:
            return obs_c

    def Multiproc(self,obs):

        tab_tot=Table()
        """
        prefix=obs['band'][0].split('::')[0]
        for band in bands:
            idx = obs['band']== prefix+'::'+band
            obs_b=obs[idx]
            if len(obs_b) > 0:
                tab_tot=vstack([tab_tot,self.Process(obs_b,band,telescope)])
        """        
                
        njobs=-1
        result_queue = multiprocessing.Queue()
        """
        prefix=''
        if '::' in obs['band'][0]:
            prefix=obs['band'][0].split('::')[0]
            prefix+='::'
        """
        for band in np.unique(obs[self.filterCol]):
            idx = obs[self.filterCol]== band
            obs_b=obs[idx]
         

            if len(obs_b) > 0:
                njobs+=1
             #print('starting multi')
                p=multiprocessing.Process(name='Subprocess-'+str(njobs),target=self.Process,args=(obs_b,band,njobs,result_queue))
                p.start()
             #print('starting multi done')
        resultdict = {}
        for j in range(njobs+1):
            resultdict.update(result_queue.get())
		
        for p in multiprocessing.active_children():
            p.join()
                
        
	#print('stacking')
          
        for j in range(0,njobs+1):
            if resultdict[j] is not None:
                tab_tot=vstack([tab_tot,resultdict[j]])

        return tab_tot

    def Process(self,obs,band,j=-1,output_q=None):
    
        deriv={}

        diff=np.copy(obs[self.mjdCol]-obs['DayMax'])
        x=self.lc_ref[band]['time']
        y=self.lc_ref[band]['z']
        z=self.lc_ref[band]['flux']
        zb=self.lc_ref[band]['fluxerr']
        xi=np.copy(obs[self.mjdCol]-obs['DayMax'])
        yi=obs['z']

   
        method='nearest'
        fluxes_obs=griddata((x,y),z,(xi, yi), method=method)
        fluxes_err=griddata((x,y),zb,(xi, yi), method=method)

        """
        print(self.lc_ref[band][['flux','time']])
        print(xi,yi)
        print(fluxes_obs)
        """
        mag_obs=-2.5*np.log10(fluxes_obs/3631.)
    
        m5=np.asarray([self.m5_ref[band]]*len(obs))
        gammaref=np.asarray([self.gamma_ref[band]]*len(obs))
    
        correct_m5=self.srand(gammaref,mag_obs,m5)/self.srand(obs['gamma'],mag_obs,obs[self.m5Col])
        """
        for key in self.key_deriv:
            xa=self.lc_ref[band]['time']
            ya=self.lc_ref[band]['z']
            za=self.lc_ref[band][key]
        
            deriv[key]=griddata((xa,ya),za,(xi, yi), method=method)
        """
        tab=Table()
        
        tab.add_column(Column(fluxes_obs,name='flux'))
        tab.add_column(Column(fluxes_err/correct_m5,name='fluxerr'))
        snr_m5_opsim = fluxes_obs/(fluxes_err/correct_m5)
        tab.add_column(Column(snr_m5_opsim, name='snr_m5'))
        tab.add_column(Column(mag_obs,name='mag'))
        tab.add_column(Column((2.5/np.log(10.))/snr_m5_opsim, name='magerr'))
        tab.add_column(Column(obs[self.mjdCol],name='time'))
        tab.add_column(
            Column(['LSST::'+obs[self.filterCol][i][-1]
                    for i in range(len(obs[self.filterCol]))], name='band',
                   dtype=h5py.special_dtype(vlen=str)))
        tab.add_column(Column([2.5*np.log10(3631)]*len(obs),
                                    name='zp'))
        tab.add_column(
            Column(['ab']*len(obs), name='zpsys',
                   dtype=h5py.special_dtype(vlen=str)))

        idx = tab['flux'] >= 0.
        tab= tab[idx]

        if output_q is not None:
            output_q.put({j : tab})
        else:
            return tab

    def srand(self,gamma,mag,m5):
        x=10**(0.4*(mag-m5))
        return np.sqrt((0.04-gamma)*x+gamma*x**2)
    
class SNFast_old:

    def __init__(self,fieldname,fieldid,sim_name,sim_type,X1,Color,zvals):

        file_obs='/sps/lsst/users/gris/Files_from_OpSim/OpSimLogs_'+str(sim_name)+'/'+fieldname+'/Observations_'+fieldname+'_'+str(fieldid)+'.txt'
        print('loading obs')
        time_ref=time.time()
        self.telescope=Telescope(airmass=1.2)
        observation=Observations(int(fieldid),filename=file_obs)
        print('end of loading obs',time.time()-time_ref)
        #print(len(observation.seasons))

        self.seasons={}
        for i in range(len(observation.seasons)):
            self.seasons[i]=self.Add_Gamma(observation.seasons[i])[['band','mjd','m5sigmadepth','gamma']]
            print(i,self.season_limit(self.seasons[i]))
        self.dir_out='/sps/lsst/users/gris/Light_Curves_'+sim_type+'/'+str(sim_name)
        if not os.path.isdir(self.dir_out) :
            os.makedirs(self.dir_out)

        self.fieldname=fieldname
        self.fieldid=fieldid
        self.sim_name=sim_name
        self.sim_type=sim_type

        self.X1=X1
        self.Color=Color
        print('loading ref',zvals)
        self.lc_ref_dict=Load_Reference(X1,Color,zvals).tab

    def __call__(self,season,param):

        outname=self.fieldname+'_'
        outname+=str(self.fieldid)+'_'
        outname+='X1_'+str(self.X1)+'_Color_'+str(self.Color)+'_'
        outname+='season_'+str(season)+'_'
        self.name_out=self.dir_out+'/'+'LC_'+outname+'.hdf5'
        self.name_ana_out='Summary_'+outname+'.npy'

        time_ref=time.time()
        #print('calling',np.unique(param['z']))

        self.Extend_Observations(param,self.lc_ref_dict)

        print('Time',time.time()-time_ref,len(param))

    def Add_Gamma(self,season):
        
        obs_c=season.copy()
        gamma=self.telescope.gamma(obs_c['m5sigmadepth'],[seas['band'][-1] for seas in obs_c],obs_c['exptime'])
        obs_c=rf.append_fields(obs_c,'gamma',gamma)
    
        return obs_c

    def season_limit(self,myseason):
        prefix=''
        if '::' in myseason['band'][0]:
            prefix=myseason['band'][0].split('::')[0]
            prefix+='::'
        iddx=np.asarray([seas['band']!=prefix+'u' for seas in myseason])
        
        mysel=myseason[iddx]
        mysel=np.sort(mysel,order='mjd')
        
        min_season=np.min(mysel['mjd'])
        max_season=np.max(mysel['mjd'])
    
        return (min_season,max_season)

    def Proc_Extend(self,params,lc_ref_dict,j=-1,output_q=None):
    
        obs_tot=None
        X1=np.unique(params['X1'])[0]
        Color=np.unique(params['Color'])[0]

        #print('X1 and Color',len(params),X1,Color)
        time_ref=time.time()
        for param in params:
            season=self.seasons[param['season']]
            obs_c = np.copy(season)
    
            obs_c=rf.append_fields(obs_c,'DayMax',[param['DayMax']]*len(obs_c))
            mean_restframe_wavelength = np.asarray([self.telescope.throughputs.mean_wavelength[obser['band'][-1]]/ (1. + param['z']) for obser in obs_c])
            p=(obs_c['mjd']-obs_c['DayMax'])/(1.+param['z'])
    
            #idx=(np.min(p)<=-5)&(np.max(p)>=10)
    
            idx = (p >= min_rf_phase)&(p<=max_rf_phase)&(mean_restframe_wavelength>blue_cutoff) & (mean_restframe_wavelength<red_cutoff)
            obs_c=obs_c[idx]
            
            if len(obs_c) > 1 :
                obs_c=rf.append_fields(obs_c,'z',[param['z']]*len(obs_c))
                obs_c=rf.append_fields(obs_c,'X1',[param['X1']]*len(obs_c))
                obs_c=rf.append_fields(obs_c,'Color',[param['Color']]*len(obs_c))
                obs_c=rf.append_fields(obs_c,'SNID',[100+j]*len(obs_c))
                if obs_tot is None:
                    obs_tot=obs_c
                else:
                    obs_tot=np.concatenate((obs_tot,obs_c))

        #print('alors',len(obs_tot),np.unique(obs_tot['DayMax']),time.time()-time_ref)
        lc=None
        if obs_tot is not None:
            time_proc=time.time()
            #print('processing')
            lc=Process_X1_Color(X1,Color,obs_tot,lc_ref_dict,self.telescope).lc_fast
            #print('processed',len(lc),np.min(lc['DayMax']),np.max(lc['DayMax']),time.time()-time_proc)
            if len(lc) ==0:
                lc=None
        if output_q is not None:
            output_q.put({j : lc})
        else:
            return obs_c
    
    def Extend_Observations(self,param,lc_ref_dict):

        distrib=param                                                                  
        nlc=len(distrib)    
        #n_multi=7
        n_multi=min(7,nlc)
        n_multi=min(2,nlc)
        print('extended ',n_multi,nlc)
        nvals=nlc/n_multi
        batch=range(0,nlc,nvals)
        if batch[-1] != nlc:
            batch=np.append(batch,nlc)
    
        result_queue = multiprocessing.Queue()

        print('hello',batch)
        for i in range(len(batch)-1):
        
            ida=int(batch[i])
            idb=int(batch[i+1])
            print('processing here',ida,idb,param[ida:idb])
            p=multiprocessing.Process(name='Subprocess-'+str(i),target=self.Proc_Extend,args=(param[ida:idb],lc_ref_dict,i,result_queue))                                
            p.start() 
        #print('processing',ida,idb,nlc,len(param[ida:idb]))
    
        resultdict = {}
        for j in range(len(batch)-1):
            resultdict.update(result_queue.get())
    
        for p in multiprocessing.active_children():
            p.join()

        lc=Table()
        for j in range(len(batch)-1):
        #print('hello lc',len(resultdict[j]))
            if resultdict[j] is not None:
                lc=vstack([lc,resultdict[j]])
        #print('calling')
        if not os.path.isfile(self.name_out):
            itot=0
            lc.write(self.name_out, path='lc_'+str(itot), compression=True)
            time_ana=time.time()
            #print('LC Analysis')
            Ana_LC(lc=lc,outname=self.name_ana_out,sim_name=self.sim_name,dir_lc=self.dir_out,sim_type=self.sim_type)
            #print('after analysi',time.time()-time_ana)
        else:
            print('This file',self.name_out,'already exists -> out')


class Process_X1_Color:
    def __init__(self,X1,Color,obs_extended,lc_ref_tot,telescope):
      
        bands=np.unique(lc_ref_tot['band'])
        
        self.lc_ref={}
        
        self.gamma_ref={}
        self.m5_ref={}
        
        self.key_deriv=['dX1','dColor','dX0']
    
        for band in bands:
            
            idx = lc_ref_tot['band']==band
            self.lc_ref[band]=lc_ref_tot[idx]
        
            self.gamma_ref[band]=self.lc_ref[band]['gamma'][0]
            self.m5_ref[band]=np.unique(lc_ref_tot[idx]['m5'])[0]
       
    
        if len(obs_extended) > 0:
            self.Multiproc(obs_extended,telescope,bands)
            """
            time_single=time.time()
            print('processing single')
            self.Process_Single(obs_extended,telescope,bands)
            print('after proc_single',time.time()-time_single)
            """
            #np.save('LC_FastSim_X1_'+str(X1)+'_Color_'+str(Color)+'.npy',lc_fast)
        #print('Time',time.time()-time_ref,len(lc_fast))

    def Process_Single(self,obs,telescope,bands):
        tab_tot=Table()
        prefix=''
        if '::' in obs['band'][0]:
            prefix=obs['band'][0].split('::')[0]
            prefix+='::'
        for band in bands:
            idx = obs['band']== prefix+band
            obs_b=obs[idx]
            if len(obs_b) > 0:
                #lc=self.Process(obs_b,band,telescope)
                tab_tot=vstack([tab_tot,self.Process(obs_b,band,telescope)])
                #tab_tot=vstack([tab_tot,lc])
    
        self.lc_fast=tab_tot

    def Multiproc(self,obs,telescope,bands):

        tab_tot=Table()
        """
        prefix=obs['band'][0].split('::')[0]
        for band in bands:
            idx = obs['band']== prefix+'::'+band
            obs_b=obs[idx]
            if len(obs_b) > 0:
                tab_tot=vstack([tab_tot,self.Process(obs_b,band,telescope)])
        """        
                
        njobs=-1
        result_queue = multiprocessing.Queue()
        prefix=''
        if '::' in obs['band'][0]:
            prefix=obs['band'][0].split('::')[0]
            prefix+='::'
        for band in bands:
            idx = obs['band']== prefix+band
            obs_b=obs[idx]
         

            if len(obs_b) > 0:
                njobs+=1
             #print('starting multi')
                p=multiprocessing.Process(name='Subprocess-'+str(njobs),target=self.Process,args=(obs_b,band,telescope,njobs,result_queue))
                p.start()
             #print('starting multi done')
        resultdict = {}
        for j in range(njobs+1):
            resultdict.update(result_queue.get())
		
        for p in multiprocessing.active_children():
            p.join()
                
        
	#print('stacking')
          
        for j in range(0,njobs+1):
            if resultdict[j] is not None:
                tab_tot=vstack([tab_tot,resultdict[j]])
        
        
           
            #print(tab_tot.dtype)
            #print(len(np.unique(tab_tot['DayMax'])))
        
        """
        #print('allo',tab_tot)
        Fisher_mat=np.empty((3,3))

        for i,key in enumerate(self.key_deriv):
            for j,keyb in enumerate(self.key_deriv):
                Fisher_mat[i][j]=np.sum(tab_tot[key]*tab_tot[keyb]/tab_tot['fluxerr']**2)

        Inv_Fisher=np.linalg.inv(Fisher_mat)
        
        #print(Fisher_mat,np.linalg.inv(Fisher_mat),np.sqrt(Inv_Fisher[0][0]),np.sqrt(Inv_Fisher[1][1]),np.sqrt(Inv_Fisher[2][2]))
        """

        self.lc_fast=tab_tot
        #print('there we go',len(tab_tot),len( self.lc_fast))

    def Process(self,obs,band,telescope,j=-1,output_q=None):
    
        deriv={}

        diff=np.copy(obs['mjd']-obs['DayMax'])
        x=self.lc_ref[band]['time']
        y=self.lc_ref[band]['z']
        z=self.lc_ref[band]['flux']
        zb=self.lc_ref[band]['fluxerr']
        xi=np.copy(obs['mjd']-obs['DayMax'])
        yi=obs['z']

   
        method='nearest'
        fluxes_obs=griddata((x,y),z,(xi, yi), method=method)
        fluxes_err=griddata((x,y),zb,(xi, yi), method=method)

        """
        print(self.lc_ref[band][['flux','time']])
        print(xi,yi)
        print(fluxes_obs)
        """
        mag_obs=-2.5*np.log10(fluxes_obs/3631.)
    
        m5=np.asarray([self.m5_ref[band]]*len(obs))
        gammaref=np.asarray([self.gamma_ref[band]]*len(obs))
    
        correct_m5=self.srand(gammaref,mag_obs,m5)/self.srand(obs['gamma'],mag_obs,obs['m5sigmadepth'])
        for key in self.key_deriv:
            xa=self.lc_ref[band]['time']
            ya=self.lc_ref[band]['z']
            za=self.lc_ref[band][key]
        
            deriv[key]=griddata((xa,ya),za,(xi, yi), method=method)
   
            
        tab=Table()
        
        tab.add_column(Column(fluxes_obs,name='flux'))
        
        tab.add_column(Column(mag_obs,name='mag'))
        tab.add_column(Column(fluxes_err/correct_m5,name='fluxerr'))
        tab.add_column(Column(obs['mjd'],name='time'))
        tab.add_column(Column(['LSST::'+band]*len(diff),name='band'))
        tab.add_column(Column(obs['m5sigmadepth'],name='m5sigmadepth'))
        tab.add_column(Column(obs['DayMax'],name='DayMax'))
        #tab.add_column(Column(obs['season'],name='season'))
        tab.add_column(Column([2.5*np.log10(3631)]*len(tab),name='zp'))
        tab.add_column(Column(['ab']*len(tab),name='zpsys'))
        tab.add_column(Column(obs['z'],name='z'))
        tab.add_column(Column(obs['X1'],name='X1'))
        tab.add_column(Column(obs['Color'],name='Color'))
        
        #tab.add_column(Column(obs['SNID'],name='SNID'))
        for key in self.key_deriv:
            tab.add_column(Column(deriv[key],name=key))
            
        
        if output_q is not None:
            output_q.put({j : tab})
        else:
            return tab

    def srand(self,gamma,mag,m5):
        x=10**(0.4*(mag-m5))
        return np.sqrt((0.04-gamma)*x+gamma*x**2)


class Load_Reference:

    def __init__(self,filename,zvals):
        
        self.fi=filename
        self.tab = self.Read_Ref(zvals)

    def Read_Ref(self,zvals,j=-1,output_q=None):

        tab_tot=Table()
        """
        keys=np.unique([int(z*100) for z in zvals])
        print(keys)
        """
        f = h5py.File(self.fi,'r')
        keys=f.keys()

        zvals_arr = np.array(zvals)
        
        for kk in keys:
            
            tab_b=Table.read(self.fi, path=kk)
            #print('hello key',kk,np.unique(tab_b['z']))
            if tab_b is not None:
                diff = tab_b['z']-zvals_arr[:,np.newaxis]
                #flag = np.abs(diff)<1.e-3
                flag_idx = np.where(np.abs(diff)<1.e-3)
                if len(flag_idx[1]) > 0:
                    tab_tot=vstack([tab_tot,tab_b[flag_idx[1]]])
                """
                print(flag,flag_idx[1])
                print('there man',tab_b[flag_idx[1]])
                mtile = np.tile(tab_b['z'],(len(zvals),1))
                #print('mtile',mtile*flag)
                
                masked_array = np.ma.array(mtile,mask=~flag)
                
                print('resu masked',masked_array,masked_array.shape)
                print('hhh',masked_array[~masked_array.mask])
                
            
                for val in zvals:
                    print('hello',tab_b[['band','z','time']],'and',val)
                    if np.abs(np.unique(tab_b['z'])-val)<0.01:
                        #print('loading ref',np.unique(tab_b['z']))
                        tab_tot=vstack([tab_tot,tab_b])
                        break
                """
        if output_q is not None:
            output_q.put({j : tab_tot})
        else:
            return tab_tot

    def Read_Multiproc(self,tab):

        #distrib=np.unique(tab['z'])
        nlc=len(tab)
        print('ici pal',nlc)
        #n_multi=8
        if nlc >=8:
            n_multi=min(nlc,8)
            nvals=nlc/n_multi
            batch=range(0,nlc,nvals)
            batch=np.append(batch,nlc)
        else:
            batch = range(0,nlc)
            
        #lc_ref_tot={}
        #print('there pal',batch)
        result_queue = multiprocessing.Queue()
        for i in range(len(batch)-1):
        
            ida=int(batch[i])
            idb=int(batch[i+1])
            
            p=multiprocessing.Process(name='Subprocess_main-'+str(i),target=self.Read_Ref,args=(tab[ida:idb],i,result_queue))
            p.start()

        resultdict = {}
        for j in range(len(batch)-1):
            resultdict.update(result_queue.get())

        for p in multiprocessing.active_children():
            p.join()

        tab_res=Table()
        for j in range(len(batch)-1):
            if resultdict[j] is not None:
                tab_res=vstack([tab_res,resultdict[j]])

        return tab_res




