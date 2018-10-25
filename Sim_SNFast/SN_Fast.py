import numpy as np
from Observations import *
from scipy import interpolate
from astropy.table import Table,Column,vstack
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
        print('loading ref',zvals)
        lc_ref_tot=Load_Reference(X1,Color,zvals).tab

        self.lc_ref={}
        
        self.gamma_ref={}
        self.m5_ref={}

        print(lc_ref_tot.dtype)
        bands = np.unique(lc_ref_tot['band'])
        for band in bands:
            
            idx = lc_ref_tot['band']==band
            lc_sel=lc_ref_tot[idx]
            print('hello',lc_sel)
            self.lc_ref[band] = lc_sel
            self.gamma_ref[band]=lc_sel['gamma'][0]
            self.m5_ref[band]=np.unique(lc_sel['m5'])[0]
            
        
    def __call__(self, obs, index_hdf5,display=False, time_display=0.):
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

        metadata = dict(zip(['SNID', 'Ra', 'Dec',
                                  'DayMax', 'X1', 'Color', 'z','survey_area','index_hdf5'], [
                                      self.SNID, ra, dec, self.sn_parameters['DayMax'],
                                      self.sn_parameters['X1'], self.sn_parameters['Color'],
                                      self.sn_parameters['z'], area, index_hdf5]))
        
        # print('Simulating SNID', self.SNID)
        obs = self.cutoff(obs, self.sn_parameters['DayMax'],
                          self.sn_parameters['z'],
                          self.sn_parameters['min_rf_phase'],
                          self.sn_parameters['max_rf_phase'])

        if len(obs) == 0:
            return None, metadata
        obs = self.Add_Gamma(obs)
        obs = self.Proc_Extend(self.sn_parameters,obs)
        lc = self.Multiproc(obs)

        print('light curve',lc)
        print(lc.dtype)
        if display:
            self.Plot_LC(lc['time','band','flux', 'fluxerr', 'zp', 'zpsys'], time_display)
        return lc, metadata

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

    def __init__(self,X1,Color,tab):
        
        thedir='Reference_LC/'
        self.fi=thedir+'LC_Ref_'+str(X1)+'_'+str(Color)+'.hdf5'
        #f = h5py.File(fi,'r')

        #print('loading reference file',self.fi)
        #self.tab=self.Read_Multiproc(tab)
        self.tab = self.Read_Ref(tab)

    def Read_Ref(self,tab,j=-1,output_q=None):

        tab_tot=Table()
        #zval=tab['z']
        keys=np.unique([int(z*100) for z in tab])
        f = h5py.File(self.fi,'r')
        keys=f.keys()

        #print('alors',tab)
        print('hello keys',keys,tab)
        for kk in keys:
            
            tab_b=Table.read(self.fi, path=kk)
            #if tab_b is not None:
            for val in tab:
                if np.abs(np.unique(tab_b['z'])-val)<0.01:
                    #print('loading ref',np.unique(tab_b['z']))
                    tab_tot=vstack([tab_tot,tab_b])
                    break
                
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




