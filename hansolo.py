import sys
import glob
import pickle
import time
import gc
from multiprocessing import Pool #Added for cluster

import numpy as np
from astropy.io import fits
from astropy import units as u
from astropy.modeling import models, fitting
from astropy import stats
from astropy.utils.exceptions import AstropyWarning
from pathlib import Path
from lsl.statistics import robust
import scipy
import scipy.signal
from scipy.optimize import least_squares
from matplotlib import pyplot as plt
import lacosmic_copy as lacosmic
import warnings
from barycorrpy import utc_tdb
import pandas as pd

warnings.simplefilter('ignore', category=AstropyWarning)


class DataFile:
    """
    A single fits file in the dataset and it's properties.
    
    Methods:
    --------
    __init__ - Create an instance
    get_type - Determine the data type
    
    Attributes:
    -----------
    name : str;
        Name of the file.
        
    date : str;
        Date on which the file was taken.
    
    chip_id : str;
        ID of the chip the data is read from.
        
    type : str;
        Type of data e.g. science, acquisition, bias, flat, arc, slit image, image
    """
    def __init__(self, fname):
        self.name = fname
        self.date = self.name.split('/')[-1][6:16]
        self.get_type()
        
    def get_type(self):
        """ Identify what type of image is in the file. """
        header = fits.getheader(self.name)
        self.chip_id = header['EXTNAME']
        
        type = header['ESO DPR TYPE']
        if type == 'SKY':
            if header['ESO DPR CATG'] == 'ACQUISITION':
                self.type = 'acquisition'
            else:
                self.type = 'science'
        elif type == 'BIAS':
            self.type = 'bias'
        elif type == 'FLAT,LAMP' or type=='FLAT,SKY':
            self.type = 'flat'
        elif type == 'WAVE,LAMP':
            self.type = 'arc'
        elif type == 'SLIT':
            self.type = 'slit'
        elif type == 'OBJECT':
            self.type = 'image'
        else:
            print(self.name)
            self.type = 'None'

class Source:
    """
    Data and methods associated with a single source in the data.
    
    Methods:
    --------
    __init__           -- initialise a source
    get_header_info    -- obtain information on the observing conditions from the header
    copy               -- make copy of the instance with the same parameters
    flux_err           -- calculate the error of the extracted flux
    clean_flux         -- remove outliers from flux measurements
    extract            -- extract flux from 2D images
    self_align         -- align all spectra to the first observed spectrum
    model_align        -- align all spectra to a model reference
    align_specs        -- align all spectra
    
    Attributes:
    -----------
    fluxes, flux_errors : 3D ndarray [time, wavelength, aperture];
        Flux values and their errors extracted from the data.
        
    fitcens, smocens : 2D ndarray [time, wavelength];
        Fitted (and smoothed) centers of the slit across the detector for each image.
        
    fwhms, backgrounds : 2D ndarray [time, wavelength];
        Full-width-half-maximum of the trace for each column on the detector in each image
        and the corresponding background flux.
        
    max_values : 2D ndarray [aperture, time];
        Maximum flux of each extracted spectrum.
        
    wavelengths, bjd, images : 1D ndarray;
        Wavelengths, timestamps and filenames of the extracted spectra.
        
    nx, n_apertures : int;
        Number of pixels in the wavelength direction and apertures at which the flux extraction is performed.
        
    index, chip : int;
        Identifier of the source and of which chip the source is extracted from.
    
    gain, ron : float;
        Detector properties of gain and read noise.
        
    airmass, exptime, parang, posang, rotation : 1D ndarray [time];
        Observing parameters for each timestamp.
        Airmass, exposure time, paralactic angle, position angle and rotation angle of the atmospheric dispersion corrector.
        These can be used as detrending parameters when fitting the light curves.
    """
    
    def __init__(self, images, nx, n_appertures, index, chip, grism_info=None, dumb=False):
        """
        Initialise a Source instance, setting up arrays for fluxes, errors, and other attributes.

        Parameters:
        -----------
        images : list;
            A list of images from which to extract data.
            
        nx : int;
            Number of pixels or points along the x-axis.
            
        n_appertures : int;
            Number of aperture sizes from which the flux is extracted.
            
        index : int;
            Identifier for the star.
            
        chip : int;
            Identifier of the detector chip on which the source is located.
            
        grism_info : str, optional;
            Grism table calibration file. Default is None.
            
        dumb :bool, optional; If True, suppresses initialization message. Default is False.
        """
        n_images = len(images)
        if not dumb:
            print('Initialising star '+str(index+1))
        self.number = index
        self.chip = chip
        self.fluxes = np.zeros([n_images, nx, n_appertures])
        self.flux_errors = np.zeros([n_images, nx, n_appertures])
        self.qis = np.zeros([n_images, nx])
        self.fitcens = np.zeros([n_images, nx])
        self.smocens = np.zeros([n_images, nx])
        self.fwhms = np.zeros([n_images, nx])
        self.backgrounds = np.zeros([n_images, nx])
        self.max_values = np.zeros([n_appertures, n_images])#np.zeros([n_images, nx])
        
        self.gain = 0.
        self.ron = 0.
        self.airmass = np.zeros([n_images])
        self.exptime = np.zeros([n_images])
        self.parang = np.zeros([n_images])
        self.posang = np.zeros([n_images])
        self.rotation = np.zeros([n_images])
        self.bjd = np.zeros([n_images])
        self.get_header_info(images)
        
        if grism_info:
            self.wmin, self.wmax, self.wstep = get_grism_parameters(grism_info)
            n_steps = np.ceil((self.wmax-self.wmin)/self.wstep)
            self.wavelengths = self.wmin + self.wstep * np.arange(n_steps)
    
    def get_header_info(self, file_list):
        """
        Obtain observation information from the image headers.
        
        Parameters:
        -----------
        file_list : list;
            List of science images in the dataset.
        """
        for i in range(len(file_list)):
            header = fits.getheader(file_list[i])
            self.filter = header['ESO INS GRIS1 NAME'].split('0')[-1]
            self.airmass[i] = np.mean([header['ESO TEL AIRM START'], header['ESO TEL AIRM END']])
            self.exptime[i] = header['EXPTIME']
            self.parang[i] = np.mean([header['ESO TEL PARANG START'], header['ESO TEL PARANG END']])
            self.posang[i] = header['ESO ADA POSANG']
            self.rotation[i] = np.mean([header['ESO ADA ABSROT START'],header['ESO ADA ABSROT END']])
            jd_utc = 2400000.5+header['MJD-OBS']
            self.bjd[i] = utc_tdb.JDUTC_to_BJDTDB(jd_utc, ra=header['RA'], dec=header['DEC'], lat=header['ESO TEL GEOLAT'], longi=header['ESO TEL GEOLON'], alt=header['ESO TEL GEOELEV'])[0][0]+self.exptime[i]*u.s.to(u.day)/2
            if i == 0:
                self.gain = header['ESO DET OUT{0} GAIN'.format(self.chip+1)] # Currently only gets read noise and gain from the first chip, need to add it so I can get header info from the second chip.
                self.ron = header['ESO DET OUT{0} RON'.format(self.chip+1)]
    
    def copy(self, dumb=False):
        """ Create a copy of the Source instance """
        copy_star = Source([], 0, 0, self.number, self.chip, dumb=dumb)
        key_list = vars(self).keys()
        for key in key_list:
            if type(vars(self)[key]) == np.ndarray:
                vars(copy_star)[key] = vars(self)[key].copy()
            else:
                vars(copy_star)[key] = 1*vars(self)[key]
        return copy_star
        
    def flux_err(self, aps, i, k):
        """ Calculate errorbars on the flux values including photon, scintillation, sky and read noise. """
        H = 8000 # scale height of the atmosphere in m
        h = 2653 # altitude of the observatory in m
        D = 8.2    # diameter of the telescope in m
        C = 1.56 # correction factor
        
        err_photon = np.sqrt(self.fluxes[i,:,k]) #gain is already included in the extracted flux
        err_scint = self.fluxes[i,:,k] * np.sqrt(10 * 1e-6 * C**2 * D**(-4/3) / self.exptime[i] * self.airmass[i]**3 * np.e**(-2*h/H)) #Osborn+ 2015, eq 7
        err_sky = np.sqrt(2*aps[k] * self.backgrounds[i] * self.gain)
        err_ron = np.sqrt(2*aps[k]) * self.ron #[self.chip]
        return np.sqrt(err_photon**2 + err_scint**2 + err_sky**2 + err_ron**2)
    
    def clean_flux(self, sig_thr=10, flags=None):
        """
        Remove outliers from the flux data and replace them with interpolated values.
        
        Parameters:
        -----------
        sig_thr : int, optional;
            Threshold for identifying outliers, expressed in sigmas. Default is 10.
            
        flags : list, optional;
            List of pixels that need to be cleaned regardless of outlier status.
        """
        print(f'\nCleaning star {self.number+1}')
        n_ims, n_x, n_apertures = self.fluxes.shape
        for i in range(n_x):
            for j in range(n_apertures):
                nan_inds = np.isnan(self.fluxes[:,i,j])
                good_inds = np.isfinite(self.fluxes[:,i,j])
                if flags:
                    good_inds[flags] = False
                nan_err_inds = np.isnan(self.flux_errors[:,i,j])
                if np.sum(nan_inds) < 10 and np.nanmedian(self.fluxes[:,i,j]) != 0:
                    x_fit = self.bjd[good_inds] - int(self.bjd[good_inds][0])
                    y_fit = self.fluxes[good_inds,i,j]/self.exptime[good_inds]
                    cs = robust_fit(x_fit, y_fit, 4, robust.std(y_fit))
                    y_poly = np.polyval(cs, x_fit)
                    residuals = y_fit - y_poly
                    res_std = running_std(residuals)
                    bad_inds = np.where((np.abs(residuals)>sig_thr*res_std)|(np.isnan(residuals)))[0]
                    true_bad_inds = np.where(good_inds)[0][bad_inds]
                    if len(bad_inds) < n_ims//10:
                        self.fluxes[true_bad_inds,i,j] = y_poly[bad_inds]*self.exptime[true_bad_inds]
        self.max_values = np.nanmax(self.fluxes,axis=1)
    
    def extract(self, file_list, slit_tops, slit_bottoms, skyreg, aps):
        """
        Extract the spectral flux from the 2D images.
        
        Parameters:
        -----------
        file_list : list;
            List of files from which the flux is to be extracted.
            
        slit_tops, slit_bottoms : list;
            List of the edges of all slits in the image for each wavelength.
            
        skyreg, list of size 4;
            List of the y-coordinates of the sky background area within the slit.
            
        aps : ndarray;
            Aperture half-widths around the source over which the flux is to be extracted.
        """
        sig_val = 100
        n_aps = len(aps)
        for i in range(len(file_list)):
            print('Star {0}/{1}, file {2}/{3} \r'.format( self.number+1, len(slit_tops), i+1, len(file_list) ), end='')
            data, header = fits.getdata(file_list[i], header=True)
            j = self.number
            y_inds = np.arange(np.median(slit_bottoms[j]),np.median(slit_tops[j]),dtype=int)
            data = data[y_inds]
            sky_inds = np.append(np.arange(skyreg[0],skyreg[1]), np.arange(skyreg[2],skyreg[3]))
            sky = data[sky_inds]
            
            count=0
            for k in range(data.shape[1]):
                moffat, self.backgrounds[i,k], data[:,k] = extract_moffat(data[:,k], sky_inds, y_inds-y_inds[0], plot_bool=False)
                if type(moffat)==float:
                    count+=1
                    self.fwhms[i,k] = np.nan
                    self.fitcens[i,k] = np.nan
                else:
                    self.fwhms[i,k] = np.abs(2*moffat.gamma.value)*np.sqrt(2**(1/moffat.alpha)-1)
                    self.fitcens[i,k] = moffat.x_0.value
            
            self.smocens[i] = thorough_smooth(self.fitcens[i],plot=False)
            cs = robust_fit(np.arange(0,data.shape[1])[np.isfinite(self.smocens[i])],self.smocens[i][np.isfinite(self.smocens[i])],2,f_scale=1e-4)
            self.smocens[i][np.isnan(self.smocens[i])] = np.polyval(cs,np.where(np.isnan(self.smocens[i]))[0])
            
            for k in range(n_aps):
                ap_lower = np.array(self.smocens[i]-aps[k], dtype=int)
                ap_higher = np.array(self.smocens[i]+aps[k], dtype=int)
                for p in range(data.shape[1]):
                    self.fluxes[i,p,k] = np.nansum(data[ap_lower[p]:ap_higher[p],p]) * self.gain
                self.flux_errors[i,:,k] = self.flux_err(aps, i, k)
            self.max_values[:,i] = np.max(self.fluxes[i],axis=0)
        
        self.apertures = aps
        self.backgrounds = self.backgrounds * self.gain
    
    def self_align(self, ref_ap, ref_im, regs, window, n_order):
        """
        Align all spectra to a single spectrum.
        
        Parameters:
        -----------
        ref_ap : int;
            Reference aperture at which the comparison spectrum was extracted.
        
        ref_im : int;
            Reference image from which the comparison spectrum was extracted.
            
        regs: list;
            List of start and end coordinates for spectral regions at which the cross-correlation function is calculated.
            
        window : int;
            Size of cross-correlation shifts in pixels.
            
        n_order : int;
            Order of the polynomial to which the shifts of the different regions are fitted. Recommended is 2.
        """
        n_apertures = self.fluxes.shape[-1]
        ref_spec = self.fluxes[ref_im,:,ref_ap]
        for i in range(self.fluxes.shape[0]):
            spec = self.fluxes[i,:,ref_ap]
            spec_range = np.where(self.fluxes[ref_im,:,ref_ap]>0)[0]
            
            # Calculate the relative shift in different parts of the spectrum.
            shifts = np.zeros([len(regs),2])*np.nan
            for j in range(len(regs)):
                inds = np.where((self.wavelengths>regs[j,0])&(self.wavelengths<regs[j,1]))[0]
                if np.sum(ref_spec[inds])!=0 and np.sum(spec[inds])!=0:
                    shifts[j] = ccf(ref_spec[inds],spec[inds],window,err=True)
            
            # Fit a polynomial to the shifts to identify the new wavelength solution.
            shifts = shifts*self.wstep
            if np.isnan(shifts[0,0]):
                new_wave = self.wavelengths
            else:
                cs = np.polyfit(np.mean(regs,axis=1), shifts[:,0], n_order)
                new_wave = self.wavelengths-np.polyval(cs,self.wavelengths)
            
            # Rebin spectrum to the new wavelength solution.
            for j in range(n_apertures):
                self.fluxes[i,:,j], self.flux_errors[i,:,j] = linear_rebin(new_wave, self.fluxes[i,:,j], self.flux_errors[i,:,j], wmin=self.wmin, wmax=self.wmax, wstep=self.wstep) #was wmin, wmax
    
    def model_align(self, cal_path, grism, ref_im, ref_ap, window, n_order, print_res=True):
        """
        Align observed spectra to a model reference spectrum.
        
        Parameters:
        -----------
        
        cal_path : str;
            Path at which calibration files are located.
            
        grism : str;
            Grism used for observations to identify regions to be compared.
            
        ref_im : int;
            Reference image from which the comparison spectrum was extracted.

        ref_ap : int;
            Reference aperture at which the comparison spectrum was extracted.
        
        window : int;
            Size of cross-correlation shifts in pixels.
            
        n_order : int;
            Order of the polynomial to which the shifts of the different regions are fitted. Recommended is 2.
            
        print_res : bool, optional;
            If True, prints the original resolution of the model spectrum and the data and shows a plot comparing wavelength solutions before and after alignment. Default is True.
        """
        # Obtain model spectrum and degrade it to the resolution of the data
        model_wave, model_flux, model_step = get_model(cal_path, self.wmin, self.wmax, print_res=print_res)
        model_norm = filter_spec(model_wave, model_flux, 2475)#
        interp_func = scipy.interpolate.CubicSpline(model_wave[np.isfinite(model_norm)], model_norm[np.isfinite(model_norm)], extrapolate=False)
        model_norm = interp_func(self.wavelengths)
        ref_norm = filter_spec(self.wavelengths,self.fluxes[ref_im,:,ref_ap],70)
        
        # Calculate new wavelength solution
        cont = get_regions(grism, self.wavelengths)
        cent = [np.mean(c) for c in cont]
        shifts = np.empty([len(cont),2])
        for i in range(len(cont)):
            inds = np.where((self.wavelengths>cont[i][0]) & (self.wavelengths<cont[i][1]))
            if len(inds[0]) < 3*window:
                inds = np.arange(inds[0][0]-window, inds[0][-1]+window+1)
            shifts[i] = ccf(ref_norm[inds],model_norm[inds],window,err=True,plot_bool=print_res)
        if np.nansum(shifts)!=0:
            cs = np.polyfit(cent, shifts[:,0]*wstep, n_order)#, w=1/shifts[:,1])
            new_wave = self.wavelengths + np.polyval(cs, self.wavelengths)
            if print_res:
                plt.errorbar(cent,shifts[:,0]*wstep,yerr=shifts[:,1]*self.wstep,fmt='.')
                plt.plot(self.wavelengths,new_wave-self.wavelengths)
                plt.plot(self.wavelengths, np.polyval(np.polyfit(cent,shifts[:,0]*wstep,1), self.wavelengths))
                plt.show()
        else:
            new_wave = self.wavelengths
        
        # Rebin spectra to the new wavelength solution
        ni,nj,nk = self.fluxes.shape
        for i in range(ni):
            for k in range(nk):
                self.fluxes[i,:,k], self.flux_errors[i,:,k] = linear_rebin(new_wave, self.fluxes[i,:,k], spec_error=self.flux_errors[i,:,k], wmin=self.wmin, wmax=self.wmax, wstep=self.wstep)
    
    def align_specs(self, cal_path, grism, n_order, print_stuff=False):
        """
        Align the spectra first to a reference spectrum within the dataset and then to a model spectrum.
        
        Parameters:
        -----------
        cal_path : str;
            Path at which calibration files are located.
            
        grism : str;
            Grism used for observations to identify regions to be compared.
            
        n_order : int;
            Order of the polynomial to which the shifts of the different regions are fitted. Recommended is 2.
        
        print_stuff : bool, optional;
            If True, prints the original resolution of the model spectrum and the data and shows a plot comparing wavelength solutions before and after alignment. Default is True.
        """
        print(f'Aligning star {self.number+1}')
        n_steps = int(np.ceil((self.wmax-self.wmin)/self.wstep))
        if n_steps != len(self.fluxes[0,:,0]):
            print('CHECK ARRAY LENGTHS')
        
        ref_im = 6
        ref_ap = 3
        window = 20
        
        lines = [4305,4668,4983,5269,7190,8230]+[3934.,5889.,6562.53]+[8498, 8542, 8662]
        new_lines = [l for l in lines if l-40>self.wmin and l+40<self.wmax]
        regs = np.array([[l-40.,l+40.] for l in new_lines])
        
        self.self_align(ref_ap, ref_im, regs, window, n_order)
        self.model_align(cal_path, grism, ref_im, ref_ap, window, n_order, print_res=print_stuff)
                    
def running_std(data, window=31):
    """
    Calculate the standard deviation over a running window.
    
    Parameters:
    -----------
    data : ndarray;
        Data over which the standard deviation is to be calculated.
    
    window : int;
        Window size over which standard deviation is calculated for each point in the data.
    """
    if len(data.shape) == 1:
        window = np.min([window,len(data)//3])
        std_vals = np.empty(len(data))
        full_std = robust.std(data[np.isfinite(data)])
        for i in range(len(data)):
            if i<window:
                sub_data = data[:i+window//2+1]
            else:
                sub_data = data[i-window//2:i+window//2+1]
            if np.sum(np.isfinite(sub_data))>10:
                 std_vals[i] = robust.std(sub_data[np.isfinite(sub_data)])
            elif np.sum(np.isfinite(data))>10:
                 std_vals[i] = robust.std(data[np.isfinite(data)])
            else:
                std_vals[i] = np.nanstd(data)
            std_vals[i] = np.max([std_vals[i],full_std])
    else:
        std_vals = np.empty(data.shape)
        if i<window//2:
            std_vals[i] = np.nanstd(data[:i+window//2+1],axis=0)
        else:
            std_vals[i] = np.nanstd(data[i-window//2:i+window//2+1],axis=0)
    return std_vals
    
def get_regions(grism_name, wave):
    """
    Identify regions in the spectrum where alignment should be done based on the grism used in the observations.
    
    Parameters:
    -----------
    grism_name : str;
        Name of the grism used during observation.
    
    wave : ndarray;
        Wavelength over which the observation is done.
    """
    if 'RI' in grism_name:
        regions = [[5171.4475,5818.8604], [5869.3082,5919.7559], [5932.3679,6491.4972], [6508.3132,6806.7958], [7706.4475,8114.2336], [8601.8953,8547.2436]]
    elif 'B' in grism_name:
        regions = [[3240.3512,3603.9152], [3603.9152,4000.9873], [4004.3381,4327.6922], [4327.6922,4640.9939], [4640.9939,4981.1021], [4981.1021,5312.8332], [5312.8332,5614.407], [5614.407,5937.7611], [5939.4365,6190.748]]
    elif 'z' in grism_name:
        regions = [[7317.7097,7554.6548], [7696.4334,7890.6507], [7958.6267,8251.8948], [8259.6635,8615.0811], [8741.3224,8960.7879], [9067.6074,9287.073]]
    elif 'R' in grism_name and 'I' not in grism_name:
        regions = [[5608.1774,5799.1553], [5805.564,5992.6968], [5999.1054,6179.8295], [6183.0338,6340.6868], [6366.9622,6527.1786], [6534.2282,6688.0359], [6713.6705,6859.7879], [6961.0446,7040.512], [7086.0134,7086.0134]]
    else:
        sys.exit('Grism not supported.')
    regions = [r for r in regions if r[0]>wave[0] and r[1]<wave[-1]]
    return regions

def telluric_regions(wave):
    """ Return regions with significant tellurics inside the specified wavelengths. """
    all_tells = [[3100.,3450.],[5100.,5120.],[5400.,5500.],[5666.,5777.],[5860.,6000.],[6260.,6330.],[6450.,6620.],[6860.,7450.],[7580.,7760.],[7840.,8610.],[8800.,10320.],[10600.,-1]]
    tells = [t for t in all_tells if t[0]>wave[0] and t[1]<wave[-1]] #TODO: FIX for telluric regions that overlap with the edge of the spectrum
    tell_inds = [[np.argmin(np.abs(wave-t[0])),np.argmin(np.abs(wave-t[1]))] for t in tells]
    if tell_inds[-1][-1]==0:
        tell_inds[-1][-1] = -1
    return tell_inds
    
def envelope(arr, window, split=False):
    """ Calculate an envelope over the spectrum in a certain window. """
    df=pd.DataFrame(data={"y":arr})
    df["y_upperEnv"]=df["y"].rolling(window=window).max().shift(int(-window/2))
    df["y_lowerEnv"]=df["y"].rolling(window=window).min().shift(int(-window/2))
    return np.array(df['y_lowerEnv']), np.array(df['y_upperEnv'])

def filter_spec(waves, spectrum, window):
    """ Filter spectrum by removing the continuum flux and normalising for wavelength calibration. """
    cs = robust_fit(waves, spectrum, 4)
    fit_spec = np.polyval(cs,waves)
    if np.sum(fit_spec)!=0:
        norm_spec = spectrum/fit_spec
    else:
        norm_spec = fit_spec
    spec_env = envelope(norm_spec, window=window)
    telluric_inds = telluric_regions(waves)
    for t in telluric_inds:
        spec_env[1][t[0]:t[1]] = np.nan
    interp_func = scipy.interpolate.CubicSpline(waves[np.isfinite(spec_env[1])], spec_env[1][np.isfinite(spec_env[1])], extrapolate=False)
    spec_env = interp_func(waves)
    spec_norm = norm_spec/spec_env
    
    return spec_norm
    
def get_model(cal_path, wmin, wmax, print_res=False):
    """ Read a model stellar spectrum from the calibration file and degrade it to data resolution. """
    model_files = sorted(glob.glob(cal_path+'*PHOENIX*'))
    model_wave = fits.getdata(model_files[0])
    model_wave = vac_to_air(model_wave)
    model_flux = fits.getdata(model_files[1])
    
    if 'RI' in grism:
        smooth = 650
    elif 'B' or 'z' in grism:
        smooth = 210
    else:
        sys.exit('Grism not supported.')
    model_flux = scipy.ndimage.gaussian_filter1d(model_flux, smooth, mode='nearest')
    
    model_flux = model_flux[(model_wave>=wmin) & (model_wave<=wmax)]
    model_wave = model_wave[(model_wave>=wmin) & (model_wave<=wmax)]
    model_step = np.median(model_wave - np.roll(model_wave,1))
    resolution = np.median(model_wave)/(model_step*smooth)
    if print_res:
        print('Resolution: ', resolution)
        print('Sigma: ', model_step*smooth)
        print('FWHM: ', model_step*smooth*2.3548)
        
    return model_wave, model_flux, model_step
                        
def ccf(vec1,vec2,rad,err=True,plot_bool=False):
    """
    Cross correlation with errors. Requires that vec1 and vec2 be the same length.
    Input:
        vec1: vector to be cross-correlated against. Has to be same length as vec2.
        vec2: vector that is being shifted in the cross-correlation. Has to be same length as vec1.
        rad:  maximum shift of vec2 when obtaining the cross-correlation function.
        err:  boolean. If true, returns an error on the calculated maximum of the ccf.
    Output:
        ccf_max: shift at which the ccf is maximised
        err_max: error on ccf_max. Only returned if err=True.
    """
    # Make sure the attempted shift is not bigger than the vectors themselves
    if len(vec1)<rad:
        rad = len(vec1)
    
    ccf_vec_len = len(vec1)-rad
    
    lags = np.arange(-rad,rad+1)
    ccf = np.zeros(len(lags))
    v1 = vec1[rad:-rad]
    for l in lags:
        v2 = vec2[rad+l:-rad+l]
        if rad == l:
            v2 = vec2[rad+l:]
        v2 = (v2-np.nanmean(v2))/np.nanstd(v2)*np.nanstd(v1)+np.nanmean(v1)
        good_inds = np.where(np.isfinite(v1)&np.isfinite(v2))
        ccf[l+rad] = scipy.signal.correlate(v1[good_inds], v2[good_inds], mode='valid')
    
    i_low = np.max([np.argmax(ccf)-3,0])
    i_upp = np.min([np.argmax(ccf)+3+1, 2*rad])
    if i_upp == 2*rad:
        i_upp = None
    sub_lag = lags[i_low:i_upp]
    sub_ccf = ccf[i_low:i_upp]
    
    if plot_bool:
        plt.plot(sub_lag,sub_ccf)
        plt.plot(lags,ccf)
    
    cs, cov = np.polyfit(sub_lag,sub_ccf,2,cov=True)
    cs_err = np.sqrt(np.diag(cov))
    
    ccf_max = np.max([-cs[1]/(2*cs[0]), -rad])
    ccf_max = np.min([ccf_max, rad])
    if plot_bool:
        plt.plot(np.linspace(sub_lag[0],sub_lag[-1],1000),np.polyval(cs,np.linspace(sub_lag[0],sub_lag[-1],1000)),color='k')
        plt.axvline(ccf_max)
        plt.show()
    err_max = np.abs(ccf_max / (2*cs[0])) * np.sqrt(cs[1]**2/(2*cs[0])**2 * cs_err[0]**2 + cs_err[1]**2)
    
    if err:
        return ccf_max, err_max
    else:
        return ccf_max
    
    
def vac_to_air(vac_waves):
    """
    Converts vacuum wavelengths to air wavelengths, using the IAU standard conversion formula by Morton, 2020, ApJ, taken from https://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion
    This solution is accurate to about 10m/s, is not valid at <2000Å
    Input:
        - vac_waves: wavelengths in vaccuum
    Output:
        - air_waves: wavelengths in air
    """
    s = 1e4/vac_waves
    n = 1 + 0.0000834254 + 0.02406147 / (130 - s**2) + 0.00015998 / (38.9 - s**2)
    air_waves = vac_waves/n
    return air_waves
                
def find_filter(grism):
    """ Find the order separation filter corresponding to the chosen grism. """
    if grism in ['600B_22','1400V_18','1200B_97']:
        filter = 'free_00'
    elif grism in ['300V_10', '1200R_93', '600RI_19']:
        filter = 'GG435_81'
    elif grism in ['300I_11','150I_17','1028z_29','600z_23']:
        filter = 'OG590_32'
    else:
        sys.exit('Invalid grism.')
        
    if grism == '150I_27':
        text_in = input('If the filter is not OG590, please input correct filter:\n'.format(filter))
        if '435' in text_in:
            filter = 'GG435_81'
        elif 'no' in text_in or 'No' in text_in:
            filter = 'free_00'
        else:
            print('Invalid filter, keeping OG590.')
            
    if grism == '300V_10':
        text_in = input('Is GG435 the correct filter?\n')
        if 'n' in text_in or 'N' in text_in:
            filter = 'free_00'
    
    return filter

def get_detector_parameters(chip1, chip2, reference_y=1246):
    """ Read relevant information on the chip from the file headers. """
    header1 = fits.getheader(chip1)
    header2 = fits.getheader(chip2)
    n_x = header1['NAXIS1']
    n_y = header1['NAXIS2']
    gains = [header1['ESO DET OUT1 GAIN'], header2['ESO DET OUT1 GAIN']]
    read_noise = [header1['ESO DET OUT1 RON'], header2['ESO DET OUT1 RON']]
    prescanx = header1['ESO DET OUT1 PRSCX']
    prescany = header1['ESO DET OUT1 PRSCY']
    overscanx = header1['ESO DET OUT1 OVSCX']
    overscany = header1['ESO DET OUT1 OVSCY']
    over_region = [n_y-overscany, n_y, overscanx, n_x-overscanx]
    pre_region = [0, prescany, prescanx, n_x-prescanx]
    data_region = [overscany, n_y-overscany, 0, n_x]
    
    w_factor = 35./280 #arcsec/pixel
    width_1 = np.array([int(header1['ESO INS MOS107 LEN']/w_factor)])
    width_2 = np.array([int(header1['ESO INS MOS108 LEN']/w_factor)])
    return n_x, n_y, gains, read_noise, [pre_region, over_region], data_region, [width_1, width_2]#, [y_1, y_2]
    
def get_grism_parameters(fname):
    """ Get the wavelength information from the grism table. """
    data = fits.getdata(fname)
    w_start = data['startwavelength'][0]
    w_end = data['endwavelength'][0]
    step = data['dispersion'][0]
    return w_start, w_end, step
    
def get_line_parameters(fname):
    """ Get calibration line information. """
    data = fits.getdata(fname)
    wave = data['WLEN']
    ion = data['CHEMICAL_ION']
    set = data['LINE_SET']
    return wave, ion, set
    
def read_files(path, name, filt, mask, n_lists):
    """ Read list of file names. """
    part_path = '/'.join(path.split('/')[:-3])+'/'
    file_lists = [None]*n_lists
    for i in range(n_lists):
        file_lists[i] = np.loadtxt(path+name+str(i+1)+mask+'.txt',dtype=str)#str(night)+'_'
        if file_lists[i].size == 1:
            file_lists[i] = np.array([file_lists[i]])
        file_lists[i] = [part_path+f[6:] for f in file_lists[i]]
    return file_lists
    
def get_border(scan, nx):
    """ Identify the border of the chip. """
    if scan:
        border = (scan[1]-scan[0])*2
    elif nx > 3000:
        border = 20
    else:
        border = 10
    return border
    
def make_mastercube(flist, nx, ny, datareg, scancor=None):
    """
    Identify parts of the image that contain data and combine all images into a cube and a master image.
    
    Parameters:
    -----------
    flist : list;
        List of filenames containing the data images.
    
    nx, ny : ints;
        Size of the chip.
        
    datareg : list;
        Borders in x and y pixels of the part of the chip that contains information.
    
    scancor : list;
        If it exists, corrects for over- or under-scan regions.
    """
    n_files = len(flist)
    border = get_border(scancor, nx)
    
    cube = np.empty([n_files, ny-border, nx])
    for i in range(n_files):
        image = fits.getdata(flist[i])
        data_image = image[datareg[0]:datareg[1],datareg[2]:datareg[3]]

        if scancor:
            scan_region = image[scancor[0]:scancor[1],scancor[2]:scancor[3]]
            scan_mean = robust.mean(scan_region, 3)
            data_image = data_image - scan_mean
        cube[i] = data_image
        
    sorted_cube = np.sort(cube,axis=0)
    if n_files > 10:
        master_image = np.mean(sorted_cube[3:-3],axis=0)
    else:
        master_image = np.mean(sorted_cube, axis=0)
    
    return master_image

def fit_tophat(data, center, width, plot=False, title=''):
    """ Fit a tophat to the data given intial guessed parameters. """
    x = np.arange(len(data))
    bg_guess = np.median(np.append(data[:int(center-width)],data[int(center+width):]))
    flux_guess = np.median(data[int(center-width/2):int(center+width/2)]) -  bg_guess
    x_guess = np.mean(np.where(data-bg_guess>flux_guess/2))
    
    t_init = models.Box1D(amplitude=flux_guess, x_0=x_guess, width=width)
    fit_t = fitting.LevMarLSQFitter()
    t = fit_t(t_init, x, data-bg_guess)
    if plot:
        plt.plot(x, data, color='k')
        plt.plot(x, t(x)+bg_guess, color='r')
        plt.title(title)
        plt.pause(.01)
        plt.clf()
    
    return t.x_0-t.width/2, t.x_0+t.width/2, t.amplitude.value
    
def trace_slits(centers, widths, image):
    """ Trace the borders of the slits in an image by fitting a tophat to each column. """
    slit_top = centers + widths/2
    slit_bottom = centers - widths/2
    slit_flux = np.ones(len(slit_top))
    
    box_pars = np.zeros([image.shape[1],len(centers),3])
    for i in range(len(centers)):
        plt.ion()
        for j in range(image.shape[1]):
            col = image[:,j]
            low = int(slit_bottom[i])-30
            if low > 0:
                col[0:low] = np.median(image[100:150]) # WHY THESE VALUES?
            high = int(slit_top+30)
            if high < image.shape[0]:
                col[high:] = np.median(image[100:150])
            box_pars[j,i] = fit_tophat(col, centers[i], widths[i], plot=True, title='Column '+str(j))
            if box_pars[j,i][2]<1:
                box_pars[j,i] = np.nan
        plt.close()
        plt.ioff()
        
        for j in range(3): #lower, upper, flux
            clipped = stats.sigma_clip(box_pars[:,i,j], sigma=5, masked=True)
            box_pars[:,i,j] = clipped.filled(np.nanmedian(box_pars[:,i,j]))
            spl_func = scipy.interpolate.UnivariateSpline(np.arange(box_pars.shape[0]), box_pars[:,i,j]) #scipy.signal.medfilt(box_pars[:,i,j], kernel_size=101)
            box_pars[:,i,j] = spl_func(np.arange(box_pars.shape[0]))
    return box_pars.T

def set_dark_zero(image, lower_bounds, upper_bounds):
    """ Set areas outside the slits to 0 in the image, given the slit boundaries. """
    ny, nx = image.shape
    y_inds = np.arange(ny)
    with0 = np.copy(image)
    withNan = np.copy(image)
    
    for i in range(nx):
        inslit = np.zeros(ny)
        for j in range(lower_bounds.shape[0]):
            inslit[np.where((y_inds>lower_bounds[j,i])&(y_inds<upper_bounds[j,i]))] = 1
        with0[np.where(inslit==0),i] = 0.
        withNan[np.where(inslit==0),i] = np.nan
    
    return with0, withNan

def make_masterflat(flist, nx, ny, datareg, masterbias, slit_c, slit_w, scancor=None):
    """
    Create a master flat image from the observed flat images.
    
    Parameters:
    -----------
    flist : list;
        List of images to be included in the master flat.
        
    nx, ny : ints;
        Size of the images.
        
    datareg : list;
        Boundaries of the area of the chip that contains information.
        
    masterbias : ndarray;
        Master bias image.
        
    slit_c, slit_w : ints;
        Estimated centers and widths of the slits in the images.
    
    scancor : any, optional;
        If values for the over- or under-scan region are given this is corrected for. Default is None.
    """
    combined_flat = make_mastercube(flist, nx, ny, datareg, scancor)
    masterflat = combined_flat - masterbias
    fits.writeto('/Users/dominiquepetit/Documents/WASP-69/IDL_output/IDLpy_flat_basicmaster.fits', masterflat, overwrite=True)
    
    lower_bounds, upper_bounds, amplitudes = trace_slits(slit_c, slit_w, masterflat)
    with0, withNan = set_dark_zero(masterflat, lower_bounds, upper_bounds)
    fits.writeto('/Users/dominiquepetit/Documents/WASP-69/IDL_output/IDLpy_flat0.fits', with0, overwrite=True)
    fits.writeto('/Users/dominiquepetit/Documents/WASP-69/IDL_output/IDLpy_flatNan.fits', withNan, overwrite=True)
    
    flat_mean = robust.mean(withNan[np.where(np.isnan(withNan)==False)], 3)
    norm_master = masterflat/flat_mean
    norm0, normNan = set_dark_zero(norm_master, lower_bounds, upper_bounds)
    fits.writeto('/Users/dominiquepetit/Documents/WASP-69/IDL_output/IDLpy_flat0_norm.fits', norm0, overwrite=True)
    fits.writeto('/Users/dominiquepetit/Documents/WASP-69/IDL_output/IDLpy_flatNan_norm.fits', normNan, overwrite=True)
    
    smooth10 = scipy.ndimage.median_filter(norm_master, size=10, mode='reflect')
    smooth50 = scipy.ndimage.median_filter(norm_master, size=50, mode='reflect')
    fits.writeto('/Users/dominiquepetit/Documents/WASP-69/IDL_output/IDLpy_flat_smooth10.fits', smooth10, overwrite=True)
    fits.writeto('/Users/dominiquepetit/Documents/WASP-69/IDL_output/IDLpy_flat_smooth50.fits', smooth50, overwrite=True)
    
    norm10 = norm_master/smooth10 #normalised flat divided by smoothed normalised flat is going to give only high frequency variations - is this present in IDL?
    norm50 = norm_master/smooth50
    norm010, normNan10 = set_dark_zero(norm10, lower_bounds, upper_bounds)
    norm050, normNan50 = set_dark_zero(norm50, lower_bounds, upper_bounds)
    fits.writeto('/Users/dominiquepetit/Documents/WASP-69/IDL_output/IDLpy_flat_final.fits', np.array([norm010,normNan10,norm050,normNan50]), overwrite=True)
    
    return normNan10, lower_bounds, upper_bounds



def onclick(event):
    """ Close plot after clicking. """
    global xvals
    xvals.append(event.xdata)
    fig.canvas.mpl_disconnect(cid)
    plt.close()
    return
    
def click_plot(x1, y1, x2, xval, threshold=3e3, click_bool=False):
    """
    Show wavelength calibration plot and either identifies calibration lines by taking the closest line over the threshold for automatic calibration or prompts the user to click the indicated line for manual wavelength calibration.
    
    Parameters:
    -----------
    x1, y1 : ndarrays;
        X and Y coordinates of the calibration spectrum.
        
    x2 : float;
        Estimated line position.
        
    xval : float;
        Wavelength of the line that is to be identified.
        
    threshold : float, optional;
        When using automatic calibration, the closest line with an intensity over the threshold is assumed to be the calibration line. Default is 3000.
    
    click_bool : bool, optional;
        If True, manual calibration is done. Else automatic calibration. Default is False.
    """
    if click_bool:
        global fig
        global cid
        
        print('Click the line at',xval,'Å, or left of the data if line not shown')
        fig = plt.figure()
        fig.set_size_inches(10,5)
        ax = fig.add_subplot(111)
        ax.plot(x1,y1,color='k')
        ax.scatter(x2,np.zeros(len(x2)),marker='x',color='r')
        ax.axvline(xval,zorder=-10)
        cid = fig.canvas.mpl_connect('button_press_event',onclick)
        plt.show()
    else:
        line_inds = np.where(y1>threshold)[0] # was 3.3, double check both wavelength solutions
        oldxval = np.argmin(np.abs(x1-xval))
        newxval = np.argmin(np.abs(line_inds-oldxval))
        global xvals
        xvals.append(x1[line_inds[newxval]])
    
def fit_moffat(x,y, plot=False):
    """ Fit a Moffat function to a line and possibly show the result. """
    nan_vals = np.isnan(y)
    if np.sum(nan_vals)!=0:
        x = x[nan_vals==False]
        y = y[nan_vals==False]
    t_init = models.Moffat1D(amplitude=np.max(y)-np.min(y), x_0=np.mean(x))
    fit_t = fitting.LevMarLSQFitter()
    t = fit_t(t_init, x, y)
    if plot:
        model_x = np.linspace(x[0],x[-1],1000)
        plt.plot(x,y,color='k')
        plt.plot(model_x,t(model_x),color='r')
        plt.title(t.x_0.value)
        plt.show(block=False)
        plt.pause(.2)
        plt.close()
        plt.clf()
    return t
    
def polynomial(coeffs, x, y):
    """ Polynomial function for robust fitting. """
    value = 0
    for i in range(len(coeffs)):
        value += coeffs[i] * x**(len(coeffs)-1-i)
    return value - y
    
def robust_fit(x,y,order=3, f_scale=None):
    """ Polynomial fit using a more statistically robust least squares algorithm. """
    if type(x) == list:
        x = np.array(x)
    if type(y) == list:
        y = np.array(y)
    good_inds = np.where(np.isfinite(x)&np.isfinite(y))
    x = x[good_inds]
    y = y[good_inds]
    if not f_scale:
        f_scale = np.max([robust.std(y),1e-4])
    guess_coeffs = np.polyfit(x,y,order)
    res_robust = least_squares(polynomial, guess_coeffs, loss='soft_l1', f_scale=f_scale, args=(x,y))
    cs = res_robust.x
    return cs

def new_solution(oldx, newx, allx, ally, inds=None, plot=False):
    """ Calculate a new wavelength solution based on the location of spectral lines in the calibration file. """
    if inds is not None:
        if type(oldx)==list or type(newx)==list:
            oldx = np.array(oldx)
            newx = np.array(newx)
        cs = robust_fit(oldx[inds],newx[inds],4)
        if plot:
            plt.scatter(oldx[inds],newx[inds],color='k')
    else:
        cs = robust_fit(oldx,newx,4) #np.polyfit(oldx, newx, 4)
        if plot:
            plt.scatter(oldx,newx,color='k')
    
    solution = np.polyval(cs,allx)
    
    if plot:
        highres_x = np.linspace(oldx[0],oldx[-1],1000)
        plt.plot(highres_x, np.polyval(cs,highres_x),color='r')
        plt.show(block=False)
        plt.pause(2)
        plt.close()
        plt.clf()
        
        plt.plot(solution,ally,color='k')
        for l in newx: plt.axvline(l,color='green',zorder=-1)
        plt.show(block=False)
        plt.pause(5)
        plt.close()
        plt.clf()
    
    return solution

def plot_window(x,y,lines):
    """ Plot a section of the calibration spectrum. """
    i, maxval = 0,0
    plt.ion()
    while maxval < np.max(x):
        minval = np.min(x)+50*i
        maxval = minval + 50
        plt.plot(x,y,color='k')
        [plt.axvline(l,color='green',zorder=-1) for l in lines]
        plt.xlim([minval,maxval])
        #plt.show(block=False)
        plt.pause(.1)
        #plt.close()
        plt.clf()
        i+=1
    plt.close()
    plt.ioff()

def check_line_fit(fitted, wmin, wmax, amp, manual):
    """ Decide if the fit of a given line is reliable enough to include in the new wavelength solution. """
    if manual:
        typed = input('Add line? (y/n)')
        if typed == 'y':
            success = True
        else:
            success = False
    else:
        if fitted.x_0.value>wmin and fitted.x_0.value<wmax and fitted.amplitude.value>amp:
            success = True
        else:
            success = False
    return success

def fitted_wavelengths(current_wave, theory_vals, window_width, spectrum, manual, print_status=False, mode='valid'):
    """ Get the parameters of lines that have been accurately identified in the calibration image.
    
    Parameters:
    -----------
    current_wave : ndarray;
        Wavelenght solution before calibration.
        
    theory_vals : ndarray;
        Estimated line positions.
        
    window_width : int;
        Width of region around the line where the fit is performed.
        
    spectrum : ndarray;
        Calibration spectrum fluxes.
        
    manual : bool;
        If True, calibration is done manually. If False, it is doen automatically.
        
    print_status : bool, optional;
        If True, print progress statements.
        
    mode : str, optional;
        If 'valid', badly fit lines are ignored. If 'same' they are included as NaNs.
    """
    fitted_x, fitted_l = [],[]
    for j in range(len(theory_vals)):
        ind_guess = np.argmin(np.abs(current_wave-theory_vals[j])) # Index of estimated line position
        x_guess = current_wave[ind_guess] # Estimated line position
        x_inds = np.where((current_wave>x_guess-window_width) & (current_wave<x_guess+window_width)) # Indices of points passed to fit
        fit = fit_moffat(current_wave[x_inds],spectrum[x_inds],plot=print_status)
        success = check_line_fit(fit, current_wave[x_inds[0][0]], current_wave[x_inds[0][-1]], 1e3, manual)
        if success:
            fitted_x = fitted_x + [fit.x_0.value]
            fitted_l = fitted_l + [theory_vals[j]]
            if print_status:
                print('line at '+str(int(np.round(theory_vals[j])))+'Å added')
        elif mode=='same':
            fitted_x = fitted_x + [np.nan]
            fitted_l = fitted_l + [np.nan]
        elif print_status:
            print('line at '+str(int(np.round(theory_vals[j])))+'Å skipped')
    return fitted_x, fitted_l

def solve_wave(arcfile, upper_bounds, lower_bounds, line_file, grism_file, nx, ny, slit_c, manual_mode=False, custom_list=None):
    """
    Create a wavelength solution for the calibration file.
    A first solution is found based on a number of lines listed in the calibration file that can be identified either automatically or manually in the calibration spectrum.
    This solution is further improved by fitting all the lines in the spectrum and then iterated over.
    
    Parameters:
    -----------
    arcfile : str;
        Name of the file with the calibration spectrum.
        
    upper_bounds, lower_bounds : lists;
        Slit boundaries.
        
    line_file : str;
        Calibration file that has a list of identified lines.
        
    grism_file : str;
        File containing the grism table.
        
    nx, ny : ints;
        Size of images.
    
    slit_c : list;
        Centers of the slits present in the image.
        
    manual_mode : bool, optional;
        If True, calibrations are done manually. If False, they are done automatically. Default is False.
    
    custom_list : str, optional;
        File containing a custom list of identifiable lines, to be used instead of ones specified in this function. Default is None.
    """
    n_slits = len(slit_c)
    wmin, wmax, wstep = get_grism_parameters(grism_file)
    waves = wmin+np.arange(nx)*wstep
    line_waves, line_ions, line_sets = get_line_parameters(line_file)
    if 'RI' in grism_file:
        click_lines = [5460.75, 5875.82, 6096.16, 6929.47, 7245.17, 7948.18, 8264.52] #These lines are valid when the real solution is shifted to the right of the theoretical values. If reducing other data in this filter, check that this is still valid!
        threshold = 3e3
    elif 'B' in grism_file:
        click_lines = [4046.56, 4678.1, 5085.8, 5460.75, 5875.62]
        threshold = 2500
    elif 'z' in grism_file:
        click_lines = [7438.9, 7635.11, 7724.21, 7948.18,  9224.50, 9657.78]
        threshold = 3e3
    elif custom_list:
        click_lines = np.loadtxt(custom_list)
    else:
        print("GRISM NOT SUPPORTED: please add a list of identifiable lines")
        sys.exit()
    solved_file = arcfile[:-4]+'lam.fits'
    xval_array = np.zeros([n_slits,len(click_lines)]) # Real fitted x values of the clearest lines for every slit
    data, header = fits.getdata(arcfile, header=True)
    solved_image = np.zeros(data.shape)
    rough_width = 20
    peak_width = 30/header['ESO DET WIN1 BINY']
    
    # For every slit in the image:
    for i in range(n_slits):
        # Extract the spectrum from the data
        rough_area = data[int(slit_c[i]-rough_width//2):int(slit_c[i]+rough_width//2)]
        rough_spec = robust.mean(rough_area, 5., axis=0)
        rough_err = np.sqrt(rough_spec)/np.sqrt(rough_width)
        plt.plot(waves,rough_spec,color='k')
        plt.axhline(threshold,color='green')
        [plt.axvline(c,color='green') for c in click_lines]
        plt.show(block=False)
        plt.pause(5)
        plt.close()
        plt.clf()
        # Obtain first wavelength solution based on identification of the lines in the click_lines list
        global xvals # Global list to enable clicking in the plot
        xvals = [] # Rough real x values of the clearest lines
        for j in range(len(click_lines)):#,,
            click_plot(waves, rough_spec, line_waves, click_lines[j], threshold=threshold, click_bool=manual_mode)
            if xvals[j]<wmin or xvals[j]>wmax: # Replace invalid values with NaN
                xvals[j] = np.nan
            else: # Fit a Moffat function to find the real value
                x_inds = np.where((waves>xvals[j]-peak_width) & (waves<xvals[j]+peak_width))
                fit = fit_moffat(waves[x_inds],rough_spec[x_inds],plot=True)
                xval_array[i,j] = fit.x_0.value
        good_inds = np.where((xval_array[i]>wmin)&(xval_array[i]<wmax))[0]
        first_solution = new_solution(xval_array[i], click_lines, waves, rough_spec, inds=good_inds, plot=True)
        while(np.any(first_solution)<=0 or np.min(first_solution)<wmin-2000):
            print('Wavelength calibration has failed. Please select the following lines by hand:')
            for j in range(len(click_lines)):#,,
                print(click_lines[j])
                click_plot(waves, rough_spec, line_waves, click_lines[j], threshold=threshold, click_bool=True)
                if xvals[j]<wmin or xvals[j]>wmax: # Replace invalid values with NaN
                    xvals[j] = np.nan
                else: # Fit a Moffat function to find the real value
                    x_inds = np.where((waves>xvals[j]-peak_width) & (waves<xvals[j]+peak_width))
                    fit = fit_moffat(waves[x_inds],rough_spec[x_inds],plot=True)
                    xval_array[i,j] = fit.x_0.value
            good_inds = np.where((xval_array[i]>wmin)&(xval_array[i]<wmax))[0]
            first_solution = new_solution(xval_array[i], click_lines, waves, rough_spec, inds=good_inds, plot=True)
            
        
        # Obtain second wavelength solution based on applying the first to all the lines available in the wavelength range of the data
        line_inds = np.where((line_waves>first_solution[0])&(line_waves<first_solution[-1])) # Indices of all lines in the calculated real wavelength range
        cut_lines = line_waves[line_inds]
        cut_ions = line_ions[line_inds]
        cut_sets = line_sets[line_inds]
        xvals, lvals = fitted_wavelengths(first_solution, cut_lines, rough_width, rough_spec, manual_mode, print_status=True)
        # Second iteration of wavelength solution, this time using all the succesfully fitted lines available. No need to specify indices, as xvals and lvals already only include lines where the fitting was succesful.
        second_solution = new_solution(xvals, lvals, first_solution, rough_spec, plot=True)
        
        # Plot the spectrum in parts to see if calculated wavelengths of lines now match up with the data
        plot_window(second_solution, rough_spec, lvals)
        
        # Obtain a wavelength solution for each row of pixels in the slit, to account for any potential tilted/curved lines
        bottom_part_row = int(np.ceil(np.min(lower_bounds)))
        bottom_full_row = int(np.ceil(np.max(lower_bounds)))
        top_part_row = int(np.floor(np.max(upper_bounds)))
        top_full_row = int(np.floor(np.min(upper_bounds)))
        complete_rows = np.arange(bottom_full_row,top_full_row+1) #Rows of pixels in the slit that are complete. I.e. the whole row is taken up by slit pixels
        partial_rows = np.append(np.arange(top_full_row+1,top_part_row+1), np.arange(bottom_part_row,bottom_full_row)) #+1 Rows that are only partially covered by the slit
        n_complete = len(complete_rows)
        line_positions = np.zeros([n_complete,len(lvals)]) # To be filled with the calculated line positions of each complete row, previously len(cut_lines) for second dimension
        all_lines = np.zeros([ny,len(lvals)]) # Same but for all rows, so it can include partial rows
        for j in range(n_complete):
            row = data[complete_rows[j]]
            line_positions[j], new_lvals = fitted_wavelengths(second_solution, lvals, rough_width, row, manual_mode, print_status=False, mode='same')  # rough_spec, where is row
            new_lvals = np.array(new_lvals)
            if len(line_positions[j][np.where(np.isfinite(line_positions[j]))]) > len(line_positions[j])/2:
                third_solution = new_solution(line_positions[j][np.where(np.isfinite(line_positions[j]))], new_lvals[np.where(np.isfinite(new_lvals))], second_solution, row)
            else:
                third_solution = second_solution.copy()
            all_lines[complete_rows[j]] = line_positions[j] # Line positions
            solved_image[complete_rows[j]] = third_solution # Wavelength solution
            print(j+1,'/',n_complete,end='\r')
            
        all_lines[np.where(all_lines==0)] = np.nan
        
        # Fit lines in the y direction with a polynomial to account for any potential tilted/curved lines
        yfit_lines = np.zeros(all_lines.shape) # All_lines, because partial rows should be included
        sigs = np.zeros(all_lines.shape[1])
        plt.ion()
        for j in range(all_lines.shape[1]):
            inds = np.where(np.isfinite(all_lines[complete_rows,j]))
            cs = robust_fit(complete_rows[inds], all_lines[complete_rows,j][inds], 3)
            yfit_lines[complete_rows,j] = np.polyval(cs, complete_rows)
            diff = yfit_lines[complete_rows,j]-all_lines[complete_rows,j]
            sigs[j] = robust.std(diff[np.isfinite(diff)])
            plt.scatter(all_lines[:,j],np.arange(ny),color='k')
            plt.plot(yfit_lines[complete_rows,j], complete_rows, color='r')
            plt.title('Arc line {0} fit in the y direction'.format(j))
            plt.pause(.1)
            plt.clf()
        plt.close()
        plt.ioff()
            
        clean_line_inds = np.where(sigs < 2*np.nanmedian(sigs))
        solved_image[np.where(solved_image==0)] = np.nan
        bad_count = 0
        # Obtain final wavelength solution for each row, taking into account the fitted curvature of each line
        for j in complete_rows:
            wave_row = solved_image[j]
            flux_row = data[j]
            if np.sum(np.isnan(all_lines[j])) < len(all_lines[j])/2:
                solved_image[j] = new_solution(all_lines[j], yfit_lines[j], wave_row, flux_row, inds=clean_line_inds, plot=False) #lvals, rough_spec, third_solution, inputs: oldx, newx, allx, ally,
            else:
                solved_image[j] = 0
                bad_count += 1
        if bad_count == 0:
            print('Finding a wavelength solution was succesful for all rows.')
        else:
            print('Finding a wavelength solution was unsuccesful for {0} out of {1} rows.'.format(bad_count, len(complete_rows)))
                
        # Extrapolate line positions to rows of pixels that are only partially covered by the slit
        for j in range(nx):
            good_inds = np.where(solved_image[complete_rows,j]!=0)
            cs = robust_fit(complete_rows[good_inds], solved_image[complete_rows,j][good_inds], 3)
            solved_image[partial_rows,j] = np.polyval(cs, partial_rows)
            solved_image[complete_rows,j] = np.polyval(cs,complete_rows)

    return solved_image

def linear_rebin(waves, spectrum, spec_error=None, wmin=None, wmax=None, wstep=None, conserve_flux=False):
    """ Linear interpolation of the spectrum from an old wavelength solution to a new one.
    
    Parameters:
    -----------
    waves, spectrum : ndarrays;
        Wavelengths and spectrum before the rebinning.
    
    spec_error : ndarray, optional;
        Errors of the spectral flux. Default is None.
    
    wmin, wmax, wstep : floats, optional;
        Parameters to calculate the new wavelength solution to which the spectrum is to be binned. Defaults are None, in which case the old solution is used.
        
    conserve_flux : bool, optional;
        If True, flux convservation is included. Default is False.
    """
    
    if not wmin:
        wmin = np.min(waves)
    if not wmax:
        wmax = np.max(waves)
    if not wstep:
        wstep = (wmax-wmin)/len(waves)
    
    if wmin-wstep/2 > waves[-1]:
        print(wmin,wmax,wstep,waves)
        print('Start value is not in valid range')
        sys.exit()
    
    if spec_error is not None:
        spec_error = spec_error[np.isfinite(spectrum)]
    waves = waves[np.isfinite(spectrum)]
    spectrum = spectrum[np.isfinite(spectrum)]
    
    n_out = int(np.ceil((wmax-wmin)/wstep))
    new_waves = np.arange(n_out)*wstep + wmin
    interp_func = scipy.interpolate.CubicSpline(waves,spectrum,extrapolate=False)
    new_spec = interp_func(new_waves)
    if spec_error is not None:
        variance_func = scipy.interpolate.CubicSpline(waves[np.isnan(spec_error)==False],spec_error[np.isnan(spec_error)==False]**2,extrapolate=False)
        variance = variance_func(new_waves)
        new_error = np.sqrt(np.abs(variance))
    if conserve_flux:
        flat_func = scipy.interpolate.CubicSpline(waves,np.ones(spectrum.shape),extrapolate=False)
        flat = flat_func(new_waves)
        new_spec = new_spec / flat
        if spec_error is not None:
            new_error = new_error / flat
    if spec_error is not None:
        return new_spec, new_error
    else:
        return new_spec

def adjust_2d(data, wavelength_solution, grism_file):
    """ Change a 2D image to a new wavelength solution through line-by-line interpolation. """
    wmin, wmax, wstep = get_grism_parameters(grism_file)
    n_rows = data.shape[0]
    n_columns = int(np.ceil((wmax-wmin)/wstep))
    new_data = np.zeros([n_rows, n_columns])
    
    for i in range(n_rows):
        good_inds = np.where(np.isfinite(wavelength_solution[i])&np.isfinite(data[i]))
        data_nonan = data[i][good_inds]
        wave_solution_nonan = wavelength_solution[i][good_inds]
        if len(wave_solution_nonan)>len(wavelength_solution[i])//2:
            new_data[i] = linear_rebin(wave_solution_nonan, data_nonan, wmin=wmin, wmax=wmax, wstep=wstep, conserve_flux=False)
        else:
            new_data[i] = np.nan
    return new_data
    
def make_masterarc(arclist, masterbias, nx, ny, slic, datareg, lower_bounds, upper_bounds, file_path, linecat, grism_file, master_file, scancor=None):
    """ Create a master calibration arc lamp file.
    
    Parameters:
    -----------
    arclist : list;
        List of arclamp files to be included.
    
    masterbias : ndarray;
        Bias image.
        
    nx, ny : ints;
        Image size.
        
    slic : list;
        Slit center locations.
        
    datareg : list;
        Boundaries of the part of the chip where information is located.
        
    lower_bounds, upper_bounds : lists;
        Boundaries of the slits on the detector.
        
    file_path : str;
        Location of the arc lamp files.
    
    linecat : str;
        Calibration line catalogue file name.
        
    grismfile : str;
        File name for grism table.
        
    master_file : str;
        Name of the master arc lamp file.
        
    scancor : list, optional;
        If coordinates are given, corrects for over- or under-scan regions. Default is None.
    """
    border = get_border(scancor, nx)
    cor_arclist = [file_path+a.split('/')[-1][:-4]+'red.fits' for a in arclist]
    for i in range(len(arclist)):
        red_arc = make_mastercube([arclist[i]], nx, ny, datareg, scancor)
        red_arc = red_arc - masterbias
        arc0, arcNan = set_dark_zero(red_arc, lower_bounds, upper_bounds)
    if len(arclist)>1:
        master_arc = make_mastercube(cor_arclist, nx, ny, datareg, scancor)
    else:
        master_arc = arcNan
    fits.writeto(master_file, master_arc, header=fits.getheader(arclist[0]), output_verify='ignore', overwrite=True)

    wavelength_solution = solve_wave(master_file, upper_bounds, lower_bounds, linecat, grism_file, nx, ny, slic)
    new_master_arc = adjust_2d(master_arc, wavelength_solution, grism_file)
    
    return wavelength_solution, new_master_arc
            
def write_steps(original, new_path, step1, step2, step3):
    """ Write out intermediate reduction steps. """
    hdr = fits.getheader(original)
    new_file = new_path+original.split('/')[-1][:-4]
    fits.writeto(new_file+'red.fits', step1, hdr, overwrite=True, output_verify='ignore')
    fits.writeto(new_file+'redcos.fits', step2, hdr, overwrite=True, output_verify='ignore')
    fits.writeto(new_file+'redcoslam.fits', step3, hdr, overwrite=True, output_verify='ignore')

def extract_moffat(data, sky_inds, y_inds, plot_bool=False):
    """
    Identify the location of the spectral trace and remove the sky background from a column.
    
    Parameters:
    -----------
    data : ndarray;
        A column of the science image.
        
    sky_inds : ndarray;
        Location of the sky background area.
        
    y_inds : ndarray;
        Pixel indices.
        
    plot_bool : bool, optional;
        If True, plots are shown. Default is False.
    """
    sig_val = 5
    sky_column = data[sky_inds]
    
    if np.sum(np.isnan(sky_column)) > 0.5*len(sky_column):
        return np.nan, np.nan, np.nan
    sky_inds = sky_inds[np.isfinite(sky_column)]
    sky_column = sky_column[np.isfinite(sky_column)]
    
    sky_mean = robust.mean(sky_column,sig_val)
    sky_sig = robust.std(sky_column)
    bad_inds = np.where((data<sky_mean-sig_val*sky_sig)|(data>120000)|(data==0))
    data[bad_inds] = np.nan
    
    if np.sum(np.isfinite(data)) > 50:
        cs = robust_fit(sky_inds, sky_column, 1)
        data = data - np.polyval(cs, y_inds)
        
        fit_func = fit_moffat(y_inds, data, plot=plot_bool)
        residual = data-fit_func(y_inds)
        bad_inds = np.where(residual>sig_val*np.std(residual))
        if len(bad_inds[0]) > 0:
            data[bad_inds] = fit_func(bad_inds)
            
            fit_func = fit_moffat(y_inds, data)
            residual = data-fit_func(y_inds)
            bad_inds = np.where(residual>sig_val*np.std(residual))
            data[bad_inds] = fit_func(bad_inds)
            
        background = np.polyval(cs, fit_func.x_0)
    else:
        fit_func = 0.
        background = 0
    
    return fit_func, background, data

def thorough_smooth(data, size=50, sigma=3, plot=True):
    """
    Smooth a time series of data.
    
    Parameters:
    -----------
    data : ndarray;
        Time series to be smoothed out.
    
    size : int, optional;
        Window over which the filter is applied. Default is 50.
        
    sigma : int, optional;
        Threshold for outlier identification. Default is 3.
    
    plot : bool, optional;
        If True, plots are shown. Default is True.
    """
    new_data = data.copy()
    bad_inds = np.where(np.abs(data-np.nanmedian(data))>10*np.nanstd(data))
    new_data[bad_inds] = np.nan
    good_inds = np.where(np.isfinite(new_data))
    data_filt = scipy.ndimage.uniform_filter1d(new_data[good_inds], size=size)
    data_filt = scipy.ndimage.uniform_filter1d(data_filt, size=size)
    new_data[good_inds] = data_filt
    
    if plot:
        plt.scatter(np.arange(len(data)),data,marker='o',color='k')
        plt.scatter(good_inds, data[good_inds], marker='.',color='r')
        plt.scatter(np.arange(len(data)),new_data, marker='*',color='green')
        plt.show(block=False)
        plt.pause(0.05)
        plt.close()
        plt.clf()
        
    return new_data
    
def fix_traces(waves, lower_bounds, upper_bounds, wmin, wmax, wstep):
    """
    Correct the boundaries of the traces on the detector for new wavelength solutions.
    
    Parameters:
    -----------
    waves : ndarray;
        Wavelengths of the old wavelength solution.
        
    upper_bounds, lower_bounds : ndarrays;
        Locations of the upper and lower bounds of the traces on the detector.
        
    wmin, wmax, wstep : ints;
        Parameters for the new wavelength solutions.
    """
    new_bounds = [[], []]
    for i in range(lower_bounds.shape[0]):
        as_inds = np.array(np.round(lower_bounds[i]),dtype=int)
        wave_int = np.array([waves[as_inds[i],i] for i in range(len(as_inds))])
        good_inds = np.where(np.isfinite(wave_int)&np.isfinite(lower_bounds[i]))
        new_lower = linear_rebin(wave_int[good_inds], lower_bounds[i][good_inds], wmin=wmin, wmax=wmax, wstep=wstep)
        nnew = int(np.ceil((wmax-wmin)/wstep))
        new_wave = wmin+wstep*np.arange(nnew)
        int_func = scipy.interpolate.interp1d(new_wave[np.isfinite(new_lower)],new_lower[np.isfinite(new_lower)],'cubic',fill_value='extrapolate')
        new_bounds[0] = new_bounds[0]+[int_func(new_wave)]
        
        as_inds = np.array(np.round(upper_bounds[i]),dtype=int)
        wave_int = np.array([waves[as_inds[i],i] for i in range(len(as_inds))])
        good_inds = np.where(np.isfinite(wave_int)&np.isfinite(upper_bounds[i]))
        new_upper = linear_rebin(wave_int[good_inds], upper_bounds[i][good_inds], wmin=wmin, wmax=wmax, wstep=wstep)
        int_func = scipy.interpolate.interp1d(new_wave[np.isfinite(new_upper)],new_upper[np.isfinite(new_upper)],'cubic',fill_value = 'extrapolate')
        new_bounds[1] = new_bounds[1]+[int_func(new_wave)]
    
    return np.array(new_bounds)
    
def read_bounds(bound_files, dtype=float):
    """ Read the locations of slit boundaries from a file. """
    if type(bound_files) != list:
        bound_files = [bound_files]
    n_chips = len(bound_files)
    lower_bounds = [None]*n_chips
    upper_bounds = [None]*n_chips
    for i in range(n_chips):
        bounds = np.loadtxt(bound_files[i])#,dtype=dtype)
        lower_bounds[i] = bounds[:,:bounds.shape[1]//2].T
        upper_bounds[i] = bounds[:,bounds.shape[1]//2:].T
    if len(bound_files) == 1:
        lower_bounds = lower_bounds[0]
        upper_bounds = upper_bounds[0]
    return lower_bounds, upper_bounds

def count_word(n):
    """ Make counting words for outputs. """
    if n%10 == 1:
        suffix = 'st'
    elif n%10 == 2:
        suffix = 'nd'
    elif n%10 == 3:
        suffix = 'rd'
    else:
        suffix = 'th'
    return str(n)+suffix
    
def bias(bias_files, nx, ny, datareg, scanreg):
    """ Check if master biases already exist. If not, make them.
    
    Parameters:
    -----------
    bias_files : list;
        List of files to be included in the master bias.
        
    nx, ny : ints;
        Size of the chip.
        
    datareg : list;
        Borders in x and y pixels of the part of the chip that contains information.
    
    scanreg : list;
        Boundaries of the over- or under-scan regions.
    """
    n_chips = len(bias_files)
    master_biases = [None]*n_chips
    for i in range(n_chips):
        master_file = impath+'masterbias{0}_py.fits'.format(i+1)
        if not Path(master_file).is_file():
            print('Creating {0} master bias...'.format(count_word(i+1)))
            master_biases[i] = make_mastercube(bias_files[i], nx, ny, datareg, scanreg[i])
            fits.writeto(master_file,master_biases[i])
        else:
            master_biases[i] = fits.getdata(master_file)
    return master_biases
    
def flat(flat_files, nx, ny, datareg, master_biases, slic, swid, scanreg):
    """
    Check if master flats already exit. If not, make them.
    
    Parameters:
    -----------
    nx, ny : ints;
        Size of the chip.
        
    datareg : list;
        Borders in x and y pixels of the part of the chip that contains information.
        
    master_biases : list;
        List of master bias files.
        
    slic, swid : ints;
        Centers and widths of the slits on the chips.
        
    scanreg : list;
        Boundaries of the over- or under-scan regions.
    """
    n_chips = len(flat_files)
    master_flats = [None]*n_chips
    lower,upper = [None]*n_chips,[None]*n_chips
    for i in range(n_chips):
        master_file = impath+'masterflat{0}_py.fits'.format(i+1)
        if not Path(master_file).is_file():
            print('Creating {0} master flat...'.format(count_word(i+1)))
            normNan10, lower[i], upper[i] = make_masterflat(flat_files[i], nx, ny, datareg, master_biases[i], slic[i], swid[i], scanreg[i])
            master_flats[i] = normNan10
            fits.writeto(master_file, normNan10)
            np.savetxt(impath+'bounds{0}.txt'.format(i+1),np.append(lower[i],upper[i],axis=0).T)
        else:
            master_flats[i] = fits.getdata(master_file)
            lower[i], upper[i] = read_bounds(impath+'bounds{0}.txt'.format(i+1))
    return master_flats, lower, upper

def arc(arc_files, master_biases, nx, ny, slic, datareg, lower, upper, impath, linecat, gris_table, scanreg):
    """
    Check if the arc lamp files have already been reduced. If not, make them.
    
    Parameters:
    -----------
    arc_files : list;
        List of files to be used for making the master arc file.
        
    master_biases : list;
        Master bias files for both chips.
        
    nx, ny : ints;
        Image size.
        
    slic : list;
        List of centers for each column of each slit.
        
    datareg : list;
        Boudaries of the area of the chip that contains the information.
        
    lower, upper : lists;
        Lists of the slit boundaries.
        
    impath : str;
        Location of the images.
        
    linecat : str;
        FORS2 calibration file containing the line catalogue.
        
    gris_table : str;
        FORS2 calibration file containg the grism information table.
        
    scanreg : list;
        Over- or under-scan region boundaries on the chip.
    """
    n_chips = len(arc_files)
    master_arcs = [None]*n_chips
    lamda_solutions = [None]*n_chips
    for i in range(n_chips):
        master_file = impath+'masterarc{0}_py.fits'.format(i+1)
        #print('MASTER FILE:', master_file)
        if not Path(master_file[:-4]+'arccor.fits').is_file():
            print('Creating {0} master arc image...'.format(count_word(i+1)))
            lamda_solutions[i], master_arcs[i] = make_masterarc(arc_files[i], master_biases[i], nx, ny, slic[i], datareg, lower[i], upper[i], impath, linecat, gris_table, master_file, scanreg[i])
            
            fits.writeto(master_file[:-4]+'arccor.fits', master_arcs[i], overwrite=True)
            fits.writeto(impath+'lamcal{0}_py.fits'.format(i+1), lamda_solutions[i], overwrite=True)
        else:
            master_arcs[i] = fits.getdata(master_file[:-4]+'arccor.fits'.format(i+1))
            lamda_solutions[i] = fits.getdata(impath+'lamcal{0}_py.fits'.format(i+1))
    return master_arcs, lamda_solutions
    
def calc_lc(stars, wave_range=None, targ_ind=0):
    """
    Create a light curve for each extraction aperture of each star and remove the reference ligth curves from the target light curves extracted at the same aperture.
    
    Parameters:
    -----------
    stars : list;
        List of Source class instances corresponding to the stars in the data.
        
    wave_range : list of length 2, optional;
        If specified only create a light curve over the specified wavelength range.
        If not, create a light curve over the full available range.
        Default is None.
        
    targ_ind : int, optional;
        Index of the target star in the 'stars' list. Other stars are used as reference stars.
    """
    if wave_range:
        inds = np.where((wave>=wave_range[0])&(wave<=wave_range[1]))[0]
    else:
        inds = np.arange(len(wave),dtype=int)
    
    ref_inds = np.where(np.arange(len(stars)) != targ_ind)[0]
    
    target = stars[targ_ind]
    target_flux = np.nansum(target.fluxes[:,inds],axis=1)
    target_error = np.sqrt(target_flux)
    
    ref_ron = np.mean([stars[i].ron for i in ref_inds])
    ref_flux = np.sum([ np.nansum(stars[i].fluxes[:,inds],axis=1) for i in ref_inds], axis=0)
    ref_bg = np.nansum([ stars[i].backgrounds for i in ref_inds] ,axis=0)
    ref_error = np.sqrt(ref_flux)
    
    light_curve = target_flux/ref_flux / np.nanmean(target_flux/ref_flux)
    light_curve_err = light_curve * np.sqrt( (target_error/target_flux)**2 + (ref_error/ref_flux)**2 )
    
    return light_curve.T, light_curve_err.T
    
def mad_lc(data):
    """ Calculate the Median Absolute Deviation (MAD) of a light curve. """
    return 1e6 * np.nanmedian(np.abs(np.ediff1d(data)))

def my_medfilt(data,window,sig):
    """
    Filter a light curve to identify outliers using a running median and standard deviation.
    
    Parameters:
    -----------
    data : ndarray;
        The light curve.
        
    window : int;
        Width of the window around a given point over which the smoothed median and standard deviation are calculated.
        
    sig : int;
        Threshold in standard deviations.
        If the difference between the point and the smoothed light curve is over the threshold it is removed.
    """
    filtered = np.zeros(len(data))
    std_arr = np.zeros(len(data))
    w = window//2
    # Create a smoothed light curve with a running median, including dealing with borders.
    for i in range(len(data)):
        if i <= w:
            segment = data[:i+w+1]
        elif i>= len(data)-w:
            segment = data[i-w:]
        else:
            segment = data[i-w:i+w+1]
        filtered[i] = np.nanmedian(segment)
    diff = data-filtered
    w+=20
    # Calculate running standard deviations, including dealing with borders.
    for i in range(len(data)):
        if i <= w:
            segment = diff[:i+w+1]
        elif i>= len(data)-w:
            segment = diff[i-w:]
        else:
            segment = diff[i-w:i+w+1]
        std_arr[i] = robust.std(segment[np.isfinite(segment)])
    # Identify outliers.
    inds = np.where(np.abs(diff)<sig*robust.std(diff))[0] #
    return inds, filtered, std_arr

def sigma_clip(data,w_filt,sigma):
    """ Sigma clip a light curve using a running median and standard deviation. """
    data_copy = data*1.*np.nan
    good_inds, f, s = my_medfilt(data,w_filt,sigma)
    sub_inds, f, s = my_medfilt(data[good_inds],w_filt,sigma)
    data_copy[good_inds[sub_inds]] = data[good_inds[sub_inds]]
    return data_copy
    
def make_bins(sources, edges=None, bin_size=100, line=None, offset=0, centre=None, return_inds=False):
    """
    Create wavelength and flux bins to make light curves.
    Returns an array of the wavelengths and fluxes corresponding to each bin.
    
    Parameters:
    -----------
    sources : list;
        List of source instances for which the bins should be determined.
        
    edges : ndarray, optional;
        List of borders of regions of the spectra that are to be included.
        This way telluric or contaminated areas of the spectrum can be excluded.
        Default is None, which includes the entire spectrum.
        
    bin_size : float, optional;
        Bin size in Angstrom. Default is 100.
        
    line : any, optional;
        Lines around which bins should be made. Currently supports 'Na' and 'K'.
        Alternatively a list of wavelengths can be input. Default is None.
        
    offset : float, optional;
        If the first bin should not start at the first pixel an offset can be aplied. Default is 0.
        
    centre : float, optional;
        If given, specifies the centre of the first wavelength bin. Default is None.
        
    return_inds : bool, optional;
        If true, the function also returns the indices of corresponding to the different bins.
        Default is False.
    """
    wave_bins, flux_bins = [],[]
    wave = source.wavelengths
    data = source.fluxes
    n_pix = int(np.round(bin_size/(wave[1]-wave[0]))) # number of pixels in a bin_sizeÅ bin
    h_pix = int(np.round(bin_size/2/(wave[1]-wave[0]))) # number of pixels in half a bin size bin_sizeÅ
    o_pix = int(np.round(offset/(wave[1]-wave[0]))) # number of pixels in the offsetÅ distance
    
    #For bins surrounding different lines
    if line:
        all_inds = []
        if line=='Na':
            lines = np.array([5889.950,5895.924,8194.824])
        elif line=='K':
            lines = np.array([4044.14, 4047.21, 7664.8991, 7698.9645])
        line_list = np.array([np.argmin(np.abs(wave-l)) for l in lines if l>wave[0] and l<wave[-1]])

        edges = np.empty([0,2],dtype=int)
        for l in line_list:
            edges = np.append(edges,[[int(l-h_pix),int(h_pix+l)]],axis=0)
            all_inds = all_inds + [np.arange(int(l-h_pix),int(h_pix+l))]
    
    #When not using the full spectrum (i.e. excluded telluric areas or for certain lines)
    if edges:
        for e in edges:
            segment = data[:,:,:,e[0]:e[1]]
            wave_seg = wave[e[0]:e[1]]
            #Add wave and flux bins for complete bins
            for i in np.arange((e[1]-e[0])//n_pix):
                print(i)
                wave_bins = wave_bins + [wave_seg[i*n_pix:(i+1)*n_pix]]
                flux_bins = flux_bins + [segment[:,:,:,i*n_pix:(i+1)*n_pix]]
            #Add last incomplete bin ((e[1]-e[0])//n_pix = 0)
            if (e[1]-e[0])%n_pix != 1:
                print('n',n_pix)
                wave_bins = wave_bins + [wave_seg[-((e[1]-e[0])%n_pix):]]
                flux_bins = flux_bins + [segment[:,:,:,-((e[1]-e[0])%n_pix):]]
    
    #When using the full spectrum
    else:
        start = edges[0,0]+o_pix
        if centre:
            first_center = centre
            start = np.max([0,np.argmin(np.abs(wave-centre))-h_pix])
        else:
            first_center = wave[start+o_pix+h_pix]
        
        all_centers = np.arange(first_center, wave[edges[0,1]], bin_size)
        all_inds = []
        for i in range(len(all_centers)):
            inds = np.where((wave>all_centers[i]-bin_size/2)&(wave<=all_centers[i]+bin_size/2))[0]
            all_inds = all_inds + [inds]
            if np.sum(np.isfinite(data[0,0,:,inds]))>0:
                print(i)
                wave_bins = wave_bins+[wave[inds]]
                flux_bins = flux_bins+[data[:,:,:,inds]]
        
        # Fix for if the spectra have unequal lengths at the ends.
        if np.any(np.isnan(flux_bins[-1][:,0])):
            nan_edge = np.min(np.where(np.isnan(flux_bins[-1][:,0]))[-1])
            flux_bins[-1] = flux_bins[-1][:,:,:,:nan_edge]
            wave_bins[-1] = wave_bins[-1][:nan_edge]
        if np.any(np.isnan(flux_bins[0][:,0])):
            nan_edge = np.max(np.where(np.isnan(flux_bins[0][:,0]))[-1])
            flux_bins[0] = flux_bins[0][:,:,:,nan_edge:]
            wave_bins[0] = wave_bins[0][nan_edge:]
        
    if return_inds:
        return wave_bins, flux_bins, all_inds
    else:
        return wave_bins, flux_bins

def sort_files(data_path, out_path=None):
    """
    Sort raw data files into lists of science, bias, flat, arc and slit images.
    
    Parameters:
    -----------
    data_path : str;
        Location of the data files.
    
    out_path : str, optional;
        Location where the lists are written to. If not provided, defaults ot a 'file_lists' folder in the data_path.
    """
    if not out_path:
        out_path = f'{data_path}/file_lists'
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    
    fits_list = np.array(sorted(glob.glob(f'{data_path}/[!M.]*.fits')))
    
    data_list = []
    for f in fits_list:
        data_list += [DataFile(f)]
    
    chips = set([data.chip_id for data in data_list])
    dates = set([data.date for data in data_list])
    for c in chips:
        for d in dates:
            spec_files = [data for data in data_list if data.chip_id==c and data.date==d]
            for type in ['science','bias','flat','arc','slit']:
                file_names = [data.name for data in spec_files if data.type==type]
                if len(file_names) != 0:
                    np.savetxt(f'{out_path}/{type}_images_{d}_{c}.txt', file_names, fmt='%s')

#"""
# Information about the data, to be input by the user
target = 'WASP-69'
grism = '600z_23' #'600RI_19' #'600z_23' #'600B_22' #
band_name = grism.split('_')[0][3:]
filter = find_filter(grism)#'GG435_81' #'OG590_32' #'free_00' #
n_chips = 2
mask = 'A'
date='08-19'


# File locations according to standard ESO pipeline installation
static_calibration_path = '/opt/local/share/esopipes/datastatic/fors-5.5.7/'
gris_table = static_calibration_path + 'FORS2_GRS_{0}_{1}.fits'.format(grism,filter)
linecat = static_calibration_path + 'FORS2_ACAT_{0}_{1}.fits'.format(grism,filter)

# Location of files containing lists of filenames of raw science/bias/flat/arc lamp frames
impath = '/Volumes/TOSHIBAEXT/'+target+'/IDL_cals/'+band_name+'/'
out_path = '/Volumes/TOSHIBAEXT/'+target+'/IDL_output/'+band_name+'/'

# Read the list of science/bias/flat/arc lamp files for both detectors
print('Reading files...')
sci_files = read_files(impath, 'science', band_name, mask, n_chips)
bias_files = read_files(impath, 'bias', band_name, mask, n_chips)
flat_files = read_files(impath, 'flat', band_name, mask, n_chips)
arc_files = read_files(impath, 'arc', band_name, mask, n_chips)

# Use the first science file to obtain information about the detector
nx, ny, gain, ron, scanreg, datareg, swid = get_detector_parameters(sci_files[0][0],sci_files[1][0], reference_y=996)
# Estimate the slit centers by finding the brightest row on each detector.
# This will have to be changed for detector images with multiple slits.
slic = [np.array([np.argmax(np.sum(fits.getdata(sci_files[i][0]),axis=1))]) for i in range(2)]
# Use the grism table to obtain wavelength information
wmin, wmax, wstep = get_grism_parameters(gris_table)
#"""

#"""
# Calculaate master bias, flat and arclamp images.
master_biases = bias(bias_files, nx, ny, datareg, scanreg)
master_flats, lower, upper = flat(flat_files, nx, ny, datareg, master_biases, slic, swid, scanreg)
master_arcs, lamda_solutions = arc(arc_files, master_biases, nx, ny, slic, datareg, lower, upper, impath, linecat, gris_table, scanreg)


#"""

#"""
n_files = len(sci_files[0]) # Number of integrations to be processed.
nx_final = int(np.ceil((wmax-wmin)/wstep)) # X-axis length of the combined images.
ny_final = [int(ny - get_border(scan, nx)) for scan in scanreg] # Y-axis length of the combined images.
#"""

for i in range(n_files):
    print('Creating 2D image {0}/{1} '.format(i+1,n_files)+sci_files[0][i][:-9].split('/')[-1])
    print(sci_files[0][i])
    new_data = np.zeros([np.sum(ny_final), nx_final])
    new_bounds = [np.empty([n_chips,nx_final]),np.empty([n_chips,nx_final])]
    header = fits.getheader(sci_files[0][i])
    for j in range(n_chips):
        header['ESO DET OUT{0} GAIN'.format(j+1)] = fits.getheader(sci_files[j][i])['ESO DET OUT1 GAIN']
        header['ESO DET OUT{0} RON'.format(j+1)] = fits.getheader(sci_files[j][i])['ESO DET OUT1 RON']
        
        # Overscan corrected, bias and flat corrected image
        image = make_mastercube([sci_files[j][i]], nx, ny, datareg, scanreg[j])
        image = (image - master_biases[j])/master_flats[j]
        
        # Remove cosmic ray artefacts
        clean_image, mask_image = lacosmic.lacosmic(image, contrast=1, cr_threshold=15, neighbor_threshold=5, effective_gain=gain[j], readnoise=ron[j], maxiter=4)
        
        # Wavelength calibration
        lam_data = adjust_2d(clean_image, lamda_solutions[j], gris_table)
        new_bounds[0][j], new_bounds[1][j] = fix_traces(lamda_solutions[j], lower[j], upper[j], wmin=wmin, wmax=wmax, wstep=wstep)
        new_data[int(np.sum(ny_final[:j])):np.sum(ny_final[:j+1])] = lam_data
        
        # Create intermediate outputs. Here only for the first set of images to conserve disk space.
        if i==0:
            write_steps(sci_files[j][i], out_path, image, clean_image, lam_data)
            np.savetxt(impath+'bounds{0}v2.txt'.format(j+1),np.column_stack([new_bounds[0][j].T, new_bounds[1][j].T]))#, fmt='%i'
    
    # Write combined and calibrated images.
    new_fname = out_path+'rcom/'+sci_files[0][i].split('/')[-1][:-4]+'rcom.fits'
    fits.writeto(new_fname, new_data, header, output_verify='ignore', overwrite=True)
#"""

#"""
# Place where combined files are located
rcom_path = out_path+'rcom/'
# Place to output final products.
out_path_no_filt = '/'.join(out_path.split('/')[:-1])+'/'

# Define aperture sizes over which to extract the data.
apertures = np.arange(25,75,5)
np.savetxt(impath+'apertures.txt',apertures)
n_apertures = len(apertures)

n_stars = 2 # Number of slits / sources in the combined image.
combined_files = sorted(glob.glob(rcom_path+'*'+date+'*rcom.fits')) # Read the combined files.
sky_region = np.array([13,33,247,267]) # Define the region within each slit to be used for sky background.

# read slit boundaries from save file.
bound_files = sorted(glob.glob(impath+'bounds*v2.txt'))
lower, upper = read_bounds(bound_files, dtype=int)
upper[1], lower[1] = np.array([upper[1],lower[1]])+np.sum(ny_final)//2 # Adjusted for being in the combined image.

start_time = time.time() # Track duration of the source extraction/cleaning/alignment process.

star_list = [] # List of extracted sources.
for i in range(n_stars):
    # Create a source instance and extract the flux from the combined image files, then save.
    star_n = Source(combined_files, nx_final, n_apertures, i, i, gris_table)
    star_n.extract(combined_files, upper, lower, sky_region, apertures)
    pickle.dump(star_n, file=open(f"{out_path}/extracted_{i}.pkl",'wb'))
    
    # Clean the flux and pick an order for the polynomial for the spectral realignment.
    # The cleaning threshold and the order are both dataset specific and require some fiddling to find the best solution.
    if band_name=='RI':
        star_n.clean_flux()
        align_order = 1
    elif band_name=='z':
        star_n.clean_flux(sig_thr=11)
        align_order = 1
    elif band_name=='B':
        star_n.clean_flux(sig_thr=4)
        align_order = 2
    pickle.dump(star_n, file=open(f"{out_path}/cleaned_{i}.pkl",'wb'))
    
    # Wavelength alignment and adding the source to the list of sources.
    star_n.align_specs('/'.join(impath.split('/')[:-2])+'/', grism, align_order, print_stuff=False)
    star_list = star_list + [star_n]
    
    time_diff = int(np.around(time.time() - start_time))
    print('Processing {0} star(s) took {1} hours {2} minutes and {3} seconds.'.format(i+1,time_diff//3600, time_diff%3600//60, time_diff%60))

# Write extracted sources to disk.
pickle.dump(star_list, file=open(out_path_no_filt+'stars_'+band_name+'.pkl','wb'))
#"""

# Specify where light curves get stored. Create a new folder if necessary.
lc_path = f'/Volumes/TOSHIBAEXT/Wasp-69/IDL_output/{band_name}/lcs/'
if not os.path.exists(lc_path):
    os.mkdir(lc_path)
    
# Identify the brightest star in the list as the target.
target_index = np.argmax([np.nanmedian(s.max_values) for s in star_list])

# Create a white light curve with errors.
wl_lcs, wl_lc_errs = calc_alt_lc(star_list, wave_range=None, targ_ind=target_index)
best_arg = np.argmin(np.array([mad_lc(w) for w in wl_lcs])) # Identify best aperture
white_curve = sigma_clip(wl_lcs[best_arg], 41, 5)
white_curve_errors = wl_lc_errs[best_arg]

lc_times = stars[target_index].bjd
nonans = np.isfinite(white_curve)
np.savetxt(lc_path+'white_light_curve.dat', np.column_stack([lc_times[nonans], white_curve[nonans], white_curve_errors[nonans]]))

# Create the wavelength bins and make light curves.
wave_bins, flux_bins, bin_inds = make_bins(star_list, bin_size=100, centre=3493.)
n_bins = len(wave_bins)
w_cens = np.array([np.mean(w) for w in wave_bins])

binned_curves = []
binned_errors = []
mads = np.zeros([n_bins, n_apertures])
for i in range(n_bins):
    # Create light curves and corresponding errors for each aperture.
    curve, error = calc_lc(star_list, wave_range=[wave_bins[i][0],wave_bins[i][-1]],targ_ind=target_index)
    binned_curves = binned_curves + [curve[best_arg]]
    binned_errors = binned_errors + [error[best_arg]]
    
    n_str = '000'[:-len(str(i+1))]+str(i+1) # bin number
    np.savetxt(f'{lc_path}/bin{n_str}.dat', np.column_stack([lc_times, curve[best_arg], error[best_arg]]))
