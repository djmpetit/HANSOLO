import sys
import os
import glob
import pickle
import time
import gc

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
import lacosmic
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

class Image:
    """
    An image from the dataset and its properties.
    
    Methods:
    --------
    __init__ - create an instance
    is_bias - check if it's a bias frame
    get_border - get the width of the border
    get_detector parameters - get information on the detector
    get_slit_width - calculate slit widths from header
    get_slit_params - get slit information
    update_slits - update slit information
    copy_slits - copy slit information from another image
    find_bg - find the average value outside the slits
    get_slit_borders - get the borders of the slits
    make_slit_map - create a boolean map of where the slits are
    set_dark - set the background outside the slits to a fixed value
    adjust_image - do over/underscan correction and remove the border
    copy - copy the image to another instance
    get_wavelengths - obtain the nominal wavelengths from a reference
    click_plot - clickable plot for wavelength calibration
    new_solution - update the existing wavelength solution
    fit_line - fit the position of a line
    fitted_wavelengths - get parameters of accurately identified lines
    plot_window - plot a part of the spectrum
    polyfit_lines - fit a polynomial to a reference line
    interp_rows - interpolate over data to fill missing row
    find_wavelength_solution - obtain a wavelength solution for the image
    adjust_2d - interpolate the image to the nominal wavelengths
    combine_data - combine images
    
    Attributes:
    -----------
    name : str;
        Name of the file the image was read from.
        
    data, header : 2D ndarray and fits header
        Information from the fits file the image was read from.
        
    grism : str;
        Filter in which data was taken.
        
    image_type : str;
        Type of image. Bias, flat, arc or science.
    
    chip, chip_ind : str, int;
        Chip identifiers.
        
    n_x, n_y : int;
        Image size.
        
    gain, read_noise : float;
        Gain and read noise of the detector.
        
    scan_region, data_region : list;
        Lists of corners of the over/underscan region and the data containing region.
        
    border : int;
        Width of the border for over/underscan regions.
        
    n_slits : int;
        Number of slits in the image.
        
    slit_centers, slit_width : ndarray, float;
        Slit parameters in units of pixels. Slit width is assumed to be the same for all slits in an image.
        
    bg_value : float;
        Average background outside the slits.
        
    adjusted : bool;
        Weather the image has been corrected for over/underscan or not.
        
    slit_map : 2D ndarray;
        Boolean array the size of the data, indicating wether a pixel is located in a slit or not.
    
    lower_bounds, upper_bounds : list;
        Lists of arrays with the lower and upper bounds of the slits in the image.
        
    wavelength_solution : 2D ndarray;
        Wavelengths corresponding to the pixels in the image.
    
    wavelengths : ndarray;
        Nominal wavelengths.
    """
    
    def __init__(self, fname=None, slit_centers=None):
        """
        Initialise an image, setting up data and related attributes.
        
        Parameters:
        -----------
            fname : str, optional;
                The file from which the data is read. Default is None.
                
            slit_centers : list, optional;
                The centers of slits on the image. Default is None.
        """
        if fname:
            self.name = fname.split('/')[-1]
            self.data = fits.getdata(fname)
            self.header = fits.getheader(fname)
            self.get_detector_parameters()
            if not slit_centers and not self.is_bias():
                self.get_slit_params()
            if not self.is_bias():
                self.find_bg()
    
    def is_bias(self):
        """ Checks if the image is a bias image. """
        if 'bias' in self.name or 'BIAS' in self.image_type:
            bool = True
        else:
            bool = False
        return bool
    
    def get_border(self):
        """ Retrieves the width of the unilluminated border around the data. """
        if self.scan_region:
            self.border = (self.scan_region[1] - self.scan_region[0]) *2
        elif self.n_x > 3000:
            self.border = 20
        else:
            self.border = 10
    
    def get_detector_parameters(self):
        """ Obtain detector related parameters."""
        if 'ESO INS GRIS1 NAME' in self.header:
            self.grism = self.header['ESO INS GRIS1 NAME']
        if 'master' in self.name:
            self.image_type = ' '.join(self.name.split('/')[-1].split('_')[:2])
        else:
            self.image_type = self.header['ESO DPR TYPE']
        self.chip = self.header['EXTNAME']
        self.chip_ind = int(self.chip[-1])
        
        self.n_x = self.header['NAXIS1']
        self.n_y = self.header['NAXIS2']
        self.gain = self.header['ESO DET OUT1 GAIN']
        self.read_noise = self.header['ESO DET OUT1 RON']
    
        prescanx = self.header['ESO DET OUT1 PRSCX']
        prescany = self.header['ESO DET OUT1 PRSCY']
        overscanx = self.header['ESO DET OUT1 OVSCX']
        overscany = self.header['ESO DET OUT1 OVSCY']
        overscan_region = [self.n_y-overscany, self.n_y, overscanx, self.n_x-overscanx]
        prescan_region = [0, prescany, prescanx, self.n_x-prescanx]
        if self.chip == 'CHIP1':
            self.scan_region = prescan_region
        else:
            self.scan_region = overscan_region
        self.data_region = [overscany, self.n_y-overscany, 0, self.n_x]
        self.get_border()
        
    def get_slit_width(self):
        """ Get the width of the slits in the image. """
        pixel_scale = self.header['ESO INS PIXSCALE']
        for i in range(1,9):
            width_as = self.header[f'ESO INS MOS10{i} LEN']
            if width_as != 0.50:
                width_px = width_as/pixel_scale
        self.slit_width = width_px
    
    def get_slit_params(self):
        """ Get the y coordinates of the centers, the width and the number of slits present in an image. """
        y_intensity = np.sum(self.data, axis=1).astype(np.float32)
        med_val = np.median(y_intensity[20:-20])
        std_val = np.std(y_intensity[20:-20])
        y_intensity[:20] = med_val
        y_intensity[-20:] = med_val
        
        centers = []
        sig = 7
        threshold = med_val + sig*std_val
        bright_rows = np.where(y_intensity > threshold)[0]
        while len(bright_rows) > 0:
            prop_cent = np.nanargmax(y_intensity)
            if y_intensity[prop_cent-1] > threshold and y_intensity[prop_cent+1] > threshold:
                centers += [prop_cent]
                y_intensity[centers[-1]-150:centers[-1]+151] = med_val
            else:
                y_intensity[prop_cent] = med_val
                
            med_val = np.nanmedian(y_intensity)
            std_val = np.nanstd(y_intensity)
            threshold = med_val + sig*std_val
            bright_rows = np.where(y_intensity > med_val+sig*std_val)[0]
            
        self.n_slits = len(centers)
        self.slit_centers = centers
        self.get_slit_width()
        
    def update_slits(self, slit_centers, slit_width, replace=True):
        """ Replace or add to slit parameters. """
        if replace:
            self.slit_centers = slit_centers
            self.slit_width = slit_width
        else:
            self.slit_centers = self.slit_centers + slit_centers
            self.slit_width = np.median([self.slit_width, slit_width])
        self.n_slits = len(self.slit_centers)
        
    def copy_slits(self, source_image):
        """ Copy slit parameters from another image. """
        self.slit_map = source_image.slit_map
        self.update_slits(source_image.slit_centers, source_image.slit_width)
        if hasattr(source_image, 'lower_bounds'):
            self.lower_bounds = source_image.lower_bounds
            self.upper_bounds = source_image.upper_bounds
            #self.slit_bounds = source_image.slit_bounds
        
    def find_bg(self):
        """ Calculate average background outside of slits. """
        y_inds = np.ones(self.n_y)
        y_inds[:self.border] = 0
        y_inds[-self.border:] = 0
        for i in range(self.n_slits):
            lower_bound = self.slit_centers[i] - self.slit_width/2 - 30
            upper_bound = self.slit_centers[i] + self.slit_width/2 + 30
            slit_inds = np.arange(lower_bound, upper_bound, dtype=int)
            y_inds[slit_inds] = 0
            
        data_copy = np.copy(self.data)
        data_copy[y_inds==0] = 0
        self.bg_value = np.median(self.data[np.where(y_inds!=0)])
        
    def get_slit_borders(self, plot=False):
        """ Get the borders of the slit traces in the image. """
        box_pars = np.zeros([self.n_x, self.n_slits, 3])
        for i in range(self.n_slits):
            if plot:
                plt.ion()
            for j in range(self.n_x):
                col = self.data[:,j]
                # To isolate a single slit
                low = int(self.slit_centers[i] - self.slit_width/2) - 50
                if low > 0:
                    col[:low] = self.bg_value
                high = int(self.slit_centers[i] + self.slit_width/2) + 50
                if high < self.n_y:
                    col[high:] = self.bg_value
                box_pars[j, i] = fit_tophat(col, self.slit_centers[i], self.slit_width, plot=plot, title=f'Column {j}')
            if plot:
                plt.close()
                plt.ioff()
            
            box_pars[:,:,2][box_pars[:,:,2] < 1] = np.nan
            x_inds = np.arange(self.n_x)
            for j in range(3):
                clipped = stats.sigma_clip(box_pars[:,i,j], sigma=5, masked=True)
                box_pars[:,i,j] = clipped.filled(np.nanmedian(box_pars[:,i,j]))
                spl_func = scipy.interpolate.UnivariateSpline(x_inds, box_pars[:,i,j])
                if plot:
                    plt.scatter(x_inds, box_pars[:,i,j], marker='.')
                    plt.plot(x_inds, spl_func(x_inds), color='r')
                    plt.title(['lower bound', 'upper bound', 'amplitude'][j])
                    plt.show(block=False)
                    plt.pause(3)
                    plt.close()
                    plt.clf()
                box_pars[:,i,j] = spl_func(x_inds)
                
        self.lower_bounds = box_pars[:,:,0]
        self.upper_bounds = box_pars[:,:,1]
        self.slit_bounds = [[box_pars[:,s,0], box_pars[:,s,1]] for s in range(self.n_slits)]
    
    def make_slit_map(self):
        """ Make a boolean map of which pixels are in the slit. """
        y_inds = np.arange(self.n_y)
        self.slit_map = np.zeros(self.data.shape)
        for i in range(self.n_x):
            for j in range(self.n_slits):
                self.slit_map[(y_inds>self.lower_bounds[i,j]) & (y_inds<self.upper_bounds[i,j]), i] = 1

    def set_dark(self, dark=0):
        """ Set pixels outside the slits to the value of 'dark'. """
        with0 = np.copy(self.data)
        with0[self.slit_map == 0] = dark
        self.data = with0
        
    def adjust_image(self):
        """ Correct for borders and over/underscan regions. """
        if not hasattr(self, 'adjusted'):
            scan_region = self.data[self.scan_region[0]:self.scan_region[1], self.scan_region[2]:self.scan_region[3]]
            scan_mean = robust.mean(scan_region)
            data_only = self.data[self.data_region[0]:self.data_region[1], self.data_region[2]:self.data_region[3]]
            data_only = data_only - scan_mean
            self.data = data_only
            self.adjusted = True
            self.n_y -= self.border
        else:
            print('This image has already been adjusted.')
            
    def copy(self):
        """ Make a copy of the current instance. """
        copy_image = Image()
        key_list = vars(self).keys()
        for key in key_list:
            if type(vars(self)[key]) == np.ndarray:
                vars(copy_image)[key] = vars(self)[key].copy()
            else:
                vars(copy_image)[key] = vars(self)[key]
        return copy_image
    
    def get_wavelengths(self, grism_file):
        """ Get nominal wavelengths from a reference file. """
        wmin, wmax, wstep = get_grism_parameters(grism_file)
        self.wavelengths = np.arange(wmin, wmax+wstep/2, wstep)
    
    def click_plot(self, index, click_bool):
        """
        Show wavelength calibration plot and either identifies calibration lines by taking the closest line over the threshold for automatic calibration or prompts the user to click the indicated line for manual wavelength calibration.
        
        Parameters:
        -----------
        index : int;
            Index of the line to be selected.
            
        click_bool : bool, optional;
            If True, manual calibration is done. Else automatic calibration. Default is False.
        """
        #waves, rough_spec, line_waves, click_lines[j], threshold=threshold, click_bool=manual_mode
        if click_bool:
            print('Click the line at',self.click_lines[index],'Å, or left of the data if line not shown')
            fig = plt.figure()
            fig.set_size_inches(10,5)
            ax = fig.add_subplot(111)
            ax.plot(self.wavelengths, self.spectrum, color='k')
            ax.scatter(self.line_waves, np.zeros(len(self.line_waves)), marker='x', color='r')
            ax.axvline(self.click_lines[index], zorder=-10)
            cid = fig.canvas.mpl_connect('button_press_event', onclick)
            plt.show()
        else:
            line_inds = np.where(self.spectrum > self.threshold)[0]
            oldxval = np.argmin( np.abs(self.wavelengths - self.click_lines[index]) )
            newxval = np.argmin( np.abs(line_inds - oldxval) )
            self.line_xvals.append( self.wavelengths[line_inds[newxval]] )
    
    def new_solution(self, current_x, new_x, previous_solution, inds=None, plot=False):
        """ Update the wavelength solution. """
        if inds is not None:
            if type(current_x)==list or type(new_x)==list:
                current_x = np.array(current_x)
                new_x = np.array(new_x)
            cs = robust_fit(current_x[inds], new_x[inds], 4)
            if plot:
                plt.scatter(current_x[inds], new_x[inds], color='k')
        else:
            cs = robust_fit(current_x, new_x)
            if plot:
                plt.scatter(current_x, new_x, color='k')
        
        solution = np.polyval(cs, previous_solution)
        if plot:
            highres_x = np.linspace(current_x[0], current_x[-1], 1000)
            plt.plot(highres_x, np.polyval(cs, highres_x),color='r')
            plt.show(block=False)
            plt.pause(2)
            plt.close()
            plt.clf()
        
            plt.plot(solution, self.spectrum, color='k')
            [plt.axvline(l, color='green', zorder=-1) for l in new_x]
            plt.show(block=False)
            plt.pause(5)
            plt.close()
            plt.clf()
            
        return solution
            
        #xval_array[i], click_lines, waves, rough_spec, inds=good_inds, plot=True
    
    def fit_line(self, index, peak_width, manual_mode):
        """ Fit a line on the detector to obtain the center location. """
        self.click_plot(index, click_bool=manual_mode)
        wmin = self.wavelengths[0]
        wmax = self.wavelengths[-1]
        if self.line_xvals[index] < wmin or self.line_xvals[index] > wmax:
            self.line_xvals[index] = np.nan
            return 0
        else:
            x_inds = np.where((self.wavelengths > self.line_xvals[index] - peak_width) & (self.wavelengths < self.line_xvals[index] + peak_width))
            fit = fit_moffat(self.wavelengths[x_inds], self.spectrum[x_inds], plot=True)
            return fit.x_0.value
    
    def fitted_wavelengths(self, spectrum, window_width, manual, plot_bool=False, mode='valid'):
        """ Get the parameters of lines that have been accurately identified in the calibration image.
        
        Parameters:
        -----------
        spectrum : ndarray;
            Calibration spectrum fluxes.
            
        window_width : int;
            Width of region around the line where the fit is performed.
            
        manual : bool;
            If True, calibration is done manually. If False, it is doen automatically.
            
        plot_bool : bool, optional;
            If True, print progress statements.
            
        mode : str, optional;
            If 'valid', badly fit lines are ignored. If 'same' they are included as NaNs.
        """
        fitted_x = []
        fitted_l = []
        for i in range(len(self.line_waves)):
            ind_guess = np.argmin( np.abs(self.solution - self.line_waves[i]) )
            x_guess = self.solution[ind_guess]
            x_inds = np.where( (self.solution > x_guess-window_width) & (self.solution < x_guess+window_width) )
            fit = fit_moffat(self.solution[x_inds], spectrum[x_inds], plot=plot_bool)
            success = check_line_fit(fit, self.solution[x_inds[0][0]], self.solution[x_inds[0][-1]], 1e3, manual)
            if success:
                fitted_x = fitted_x + [fit.x_0.value]
                fitted_l = fitted_l + [self.line_waves[i]]
                if plot_bool:
                    print(f'Line at {str(int(np.round(self.line_waves[i])))} Å added.')
            elif mode=='same':
                fitted_x = fitted_x + [np.nan]
                fitted_l = fitted_l + [np.nan]
                if plot_bool:
                    print(f'Line at {str(int(np.round(self.line_waves[i])))} Å skipped.')
            elif plot_bool:
                print(f'Line at {str(int(np.round(self.line_waves[i])))} Å skipped.')
        return fitted_x, fitted_l
        
    #first_solution, cut_lines, rough_width, rough_spec, manual_mode, print_status=True
    
    def plot_window(self):
        """ Plot segments of the spectrum from the start to the end of the wavelength range. """
        maxval = 0
        i = 0
        plt.ion()
        while maxval < np.max(self.solution):
            minval = np.min(self.solution) + 50*i
            maxval = minval + 50
            plt.plot(self.solution, self.spectrum, color='k')
            [plt.axvline(l, color='green', zorder=-1) for l in self.line_waves]
            plt.xlim([minval, maxval])
            plt.pause(.1)
            plt.clf()
            i += 1
        plt.close()
        plt.ioff()
            
        #second_solution, rough_spec, lvals
    
    def polyfit_lines(self, all_lines, complete_rows):
        """ Fit a polynomial to the line traces on the detector.
        
        Parameters:
        -----------
        all_lines : list;
            X positions of the lines for each row.
        
        complete_rows : ndarray;
            Y indices of the rows which are completely in the slit.
        """
        n_lines = len(self.line_waves)
        yfit_lines = np.zeros(all_lines.shape)
        sigs = np.zeros(n_lines)
        
        plt.ion()
        for i in range(n_lines):
            inds = np.where(np.isfinite(all_lines[complete_rows, i]))
            if len(inds[0]) > len(all_lines[complete_rows, i])/2:
                cs = robust_fit(complete_rows[inds], all_lines[complete_rows,i][inds], 3)
                yfit_lines[complete_rows,i] = np.polyval(cs, complete_rows)
                diff = yfit_lines[complete_rows,i]-all_lines[complete_rows,i]
                sigs[i] = robust.std(diff[np.isfinite(diff)])
                #plt.scatter(all_lines[:,i],np.arange(self.n_y),color='k')
                plt.scatter(all_lines[complete_rows, i], complete_rows, color='k')
                plt.plot(yfit_lines[complete_rows,i], complete_rows, color='r')
                plt.title(f'Arc line {i} fit in the y direction')
                #plt.show(block=False)
                plt.pause(.1)
                #plt.close()
                plt.clf()
        plt.close()
        plt.ioff()
        
        clean_line_inds = np.where(sigs < 2*np.nanmedian(sigs))
        return clean_line_inds
        
    def interp_rows(self, bad_rows):
        """ If a row is not fit properly, interpolate from the closest rows above and below. """
        for i in range(len(bad_rows)):
            row_up = bad_rows[i]+1
            row_above = self.wavelength_solution[row_up]
            while np.sum(row_above) == 0:
                row_up += 1
                row_above = self.wavelength_solution[row_up]
            row_lo = bad_rows[i]-1
            row_below = self.wavelength_solution[row_lo]
            while np.sum(row_below) == 0:
                row_lo -= 1
                row_below = self.wavelength_solution[row_lo]
            self.wavelength_solution[bad_rows[i]] = np.mean([row_below, row_above], axis=0)
    
    
    def find_wavelength_solution(self, line_file, grism_file, manual_mode=False, custom_line_list=None):
        """
        Create a wavelength solution for the calibration file.
        A first solution is found based on a number of lines listed in the calibration file that can be identified either automatically or manually in the calibration spectrum.
        This solution is further improved by fitting all the lines in the spectrum and then iterated over.
        
        Parameters:
        -----------
        line_file : str;
            Calibration file that has a list of identified lines.
            
        grism_file : str;
            File containing the grism table.
            
        manual_mode : bool, optional;
            If True, calibrations are done manually. If False, they are done automatically. Default is False.
        
        custom_list : str, optional;
            File containing a custom list of identifiable lines, to be used instead of ones specified in this function. Default is None.
        """
        self.wavelength_solution = np.zeros(self.data.shape)
        wmin, wmax, wstep = get_grism_parameters(grism_file)
        self.wavelengths = wmin + np.arange(self.n_x) * wstep
        self.line_waves, line_ions, line_sets = get_line_parameters(line_file)
        # Lines used for first iteration of the wavelength calibration:
        self.click_lines, self.threshold = wave_id_lines(self.grism, custom_line_list)
        n_lines = len(self.click_lines)
        
        guess_range = 20
        peak_width = 30
        xval_array = np.zeros([self.n_slits, n_lines])
        for i in range(self.n_slits):
            # Extract the spectrum from the data
            spec_guess_area = self.data[int(self.slit_centers[i] - guess_range/2) : int(self.slit_centers[i] + guess_range/2)]
            self.spectrum = robust.mean(spec_guess_area, 5., axis=0)
            self.error = np.sqrt(self.spectrum / guess_range)
            plt.plot(self.wavelengths, self.spectrum, color='k')
            plt.axhline(self.threshold, color='green')
            [plt.axvline(c, color='green') for c in self.click_lines]
            plt.show(block=False)
            plt.pause(5)
            plt.close()
            plt.clf()
            
            # Obtain first iteration of the wavelength solution.
            self.line_xvals = []
            for j in range(n_lines):
                    xval_array[i,j] = self.fit_line(j, peak_width, manual_mode)
            good_inds = np.where( (xval_array[i]>wmin) & (xval_array[i]<wmax) )[0]
            self.solution = self.new_solution(xval_array[i], self.click_lines, self.wavelengths, inds=good_inds)
            # If the solution didn't work, try again, but manually.
            while( np.any(self.solution)<=0 or np.min(self.solution)<wmin-2000 ):
                print('Wavelength calibration has failed. Please select the following lines by hand:')
                for j in range(n_lines):
                    print(self.click_lines[j])
                    xval_array[i,j] = self.fit_line(j, peak_width, manual_mode=True)
                good_inds = np.where( (xval_array[i]>wmin) & (xval_array[i]<wmax) )[0]
                self.solution = self.new_solution(xval_array[i], self.click_lines, self.wavelengths, inds=good_inds)
            
            # Get the second iteration based on applying the first to all the available lines.
            line_inds = np.where((self.line_waves > self.solution[0]) & (self.line_waves < self.solution[-1]))
            self.line_waves = self.line_waves[line_inds]
            xvals, lvals = self.fitted_wavelengths(self.spectrum, guess_range, manual_mode, plot_bool=True)
            self.solution = self.new_solution(xvals, lvals, self.solution)
            self.plot_window()
            
            # Select the data in the area around the slit.
            lower_guess = int(self.slit_centers[i] - self.slit_width/2 - guess_range)
            upper_guess = int(self.slit_centers[i] + self.slit_width/2 + guess_range)
            # Identify rows wich only/partially contain slit data.
            row_counts = np.sum(self.slit_map[lower_guess:upper_guess], axis=1)
            complete_rows = np.where(row_counts == self.n_x)[0] + lower_guess
            partial_rows = np.where((row_counts!=0) & (row_counts!=self.n_x))[0] + lower_guess
            n_complete = len(complete_rows)
            n_partial = len(partial_rows)
            print('Complete rows:',n_complete)
            
            # Calculate the positions of the lines in each complete row.
            line_positions = np.zeros([n_complete, len(self.line_waves)])
            all_lines = np.zeros([self.n_y, len(self.line_waves)])
            for j in range(n_complete):
                row = self.data[complete_rows[j]]
                line_positions[j], new_lvals = self.fitted_wavelengths(row, guess_range, manual_mode, mode='same')
                print(f'{j+1}/{n_complete}',end='\r')
                if len(line_positions[j][np.isfinite(line_positions[j])] > len(line_positions[j]/2)):
                    valid_inds = np.where(np.isfinite(line_positions[j]))
                    valid_pos = line_positions[j][valid_inds]
                    valid_lvals = np.array(new_lvals)[valid_inds]
                    line_solution = self.new_solution(valid_pos, valid_lvals, self.solution)
                all_lines[complete_rows[j]] = line_positions[j]
                self.wavelength_solution[complete_rows[j]] = line_solution
            
            # Set out of slit and failed wavelength calibrations to NaNs
            self.wavelength_solution[self.wavelength_solution==0] = np.nan
            clean_inds = self.polyfit_lines(all_lines, complete_rows)
            
            # Obtain a final solution for each complete row.
            bad_rows = []
            for j in complete_rows:
                if np.sum(np.isfinite(all_lines[j])) > 5: #< len(all_lines[j])/2:
                    self.wavelength_solution[j] = self.new_solution(all_lines[j], self.line_waves, self.wavelength_solution[j])
                else:
                    self.wavelength_solution[j] = 0
                    bad_rows = bad_rows + [j]
            
            # Interpolate missed rows if necessary.
            if len(bad_rows) == 0:
                print('Finding a wavelength solution was succesful for all rows.')
            else:
                print(f'Finding a wavelength solution was unsuccesful for {len(bad_rows)} out of {n_complete} rows: {bad_rows} \n Interpolating a solution for the missing rows.')
                self.interp_rows(bad_rows)
            
            # Extend the solution to partial rows.
            for j in range(self.n_x):
                good_inds = np.where(self.wavelength_solution[complete_rows,j] != 0)
                cs = robust_fit(complete_rows[good_inds], self.wavelength_solution[complete_rows,j][good_inds], 3)
                self.wavelength_solution[partial_rows,j] = np.polyval(cs, partial_rows)
                
    def adjust_2d(self):
        """ Interpolate all the rows in the data to match the nominal wavelength solution. """
        new_data = np.zeros([self.n_y, len(self.wavelengths)])
        for i in range(self.n_y):
            good_inds = np.where(np.isfinite(self.wavelength_solution[i]) & np.isfinite(self.data[i]))
            data_nonan = self.data[i][good_inds]
            wave_solution_nonan = self.wavelength_solution[i][good_inds]
            if len(wave_solution_nonan) > self.n_x//2:
                wmin = self.wavelengths[0]
                wmax = np.round(self.wavelengths[-1])
                wstep = np.median(self.wavelengths - np.roll(self.wavelengths,1))
                new_data[i] = linear_rebin(wave_solution_nonan, data_nonan, wmin=wmin, wmax=wmax, wstep=wstep)
            else:
                new_data[i] = np.nan
        self.data = new_data
        self.n_x = len(self.wavelengths)
    
    def combine_data(self, second_image):
        """ Combine data from different images. """
        new_image = self.copy()
        new_image.data = np.append(self.data, second_image.data, axis=0)
        new_image.slit_map = np.append(self.slit_map, second_image.slit_map, axis=0)
        if hasattr(self, 'lower_bounds'):
            sec_lower = [lb + self.n_y for lb in second_image.lower_bounds]
            sec_upper = [ub + self.n_y for ub in second_image.upper_bounds]
            new_image.lower_bounds = np.append(self.lower_bounds, sec_lower, axis=1)
            new_image.upper_bounds = np.append(self.upper_bounds, sec_upper, axis=1)
        second_image.slit_centers = [sc+self.n_y for sc in second_image.slit_centers]
        self.update_slits(second_image.slit_centers, second_image.slit_width, replace=False)
        self.n_y, self.n_x = new_image.data.shape
        return new_image
        
class Cube:
    """
    A data cube made up of different images.
    
    Methods:
    --------
    __init__ - Create an instance
    add_frame - Add an image to the cube
    collapse_cube - combine the cube into a master image.
    
    Attributes:
    -----------
    nx, ny, nz : int;
        The shape of the cube.
        
    data : 3D ndarray;
        Data contained in the stacked images.
        
    collapsed:
        Master image made by sigma-clipped averaging over the z-axis of the cube.
    """
    
    def __init__(self, flist):
        """
        Create a cube by adding images from the input list.
        
        Parameters:
        -----------
        flist : list;
            List of input images.
        """
        for f in flist:
            self.add_frame(f)
        self.nx = self.data.shape[2]
        self.ny = self.data.shape[1]
        self.nz = self.data.shape[0]
        
    def add_frame(self, image):
        """ Correct an image for borders and over/underscan and add it to the cube. """
        image.adjust_image()
        frame = image.data
        if not hasattr(self, 'data'):
            self.data = np.array([frame])
        else:
            self.data = np.append(self.data, [frame], axis=0)
            
    def collapse_cube(self, n_excl=3):
        """ Sigma clipped average of the cube.
        
        Parameters:
        -----------
        n_excl : int;
            Number of excluded pixels at the top and bottom of the distribution along the z-axis.
        """
        self.data = np.sort(self.data, axis=0)
        if self.nz > 3*n_excl+1:
            image = np.mean(self.data[n_excl:-n_excl], axis=0)
        else:
            image = np.mean(self.data, axis=0)
        self.collapsed = image

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
    
    def __init__(self, images, n_appertures, index, chip, grism_info=None, dumb=False):
        """
        Initialise a Source instance, setting up arrays for fluxes, errors, and other attributes.

        Parameters:
        -----------
        images : list;
            A list of images from which to extract data.
            
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
        self.file_list = images
        self.number = index
        self.n_x = fits.getdata(images[0]).shape[1]
        self.fluxes = np.zeros([n_images, self.n_x, n_appertures])
        self.flux_errors = np.zeros([n_images, self.n_x, n_appertures])
        self.qis = np.zeros([n_images, self.n_x])
        self.fitcens = np.zeros([n_images, self.n_x])
        self.smocens = np.zeros([n_images, self.n_x])
        self.fwhms = np.zeros([n_images, self.n_x])
        self.backgrounds = np.zeros([n_images, self.n_x])
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
            self.wavelengths = self.wmin + self.wstep * np.arange(self.n_x)
    
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
                self.gain = header[f'ESO DET OUT1 GAIN'] # Currently only gets read noise and gain from the first chip, need to add it so I can get header info from the second chip.
                self.ron = header[f'ESO DET OUT1 RON']
    
    def copy(self, dumb=False):
        """ Create a copy of the Source instance """
        copy_star = Source([], 0, 0, self.number , dumb=dumb)
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
        err_ron = np.sqrt(2*aps[k]) * self.ron
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
        print(f'\nCleaning star {self.number}')
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
    
    def extract(self, slit_tops, slit_bottoms, skyreg, aps):
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
        for i in range(len(self.file_list)):
            print('Star {0}/{1}, file {2}/{3} \r'.format( self.number, len(slit_tops), i+1, len(self.file_list) ), end='')
            data, header = fits.getdata(self.file_list[i], header=True)
            y_inds = np.arange(np.median(slit_bottoms),np.median(slit_tops),dtype=int)
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
        model_wave, model_flux, model_step = get_model(cal_path, self.wmin, self.wmax, grism, print_res=print_res)
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
            cs = np.polyfit(cent, shifts[:,0]*self.wstep, n_order)#, w=1/shifts[:,1])
            new_wave = self.wavelengths + np.polyval(cs, self.wavelengths)
            if print_res:
                plt.errorbar(cent,shifts[:,0]*self.wstep,yerr=shifts[:,1]*self.wstep,fmt='.')
                plt.plot(self.wavelengths,new_wave-self.wavelengths)
                plt.plot(self.wavelengths, np.polyval(np.polyfit(cent,shifts[:,0]*self.wstep,1), self.wavelengths))
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
        print(f'Aligning star {self.number}')
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
    
def get_model(cal_path, wmin, wmax, grism, print_res=False):
    """ Read a model stellar spectrum from the calibration file and degrade it to data resolution. """
    model_files = sorted(glob.glob(f'{cal_path}/*PHOENIX*'))
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
    
def read_files(path, data_type, data_set, n_lists):
    """ Read list of file names. """
    file_lists = [None]*n_lists
    for i in range(n_lists):
        file_lists[i] = np.loadtxt(f'{path}/file_lists/{data_type}_images_{data_set}_CHIP{i+1}.txt',dtype=str)
        if file_lists[i].size == 1:
            file_lists[i] = np.array([file_lists[i]])
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
    
def make_master_flat(flist, master_bias, calib_path=None, slit_centers=None, slit_width=None):
    """
    Create a master flat from a list of input images.
    
    Parameters:
    -----------
    flist : list;
        List of flat images to be combined into the master flat.
        
    master_bias : Image;
        Master bias image to subtract from the flat.
    
    calib_path : str, optional;
        Path to save intermediate products if it is provided. Default is None.
        
    slit_centers : list, optional;
        List of y-indices of the slits on the image. Default is None.
    
    slit_width : float, optional;
        Width of the slits on the detector. Default is None.
    """
    master_flat = master_image(flist)
    master_flat.data = master_flat.data - master_bias.data
    master_flat.update_slits(slit_centers, slit_width)
    master_flat.get_slit_borders(plot=True)
    master_flat.make_slit_map()
    if calib_path:
        fits.writeto(f'{calib_path}/flat_slit_map_{master_flat.chip_ind}.fits', master_flat.slit_map, overwrite=True)
    master_flat.set_dark(dark = 0)
    if calib_path:
        fits.writeto(f'{calib_path}/flat0_{master_flat.chip_ind}.fits', master_flat.data, overwrite=True)
    
    # Normalise the flat
    flat_mean = robust.mean(master_flat.data[master_flat.slit_map == 1], 3)
    master_flat.data = master_flat.data / flat_mean
    if calib_path:
        fits.writeto(f'{calib_path}/flatnorm_{master_flat.chip_ind}.fits', master_flat.data, overwrite=True)
        
    # Remove low frequency variations by removing a smoothed version of the flat
    smoothed_data = scipy.ndimage.median_filter(master_flat.data, size=10, mode='reflect')
    if calib_path:
        fits.writeto(f'{calib_path}/flatsmooth_{master_flat.chip_ind}.fits', smoothed_data, overwrite=True)
    master_flat.data = master_flat.data / smoothed_data
    master_flat.set_dark(dark = 0)
    return master_flat

def onclick(event):
    """ Close plot after clicking. """
    global xvals
    xvals.append(event.xdata)
    fig.canvas.mpl_disconnect(cid)
    plt.close()
    return
        
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
    
def make_master_arc(arclist, master_bias, master_flat, linecat, grism_file, calib_path=None):
    """ Create a master calibration arc lamp file.
    
    Parameters:
    -----------
    arclist : list;
        List of arclamp files to be included.
    
    masterbias : Image;
        Bias image.
        
    master_flat : Image;
        Flat image.
        
    linecat : str;
        Calibration line catalogue file name.
        
    grismfile : str;
        File name for grism table.
        
    calib_path : str, optional;
        Folder to write intermediate product if given. Default is None.
    """
    master_arc_list = []
    for i in range(len(arclist)):
        master_arc = Image(arclist[i])
        master_arc.adjust_image()
        master_arc.data = master_arc.data - master_bias.data
        master_arc.copy_slits(master_flat)
        master_arc.set_dark(dark = np.nan)
        master_arc_list = master_arc_list + [master_arc]
    master_arc = master_image(master_arc_list)
    if calib_path:
        fits.writeto(f'{calib_path}/arc_combined.fits', master_arc.data, header=master_arc.header, output_verify='ignore', overwrite=True)
    
    master_arc.find_wavelength_solution(linecat, grism_file)
    master_arc.adjust_2d()
    return master_arc
    
def write_step(image, path, step):
    """ Write an image to a fits file. """
    new_file = f'{path}/{image.name[:-5]}'
    fits.writeto(f'{new_file}_{step}.fits', image.data, image.header, overwrite=True, output_verify='ignore')

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
    
def read_bounds(bound_file):
    """ Read the upper and lower boundaries of slits from the provided file. """
    cols = np.loadtxt(bound_file)
    n_slits = cols.shape[1]//2
    lower = [cols[:,i] for i in range(n_slits)]
    upper = [cols[:,i+n_slits] for i in range(n_slits)]
    return lower, upper

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
    
def master_image(flist):
    """ Create a master image of the provided files. """
    if type(flist[0]) == np.str_:
        frames = [Image(b) for b in flist]
    else:
        frames = flist
    cube = Cube(frames)
    cube.collapse_cube()
    master = frames[0].copy()
    master.image_type = 'master '+master.image_type
    master.data = cube.collapsed
    return master
    
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

def get_slit_info(flist):
    """ Get the y-indices of the slits and their width from an image. """
    slit_centers = []
    slit_widths = []
    for i in range(len(flist)):
        ims = [Image(f) for f in flist[i]]
        cents = np.array([i.slit_centers for i in ims])
        slit_centers = slit_centers + [list(np.mean(cents, axis=1, dtype=int)+[631,990][i])]
        slit_widths = slit_widths + [np.mean([i.slit_width for i in ims])]
    if len(slit_centers) == 1:
        slit_centers = [slit_centers]
    return slit_centers, slit_widths

def wave_id_lines(grism, custom_list):
    """ Return wavelengths of calibration lines to be used for first pass wavelength calibration based on the wavelength range of the observed data. Returns the wavelengths and an intensity threshold for automatic line identification.
    
    Parameters:
    -----------
    grism : str;
        Name of the grism in which the observations are done.
    
    custom_list : list, optional;
        List of custom lines to be used instead of the defaults if provided. Default is None.
    """
    if 'RI' in grism:
        click_lines = [5460.75, 5875.82, 6096.16, 6929.47, 7245.17, 7948.18, 8264.52] #These lines are valid when the real solution is shifted to the right of the theoretical values. If reducing other data in this filter, check that this is still valid!
        threshold = 3e3
    elif 'B' in grism:
        click_lines = [4046.56, 4678.1, 5085.8, 5460.75, 5875.62]
        threshold = 2500
    elif 'z' in grism_file:
        click_lines = [7438.9, 7635.11, 7724.21, 7948.18,  9224.50, 9657.78]
        threshold = 3e3
    elif custom_list:
        click_lines = np.loadtxt(custom_list)
        threshold = 3e3
    else:
        raise ValueError('GRISM NOT SUPPORTED: please add a list of identifiable lines.')
    return click_lines, threshold

# Information about the data, to be input by the user
target = 'WASP-69'
grism = '600RI_19'
band_name = grism.split('_')[0][3:]
filter = find_filter(grism)
n_chips = 2
date='2017-07-19'

# File locations according to standard ESO pipeline installation
static_calibration_path = '/opt/local/share/esopipes/datastatic/fors-5.5.7/'
gris_table = static_calibration_path + 'FORS2_GRS_{0}_{1}.fits'.format(grism,filter)
linecat = static_calibration_path + 'FORS2_ACAT_{0}_{1}.fits'.format(grism,filter)

# Location of files containing lists of filenames of raw science/bias/flat/arc lamp frames
impath = 'data'
calib_path = f'calib_files'
out_path = 'output_files'

# Read the list of science/bias/flat/arc lamp files for both detectors
print('Reading files...')
sort_files(impath)
sci_files = read_files(impath, 'science', date, n_chips)
bias_files = read_files(impath, 'bias', date, n_chips)
flat_files = read_files(impath, 'flat', date, n_chips)
arc_files = read_files(impath, 'arc', date, n_chips)
slit_files = read_files(impath, 'slit', date, n_chips)

# Iterative identification of slits based on the provided image.
slit_centers, slit_widths = get_slit_info(slit_files)

# Master biases
master_biases = [None]*n_chips
for i in range(n_chips):
    master_file = f'{calib_path}/master_bias_chip{i+1}.fits'
    if not Path(master_file).is_file():
        print(f'Creating {count_word(i+1)} master bias...')
        master_biases[i] = master_image(bias_files[i])
        fits.writeto(master_file, master_biases[i].data, header=master_biases[i].header, output_verify='ignore')
        master_biases[i].name = master_file
    else:
        master_biases[i] = Image(master_file)

# Master flats
master_flats = [None]*n_chips
for i in range(n_chips):
    master_file = f'{calib_path}/master_flat_chip{i+1}.fits'
    if not Path(master_file).is_file():
        print(f'Creating {count_word(i+1)} master flat.')
        master_flats[i] = make_master_flat(flat_files[i], master_biases[i], calib_path, slit_centers[i], slit_widths[i])
        fits.writeto(master_file, master_flats[i].data, header=master_flats[i].header, output_verify='ignore')
        fits.writeto(f'{calib_path}/slit_map_chip{i+1}.fits', master_flats[i].slit_map, overwrite=True)
        np.savetxt(f'{calib_path}/slit_bounds_chip{i+1}.dat', np.column_stack([master_flats[i].lower_bounds, master_flats[i].upper_bounds]))
    else:
        master_flats[i] = Image(master_file)
        master_flats[i].update_slits(slit_centers[i], slit_widths[i])
        master_flats[i].slit_map = fits.getdata(f'{calib_path}/slit_map_chip{i+1}.fits')
        master_flats[i].lower_bounds, master_flats[i].upper_bounds = read_bounds(f'{calib_path}/slit_bounds_chip{i+1}.dat')

# Master arcs
master_arcs = [None]*n_chips
for i in range(n_chips):
    master_file = f'{calib_path}/master_arc_chip{i+1}.fits'
    if not Path(master_file).is_file():
        print(f'Creating {count_word(i+1)} master arc image.')
        master_arcs[i] = make_master_arc(arc_files[i], master_biases[i], master_flats[i], linecat, gris_table) # To use non-default lines, check out the wave_id_lines function.
        fits.writeto(master_file, master_arcs[i].data, header=master_arcs[i].header, output_verify='ignore')
        fits.writeto(f'{calib_path}/master_lam_chip{i+1}.fits', master_arcs[i].wavelength_solution, overwrite=True)
    else:
        master_arcs[i] = Image(master_file)
        master_arcs[i].wavelength_solution = fits.getdata(f'{calib_path}/master_lam_chip{i+1}.fits')

n_files = len(sci_files[0]) # Number of integrations to be processed.
write_steps = True # Write intermediate outputs.
for i in range(n_files):
    print(f'Creating 2D image {i+1}/{n_files}: {sci_files[0][i][:-9].split("/")[-1]}')
    
    sci_ims = [] # List of science images for each chip.
    for j in range(n_chips):
        sci_image = Image(sci_files[j][i]) # Read image
        sci_image.adjust_image() # Overscan correction
        sci_image.copy_slits(master_flats[j]) # Identify slits
        sci_image.data = (sci_image.data - master_biases[j].data) / master_flats[j].data # Flat and bias correction
        if write_steps:
            write_step(sci_image, out_path, 'red')
        
        # Remove cosmic ray artefacts. These parameters need to be tuned to the data.
        sci_image.data, mask_image = lacosmic.lacosmic(sci_image.data, contrast=1, cr_threshold=15, neighbor_threshold=5, effective_gain=sci_image.gain, readnoise=sci_image.read_noise, maxiter=4)
        if write_steps:
            write_step(sci_image, out_path, 'redcos')

        # Wavelength calibration
        sci_image.wavelength_solution = master_arcs[j].wavelength_solution
        sci_image.get_wavelengths(gris_table)
        sci_image.adjust_2d()
        if write_steps:
            write_step(sci_image, out_path, 'redcoslam')
            
        sci_ims = sci_ims + [sci_image]
    
    sci_image = sci_ims[0].combine_data(sci_ims[1]) # Combine data from different chips.
    if i==0: # Write the new slit positions in the combined image.
        np.savetxt(f'{calib_path}/slit_bounds_combined.dat', np.column_stack([sci_image.lower_bounds, sci_image.upper_bounds]))
    write_step(sci_image, out_path, 'combined') # Write combined and calibrated images.

# Define aperture sizes over which to extract the data.
apertures = np.arange(25,75,5)
np.savetxt(f'{calib_path}/apertures.txt',apertures)
n_apertures = len(apertures)

n_stars = 2 # Number of slits / sources in the combined image.
combined_files = sorted(glob.glob(f'{out_path}/*{date}*_combined.fits')) # Read the combined files.
sky_region = np.array([13,33,247,267]) # Define the region within each slit to be used for sky background.

lower, upper = read_bounds(f'{calib_path}/slit_bounds_combined.dat') # read slit boundaries from save file.

start_time = time.time() # Track duration of the source extraction/cleaning/alignment process.

star_list = [] # List of extracted sources.
for i in range(n_stars):
    # Create a source instance and extract the flux from the combined image files, then save.
    star_n = Source(combined_files, n_apertures, i+1, gris_table)
    star_n.extract(upper[i], lower[i], sky_region, apertures)
    pickle.dump(star_n, file=open(f"{out_path}/extracted_{i}.pkl",'wb'))
    
"""
This is where this example stops working, as it does not have enough science images included to do a proper outlier rejection in the cleaning.
"""
    
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
    star_n.align_specs(calib_path, grism, align_order, print_stuff=False)
    star_list = star_list + [star_n]
    
    time_diff = int(np.around(time.time() - start_time))
    print('Processing {0} star(s) took {1} hours {2} minutes and {3} seconds.'.format(i+1,time_diff//3600, time_diff%3600//60, time_diff%60))

# Write extracted sources to disk.
pickle.dump(star_list, file=open('{out_path}/stars_{band_name}.pkl','wb'))

# Specify where light curves get stored. Create a new folder if necessary.
lc_path = 'lcs'
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

