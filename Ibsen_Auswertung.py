'''
Created on 14.04.2016

@author: ried_st
'''


'''
plot_reflectance_winnowed:
    - show average over all toggle
    - extra plot with all included spectra
    
median vs mean problem
'''

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

#Just change the input directory
input_directory = r'C:\Users\ried_st\OneDrive\Austausch\CoastMap 2016\T4\ST01'




def read_data(filename):
    '''
    takes the input directory from line 22 and a filename as input
    output: writes a figure of the input file into the same directory
    '''
    
    
    input_filename = filename + '.asc'
    ibsendata_directory = os.path.join(input_directory, input_filename)
    #ibsendata_directory_noextension = os.path.join(input_directory, filename)
    
    
    data_matrix = []
    
    with open(ibsendata_directory, 'r') as ibsendata:
        searchlines = ibsendata.readlines()
        
    for i, line in enumerate(searchlines):
        if '[DataRaw]' in line: # metadata is collected
            beginning_data = i
            tmp = searchlines[i+1].split()
            number_columns = len(tmp) # gets the number of columns, eg 33 for 30 measurements
            comment = searchlines[i-10] # gets the line which contains the comment in the usual format
            
    for i, line in enumerate(searchlines):
        if (i>beginning_data+100): # only reads the relevant range (942 is lower limit, below plotting not sensible because of detector noise)
            row2 = np.array([float(w) for w in line.split()])
            data_matrix.append(row2)
    
    np_data = np.array(data_matrix)
    np_data = np.transpose(np_data) #columns contain the formatted data
    return([np_data, number_columns, comment])



def readandplot(filename):
    '''
    takes the input directory from line 22 and a filename as input
    output: writes a figure of the input file into the same directory
    '''
    
    
    input_filename = filename + '.asc'
    ibsendata_directory = os.path.join(input_directory, input_filename)
    ibsendata_directory_noextension = os.path.join(input_directory, filename)
    
    
    data_matrix = []
    
    with open(ibsendata_directory, 'r') as ibsendata:
        searchlines = ibsendata.readlines()
        
    for i, line in enumerate(searchlines):
        if '[DataRaw]' in line: # metadata is collected
            beginning_data = i
            tmp = searchlines[i+1]
            tmp =  tmp.split()
            number_rows = len(tmp)
            comment = searchlines[i-10]
    
    
            
    for i, line in enumerate(searchlines):
        if i>beginning_data:
            row = np.array(line.split())
            data_matrix.append(row)
    
    np_data = np.array(data_matrix)
    np_data = np.transpose(np_data) #columns contain the formatted data
    
    #next chapter: plot all in one
    matrix = np_data
    
    fig = plt.figure(figsize=(18, 10))
    for i in range(3, number_rows):
        xi = matrix[0]
        yi = matrix[i]
        plt.plot(xi, yi)
        
    plt.xlabel('Wavelength', fontsize = 18)
    plt.ylabel('dn', fontsize = 18)
    fig.suptitle(filename + '.asc: ' + '\n' + comment)
    
    #plt.show()
    fig.savefig(ibsendata_directory_noextension + '_rawdata.png')
    plt.close()



def plot_reflectance(dark_current, reference, target):
    '''
    takes the filenames of input files and calculates reflectance
    also uses input_directory specified in line 22
    '''
    
    ibsendata_directory_noextension = os.path.join(input_directory, str(target))
    wavelength = read_data(target)[0][0]
    
    dark_current_avg = read_data(dark_current)[0][1]
    reference_avg = read_data(reference)[0][1]
    target_avg = read_data(target)[0][1]
    

    reference_subtr = np.subtract(reference_avg, dark_current_avg)
    target_subtr = np.subtract(target_avg, dark_current_avg)
    reflectance = target_subtr/reference_subtr

    fig = plt.figure(figsize=(18, 10))
    xi = wavelength
    yi = reflectance
    plt.plot(xi, yi)
        
    plt.xlabel('Wavelength', fontsize = 18)
    plt.ylabel('dn', fontsize = 18)
    fig.suptitle(str(target) + '.asc: ' + '\n' + read_data(target)[2])
    
    fig.savefig(ibsendata_directory_noextension + str(target) + '_' + str(reference) + '-' + str(dark_current) + '.png')
    plt.show()
    plt.close()
    
    
    
def plot_reflectance_winnowed(dark_current, reference, target, std_dark, std_ref, std_tar_plus, std_tar_minus, std_tar_r2, plot_avg_all):
    '''
    takes the filenames of input files and calculates reflectance, outliers are thrown out
    also uses input_directory specified in line 22
    std_dark = multiple of the standard deviation calculated from integrals over dark current spectra. Spectra above or below std_dark*standard deviation are ignored
        (larger value means more spectra are included)
    std_ref = same as std_dark for the reference spectrum
    std_tar_plus = boundary value for brighter spectra, larger value means more spectra are included
    str_tar_minus = boundary value for darker spectra, larger value means more spectra are included
    std_tar_r2 = boundary value for R^2 coefficient. Values with lower R^2 score than mean-standard deviation are excluded. Larger value means more spectra are included
    '''
    
    ibsendata_directory_noextension = os.path.join(input_directory, str(target))
    wavelength = read_data(target)[0][0] #contains the wavelengths for plotting
    
    '''
    reminder:
    read_data returns: return([np_data, number_rows, comment])
    '''
    
    dark = read_data(dark_current)
    #dark_current_avg = dark_current_avg[100:]
    ref = read_data(reference)
    #reference_avg = reference_avg[100:]
    tar = read_data(target)
    #target_avg = target_avg[100:]

    
    # dark current section______________________________________________________________________________________________________________________________________________________________
    dark_sum = np.sum(ref[0][3:ref[1]], axis=1) # integrates over the whole spectrum (for identification of outliers), adds the sum over each dark current spectrum to dark_sum
    dark_sum_mean = np.mean(dark_sum) #gets the mean of all sum values
    dark_sum_std = np.std(dark_sum) #gets standard deviation
    
    dark_use = []
    for i in range(0,dark[1]-3): #throws all dark currents out which are too high or low
        if abs(dark_sum[i] - dark_sum_mean) < std_dark*dark_sum_std: #1.5 times standard deviation is used
            dark_use.append(i)
    
    #dark_use_spectra = np.zeros(924) #preallocation
    dark_use_spectra = []
    for n in dark_use: # get all spectra which have not been sorted out from dark_use
        dark_use_spectra.append(dark[0][n+3]) # n+3 because first 3 rows contain wl, mean and std of all spectra

    dark_use_spectra = np.array(dark_use_spectra) #creates a numpy array from normal python array
    dark_mean = np.mean(dark_use_spectra, axis=0) #gets mean over all curves for further use
    #dark_std = np.std(dark_use_spectra, axis=0)
        
        
    # reference spectrum section________________________________________________________________________________________________________________________________________________________ 
    ref[0][3:ref[1]] -= dark_mean # subtracts dark current from reference
    ref_sum = np.sum(ref[0][3:ref[1]], axis=1) # adds the sum over each reference spectrum to ref_sum
    ref_sum_mean = np.mean(ref_sum) #gets the mean of all sum values
    ref_sum_std = np.std(ref_sum) #gets standard deviation
    
    ref_use = []
    for i in range(0,ref[1]-3): #throws all reference spectra out which are too bright or dark
        if abs(ref_sum[i] - ref_sum_mean) < std_ref*ref_sum_std: #1.5 times standard deviation is used
            ref_use.append(i)
            
    ref_use_spectra = []
    for n in ref_use: # get all spectra which have not been sorted out from ref_use
        ref_use_spectra.append(ref[0][n+3]) # n+3 because first 3 rows contain wl, mean and std of all spectra
        
    ref_use_spectra = np.array(ref_use_spectra) #creates a numpy array from normal python array
    ref_mean = np.median(ref_use_spectra, axis=0) #gets median over all curves for further use
    ref_std = np.std(ref_use_spectra, axis=0)
    ref_mean_plus = np.add(ref_mean, ref_std)
    ref_mean_minus = np.subtract(ref_mean, ref_std)
    
    
    # target spectrum section________________________________________________________________________________________________________________________________________________________
    tar[0][3:tar[1]] -= dark_mean # subtracts dark current from target
    tar_sum = np.sum(tar[0][3:tar[1]], axis=1) # adds the sum over each target spectrum to tar_sum
    tar_sum_mean = np.mean(tar_sum) #gets the mean of all sum values
    tar_sum_std = np.std(tar_sum) #gets standard deviation
    
    tar_use = []
    for i in range(0,tar[1]-3): #throws all target spectra out which are too bright or dark
        if ((tar_sum[i] - tar_sum_mean) < std_tar_plus*tar_sum_std) and ((tar_sum[i] - tar_sum_mean) > -std_tar_minus*tar_sum_std): #contains conditions for including or excluding spectra. Separate values are used for upper and lower limit
            tar_use.append(i)
            
    tar_use_spectra = []
    for n in tar_use: # get all spectra which have not been sorted out from tar_use
        tar_use_spectra.append(tar[0][n+3]) # n+3 because first 3 rows contain wl, mean and std of all spectra
        
    tar_use_spectra = np.array(tar_use_spectra) #creates a numpy array from normal python array
    tar_mean = np.median(tar_use_spectra, axis=0) #gets median over all curves for further use
    tar_std = np.std(tar_use_spectra, axis=0, ddof=1)
    tar_mean_plus = np.add(tar_mean, tar_std)
    tar_mean_minus = np.subtract(tar_mean, tar_std)
    
    tar_use_r2 = []
    for spectrum in tar_use_spectra:
        R2 = r2_score(tar_mean, spectrum) # get R^2 value for each spectrum
        tar_use_r2.append(R2)
        
    tar_use_r2_mean = np.mean(tar_use_r2) #gets the mean of all R^2 values
    tar_use_r2_std = np.std(tar_use_r2) #gets standard deviation of R^2 values
    
    tar_use_r2_spectra = []
    for value in tar_use_r2:
        if value - tar_use_r2_mean > -std_tar_r2*tar_use_r2_std: # condition to throw out spectra with low r^2, higher value means less spectra
            tar_use_r2_spectra.append(tar_use_spectra[tar_use_r2.index(value)]) # appends good spectra to the list
    
    reflectance = tar_mean/ref_mean
    reflectance_plus = tar_mean_plus/ref_mean
    reflectance_minus = tar_mean_minus/ref_mean
    
    print('number of spectra used: dark current:', len(dark_use),' reference:', len(ref_use), ' target:', len(tar_use_r2_spectra))   
    
    # average over all section______________________________________________________________________________________________________________________________________________________
    reference_all = ref[0][1] - dark_mean
    target_all = tar[0][1] - dark_mean
    reflectance_all = target_all/reference_all
    
    
    # plotting section______________________________________________________________________________________________________________________________________________________________
    
    
    fig = plt.figure(figsize=(18, 10))
    plt.plot(wavelength, reflectance)
    plt.plot(wavelength, reflectance_plus, 'r')
    plt.plot(wavelength, reflectance_minus, 'r')
    
    if plot_avg_all == 'y':
        plt.plot(wavelength, reflectance_all, 'g')
        
    plt.xlabel('Wavelength', fontsize = 18)
    plt.ylabel('dn', fontsize = 18)
    fig.suptitle(str(target) + '.asc: ' + '\n' + read_data(target)[2])
    
    fig.savefig(ibsendata_directory_noextension + str(target) + '_' + str(reference) + '-' + str(dark_current) + 'winnowed_.png')
    plt.show()
    plt.close()
    



'''
end of programming section
'''

# for file in os.listdir(input_directory):
#     if file.endswith('.asc'):
#         filename, file_extension = os.path.splitext(file)
#         readandplot(filename)

plot_reflectance_winnowed('darkcurrent000', 'reference000', 'target002', std_dark = 1.5, std_ref = 1.5, std_tar_plus = 2, std_tar_minus = 2, std_tar_r2 = 2, plot_avg_all = 'y')
#plot_reflectance('darkcurrent000', 'reference000', 'target006')