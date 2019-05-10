#!/usr/bin/env python
# coding: utf-8

# In[2]:


## Imports: 
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import StatisticalMetrics as SM 
import readRthRaw as RTH 
import snrCalc as SNRC
import glob
import re
import os 
from shutil import copyfile
import math
import seaborn as sns
sns.set_context('talk')


# # trackTest-18Apr2019
# 
# ### Experiment Notes: 
# In this experiment three electrophysiology catheters from the same manufacturer batch were placed in a fixture in a PAA phantom aligned along the main magnetic field. They were placed in two positions: near iso-centre, as well as at an off-isocentre position.
# 
# The full MR protocol is described [here](http://panoptes.sri.utoronto.ca:8090/confluence/display/IP/trackTest-18Apr2019+Catheter+Variance+Protocol)
# 
# Data stored at ircci-share/Data/track
# 	•	Full analysis done by Jay Soni
# 
# ### Data Analysis
# For simplicity, only the isocentre positions will be analysed. (Off-isocentre positions require a proprietary unwarping algorithm.)
# 
# ### Localization Algorithms
# Two custom algorithms from the Statistical Metrics module are run in the cells below.
# 
# #### CentAlg1: FWHM
# Is the "Fixed Width Half Max" centroid algorithm with window_scale 2.5 and cut off value at 0.5 
# 
# #### CentAlg2: FWHM
# This is the "Fixed Width Half Max" centroid algorithm with window_scale 2.5 and cut off value at 0 
# 
# #### CentAlg3: PNG
# This is the "Peak-normed Gaussian" centroid algorithm with a peak-normed gaussian-weighted centroid and window width of 8 mm
# 
# #### CentAlg4: PNG
# This is the "Peak-normed Gaussian" centroid algorithm with a peak-normed gaussian-weighted centroid and window width of 1 mm

# In[18]:


## Path/Constant variables:
experiment_path = "trackTest-18Apr2019/"
## this is a local directory, (used to reduce processing time) a global directory can be used as well: 
## experiment_path = "/volumes/ircci-share/data/trackTest-18Apr2019/"

catheters = ["cath284", "cath299", "cath285"]
positions = ["iso"]
sequences = ["fastHadamard", "SRI"]
algos = {"CentAlg1":SM.custm_centroid, "CentAlg4":SM.new_centroid_algorithm}
algo_ind = {}
alg_params = {"CentAlg1":{"CoilLength":2.0,"Cutoff_value":0.5}, "CentAlg4":{"window_width":1}}
coil_label_patternSRI = ["cathcoil4", "cathcoil5"]
coil_num_patt_SRI = ".*cathcoil([0-9]+)-"
coil_label_patternFH = ["cathHVC4", "cathHVC5"]
projection_label_pattern = ["P0", "P1", "P2", "P3"]
dither_label_pattern = ["D0", "D1", "D2"]
fieldofview = ["fov60"] 

linearRecomb = [[-1,1,1,-1],[-1,1,-1,1],[-1,-1,1,1]]
linearRecomb = np.array(linearRecomb)
linearRecomb = (3**0.5/4.0)*linearRecomb

## Ground truth Coordinates: 
Off_Iso_Transform = [8.025, 0.0, -149.71]

coordinates_iso = [[[3.738, 5.541, 1.565], [3.70, 5.564, -6.205]],  ## Cath284  (mm)
                   [[6.638, 5.914, 2.940], [6.774, 6.028, -4.828]],  ## Cath299 (mm)
                   [[5.969, 6.055, 0.381], [5.929, 6.079, -7.389]]]  ## Cath285 (mm)
                        ## Distal (4)          Proximal (5)
    
coordinates_offiso = [[[11.763, 5.541, -148.145], [11.725, 5.564, -155.915]],  ## Cath284  (mm)
                      [[14.663, 5.914, -146.77], [14.799, 6.028, -154.538]],  ## Cath299 (mm)
                      [[13.994, 6.055, -149.329], [13.954, 6.079, -157.099]]]  ## Cath285 (mm)
                          ## Distal (4)             Proximal (5)
    
    
Ground_Truth = [coordinates_iso, coordinates_offiso]


# In[69]:


# Setup arrays to hold data
coordinates_byAlgo = []
alg_index = 0
for alg in algos:
    coordinates_byAlgo.append([[],[],[]])
    algo_ind[alg] = alg_index
    alg_index += 1
trig = []; resp = []; time = []; coils = []; position_arr = []; fovs = []; seqs = []; caths = []


for seq in sequences:
    for cath in catheters: 
        for pos in positions: 
            if (seq == "SRI"): 
                path = experiment_path + seq + "/" + cath + "/" + 'SRI_Catheter_Tracking-' + pos + "/"
                print(path)

                files = glob.glob(path+"cathcoil[4-5]*projections")
                p = re.compile(coil_num_patt_SRI)
                for file in files: 
                    xsize,ysize,zsize,FOV,projNum,triggerTimes,respPhases,timestamps,projComplex = RTH.readProjections(file)
                    fts = RTH.reconstructProjections(projComplex,xsize,ysize)

                    file_suffix = (file.split("/"))[-1]
                    for alg in algos:
                        lst_of_vals = [[],[],[]]
                        #for index, element in enumerate(fts):
                        new_name = "warpedcoords" + alg + '-' + seq + "-" + cath + "-" + pos + "-" + (file_suffix.split("."))[0]
                        txt_file = open(experiment_path + "WarpedCoordinates/" + new_name + ".txt", 'w')
                        axes = 3
                        for index in range(0,len(fts),axes):
                            parms = alg_params[alg]
                            coord = []
                            for axis in range(0,axes):
                                if index + axis >= len(fts):
                                    break
                                parms["ProjectionFT"] = fts[index + axis]
                                parms["FieldOfView"] = FOV
                            
                                centroidInd, centroidCoord = algos[alg](**parms) 
                                                                           ## calculates coil position
                                                                            ## using centroid algorithm
                                coordinates_byAlgo[algo_ind[alg]][axis].append(centroidCoord)
                                lst_of_vals[axis].append(centroidCoord)
                            if algo_ind[alg] == 0:
                                coil = int(p.match(file).group(1))
                                coils.append(int(coil))
                                seqs.append(seq)
                                position_arr.append(pos)
                                fovs.append(FOV)
                                caths.append(cath)
                            
                        for i in range(min(len(lst_of_vals[0]),len(lst_of_vals[1]),len(lst_of_vals[2]))):
                            txt_file.write("{} {} {} \n".format(lst_of_vals[0][i],lst_of_vals[1][i],lst_of_vals[2][i]))
                                                                                 ## writes data to text file
                                                                                  ## first column = x-axis
                                                                                  ## second column = y-axis
                        

            else:
                path = experiment_path + seq + "/" + cath + "/" + 'fastHadamard_Tracking-' + pos + "/"
                print(path)
                for coil_label in (coil_label_patternFH):
                    for alg in algos:
                        p0 = []
                        p1 = []
                        p2 = []
                        p3 = []
                        Projections = [p0,p1,p2,p3]
                        txt_file = open(experiment_path + "WarpedCoordinates/" + "warpedcoords" + alg + "-" + seq + "-" + cath + "-" + pos + "-" + coil_label + ".txt", 'w')

                        ## CentAlg1 : is the standard centroid algorithm with 
                        ##      window_scale 2.5 and cut off value at 0.5 

                        ## CentAlg2 : is the standard centroid algorithm with 
                        ##      window_scale 2.5 and cut off value at 0 

                        ## CentAlg3 : is the new centroid algorithm with 
                        ##      peak normed gaussian and width is 8 mm

                        ## CentAlg4 : is the new centroid algorithm with 
                        ##      peak normed gaussian and width is 1 mm

                        for projection_number, projection_label in enumerate(projection_label_pattern): 
                            fft = [[],[],[]]
                            pNum = [0,0,0]
                            FOVS = [0,0,0]

                            for dither_number, dither_label in enumerate(dither_label_pattern): 
                                file_name = (glob.glob(path + coil_label + projection_label + dither_label + "*.projections"))[0]

                                xsize,ysize,zsize,FOV,projNum,triggerTimes,respPhases,timestamps,projComplex = RTH.readProjections(file_name)
                                fts = RTH.reconstructProjections(projComplex,xsize,ysize)

                                fft[dither_number] = fts 
                                pNum[dither_number] = projNum
                                FOVS[dither_number] = FOV

                            for index in range(min(pNum)):
                                mag0 = abs(fft[0][index])
                                mag1 = abs(fft[1][index])
                                mag2 = abs(fft[2][index])

                                peak0 = max(mag0) 
                                peak1 = max(mag1)
                                peak2 = max(mag2)

                                dither = np.argmax(np.array([peak0,peak1,peak2]))
                                parms = alg_params[alg]
                                parms["ProjectionFT"] = fft[dither][index]
                                parms["FieldOfView"] = FOVS[dither]
                                centroidInd, centroidCoord = algos[alg](**parms)

                                                                                ## calculates coil position
                                                                                ## using centroid algorithm

                                Projections[projection_number].append(centroidCoord)


                        for i in range(min(len(p0), len(p1), len(p2), len(p3))):
                            projection_array = np.array([[p0[i]], [p1[i]], [p2[i]], [p3[i]]])
                            coords = np.dot(linearRecomb, projection_array)

                            txt_file.write("{} {} {} \n".format(float(coords[0]),float(coords[1]),float(coords[2])))
                            for axis in range(0,3):
                                coordinates_byAlgo[algo_ind[alg]][axis].append(float(coords[axis]))
                            if algo_ind[alg] == 0:
                                coil = int(coil_label[-1])
                                coils.append(int(coil))
                                seqs.append(seq)
                                position_arr.append(pos)
                                fovs.append(FOV)
                                caths.append(cath)
                        txt_file.close()
                

print("Done Copying!")


# In[70]:


print("Sequence array len = " + str(len(seqs)))
print("Position array len = " + str(len(position_arr)))
print("Cath array len = " + str(len(caths)))


# In[71]:


# Pandify the warped coordinates and metadata
warpedFrame = pd.DataFrame.from_dict({"Sequence":seqs, "Position": position_arr, "FOV":fovs, "Cath":caths, "Coil":coils})
print("Set up warped frame of length: " + str(len(warpedFrame)) )
axes = {'X':0, 'Y':1, 'Z':2}
for alg in algos:
    for axis in axes:
        colName = alg+"_"+axis
        print("For algo " + alg + " axis: " + axis + ", #coords = " + str(len(coordinates_byAlgo[algo_ind[alg]][axes[axis]])))
        warpedFrame[colName] = pd.Series(coordinates_byAlgo[algo_ind[alg]][axes[axis]],index=warpedFrame.index)

warpedFrame.describe()


# In[72]:


warpedFrame.tail()


# #### Unwarping not performed
# 
# The coil coordinates which have been saved to the WarpedCoordinates folder could have potentially fallen victim to gradient warping and should be unwarped, however the unwarping code is not available outside of SRI. Only near-isocentre positions will be analysed.

# In[43]:


## This bit of code copys the unwarped coordinates for each coil and places them in the correct folder so they can be easily
## accessed for future use. 
counter = 0

for seq in sequences:
    for cath in catheters: 
        for pos in positions:
            for alg in algos:
                if (seq == "SRI"): 
                    path = experiment_path + seq + "/" + cath + "/" + 'SRI_Catheter_Tracking-' + pos + "/"
                    path_warped = experiment_path + "WarpedCoordinates/"

                    files = glob.glob(path_warped + "warpedcoords" + alg + "-" + seq + "-" + cath + "-" + pos + "-cathcoil*.txt")

                    for file in files: 
                        name = (file.split("/"))[-1]
                        copyfile(file, path + name)     
                        counter += 1

                else:
                    path = experiment_path + seq + "/" + cath + "/" + 'fastHadamard_Tracking-' + pos + "/"
                    path_warped = experiment_path + "WarpedCoordinates/"

                    file1_name = "warpedcoords" + alg + "-" + seq + "-" + cath + "-" + pos + "-" + "cathHVC4.txt"
                    file1_copy = "warpedcoords" + alg + "-" + seq + "-" + cath + "-" + pos + "-" + "cathHVC4-warp.txt"
                    file2_name = "warpedcoords" + alg + "-" + seq + "-" + cath + "-" + pos + "-" + "cathHVC5.txt"
                    file2_copy = "warpedcoords" + alg + "-" + seq + "-" + cath + "-" + pos + "-" + "cathHVC5-warp.txt"


                    copyfile(path_warped + file1_name, path + file1_copy)
                    copyfile(path_warped + file2_name, path + file2_copy)
                    counter += 2

print("Done Copying!")
print(counter)


# In[44]:


SRI_iso_error = [[], ## cath284
                 [], ## cath299
                 []] ## cath285

Fh_iso_error =  [[], ## cath284
                 [], ## cath299
                 []] ## cath285

Data_Array = [[Fh_iso_error], #FH 
              [SRI_iso_error]] #SRI  

SRI_iso =       [[], ## cath284
                 [], ## cath299
                 []] ## cath285

Fh_iso =        [[], ## cath284
                 [], ## cath299
                 []] ## cath285

Data_Array2 = [[Fh_iso],  #FH 
               [SRI_iso]]   # SRI

for SIndex, seq in enumerate(sequences): 
    for CIndex, cath in enumerate(catheters): 
        for PIndex, pos in enumerate(positions): 
            
            if (seq == "fastHadamard"):
                path = experiment_path + seq + "/" + cath + "/" + 'fastHadamard_Tracking-' + pos + "/"
            elif (seq == 'SRI'): 
                path = experiment_path + seq + "/" + cath + "/" + 'SRI_Catheter_Tracking-' + pos + "/"
                
            files = glob.glob(path + "warpedcoordsCentAlg4*.txt")
            
            X_coord = []
            Y_coord = []
            Z_coord = []
            
            for file in files: 

                
                file_suffix1 = (((file.split("/"))[-1]).split("-"))[-2] ## cathcoil#      two different file name types 
                file_suffix2 = (((file.split("/"))[-1]).split("-"))[-2] ## cathHVC#      need to  be seperated to determine coil number! 
            
                if((file_suffix2 == "cathHVC4") or (file_suffix1 == "cathcoil4")): 
                    true_position = Ground_Truth[PIndex][CIndex][0] 

                elif((file_suffix2 == "cathHVC5") or (file_suffix1 == "cathcoil5")): 
                    true_position = Ground_Truth[PIndex][CIndex][1]
                
                else: 
                    raise Exception('Error could not find suffix! : {}'.format(file))
                
                X_truth = true_position[0]
                Y_truth = true_position[1]
                Z_truth = true_position[2]
                
                file_coords = open(file, 'r')
                experiment_info = file_coords.readlines()
                file_coords.close()
                
                for line in experiment_info:
                    experimental_coords = (line.split(" ")) 
                    X_coord.append(float(experimental_coords[0]) - X_truth)
                    Y_coord.append(float(experimental_coords[1]) - Y_truth)
                    Z_coord.append(float(experimental_coords[2]) - Z_truth)
            
            X_coord_mean = np.mean(X_coord)
            Y_coord_mean = np.mean(Y_coord)
            Z_coord_mean = np.mean(Z_coord)

            X_std = np.std(X_coord)
            Y_std = np.std(Y_coord)
            Z_std = np.std(Z_coord)

            Std = ((X_std)**2 + (Y_std)**2 + (Z_std)**2)**(1/2)

            Error = ((X_coord_mean)**2 + (Y_coord_mean)**2 + (Z_coord_mean)**2)**(1/2)

            Data_Array[SIndex][PIndex][CIndex].append(Error)
            Data_Array[SIndex][PIndex][CIndex].append(Std)
            Data_Array2[SIndex][PIndex][CIndex].append(X_coord)
            Data_Array2[SIndex][PIndex][CIndex].append(Y_coord)
            Data_Array2[SIndex][PIndex][CIndex].append(Z_coord)

            print(seq + ' ' + cath + ' ' + pos + 'Error {}, Std {}, n = {} \n'.format(Error, Std, len(experiment_info)))

            
Iso_error = [SRI_iso_error[0][0], SRI_iso_error[1][0], SRI_iso_error[2][0]] 
Iso_std = [SRI_iso_error[0][1], SRI_iso_error[1][1], SRI_iso_error[2][1]]

print(Iso_error)
hori_axis = np.array([0.5, 1.5, 2.5])

plt.figure(figsize=(12,10))
plt.title("Absolute error in position of micro-coils for varying catheters")
plt.ylabel("Absolute Error (mm)")
plt.xlabel("Catheters")
plt.errorbar(hori_axis, Iso_error, yerr = Iso_std, fmt = "--o", label = 'Isocenter')
#plt.errorbar(hori_axis, Off_error, yerr = Off_std, fmt = "--^", label = 'Off-isocenter')
plt.hlines(1.5, 0, 3, linestyles='dashed', label='Accuracy Requirement')
plt.legend()
plt.xticks(hori_axis, catheters)
plt.show()
        
print('Done!')


# In[34]:


## temp copy paste to graph multiple algorithms on same chart 

SRI_iso_PNG_error = [1.00, 2.67, 1.70]           ## Cath284, Cath299, Cath285
SRI_iso_PNG_std = [1.18, 1.47, 1.71] 

SRI_iso_FWHM_error = [1.21, 3.04, 2.66]          ## Cath284, Cath299, Cath285
SRI_iso_FWHM_std = [1.31, 1.57, 0.79] 

hori_axis = np.array([1,2,3])

plt.figure(figsize=(12,10))
plt.title("Absolute error in position of micro-coils for varying catheters")
plt.ylabel("Absolute Error (mm)")
plt.xlabel("Catheters")
plt.errorbar(hori_axis + 0.55, SRI_iso_FWHM_error, yerr = SRI_iso_FWHM_std, color = 'blue', fmt = "--o", label = 'FWHM')
plt.errorbar(hori_axis + 0.45, SRI_iso_PNG_error, yerr = SRI_iso_PNG_std, color = 'orange', fmt = "--^", label = 'PNG')
plt.hlines(1.5, 1, 4, linestyles='dashed', label='Accuracy Requirement')
plt.legend()
plt.xticks(hori_axis + 0.5, catheters)
plt.show()


# In[ ]:




