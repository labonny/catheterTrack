import numpy as np 
from scipy.stats import * 
import readRthRaw as RTH 
import snrCalc as SNRC
import math

## Customizable Centroid Algorithm 

## custm_centroid requires a "projectionFT" which is an array of Complex floats which represents the fourier transform of 
## the time domain signal, a "FieldOfView" which is a float (FOV is in mm), the coil length in mm, the window scale factor
## which determins the area around the peak to apply the centroid algorithm to and a cutoff value which ignores any signal
## under a certain intensity. It returns the coordinates of the centroid and its index on the ProjectionFT graph. 
def custm_centroid(ProjectionFT, FieldOfView, CoilLength, window_scale_factor=2.5, Cutoff_value=0.5):
    size = len(ProjectionFT)
    res = FieldOfView / size
 
    mag = abs(ProjectionFT)
    peak = max(mag)
    peakInd = list(mag).index(peak)
         
    centroidHalfWindowSize = math.floor(math.ceil(window_scale_factor * (CoilLength / res)) / 2.0)
    sum1 = 0
    sum2 = 0
    
    for index in range(peakInd - centroidHalfWindowSize, peakInd + centroidHalfWindowSize): 
        if mag[index] >= Cutoff_value * peak:
            sum1 += mag[index] * index
            sum2 += mag[index] 
    
    centroidInd = sum1/sum2   ## Note that this is an interpolated index and thus will be a float which cannot be 
                              ## used to call this element of the array ProjectionFT! 
    centroidCoord = res * (centroidInd - (size/2))  ## this is the transform from centroid index to centroid coord
    
    return centroidInd, centroidCoord  

## New centroid algorithm 

## new_centroid_algorithm requires a "projectionFT" which is an array of Complex floats which represents the fourier transform of 
## the time domain signal, a "FieldOfView" which is a float (FOV is in mm), the "window_width" which determins the area (mm) around 
## peak to apply the centroid algorithm. It returns the coordinates of the centroid and its index on the ProjectionFT graph. 

def peak_normed_gaussian(x):                         ## helper function to generate a gaussian centered at x = 0 with max height = 1! 
    sigma_squared = 10
    y = np.exp((-1/(2*sigma_squared)) * ((x)**2))        
    return y 

def weighted_window(center,half_window_width,x,y):  ## a weight function to generate a soft boundry to our window! 
    dist_from_center = abs(x - center)              
    if (dist_from_center <= half_window_width):     ## scale by 1 if inside the window
        return y * 1

    else: 
        dist_out = dist_from_center - half_window_width  ## scale down from 1 as a gaussian based on how far out you are from window 
        return y * peak_normed_gaussian(dist_out)

# ## Quick test to visualize the weigth function: 
# x = np.linspace(-20,20,41)
# y = [] 
# for i in x:
    # y.append(weighted_window(0,4,i,1))

# plt.plot(x,y, marker='o', linestyle='--')
# plt.show()

def new_centroid_algorithm(ProjectionFT, FieldOfView, window_width=8): 
    size = len(ProjectionFT)
    res = FieldOfView / size  
    window_half_width = math.ceil(math.ceil(window_width / res) / 2.0)

    iteration = 0    ## initialize the iteration number to be 0
    delta = size + 1 ## just initializing the delta value to be greater then the largest delta possible

    RealProjection = np.abs(ProjectionFT)    ## convert complext FFT into a real array
     
    peak = max(RealProjection)               ## find peak and determine its index 
    peakInd = list(RealProjection).index(peak)

    sum1 = 0   ## initialize Sum 1 and 2 for the centroid calculation 
    sum2 = 0

    for index in range(len(RealProjection)):         ## iterate over the entire array 
        indexCoord = res * (index - (size / 2))        ## convert index into spatial position (in mm)

        sum1 += weighted_window(peakInd, window_half_width, index, RealProjection[index]) * index  ## apply window to signal and calculate centroid
        sum2 += weighted_window(peakInd, window_half_width, index, RealProjection[index])

    centroidInd1 = sum1/sum2   ## Note that this is an interpolated index and thus will be a float which cannot be 
                              ## used to call this element of the array ProjectionFT! 

    centroidCoord1 = res * (centroidInd1 - (size/2)) 

    final_centroid_coord = 0 

    while(0.01 < delta):      ## iteratively repeat this process but shift center of window to the centroid index ! 
        iteration += 1  ## re-initialize sums and add 1 to iteration number 
        sum1 = 0
        sum2 = 0

        Ind = centroidInd1   ## we must round this value because the centroid calculation yeilds an interpolated index (float) which cannot be called on a list! 
        
        for index in range(len(RealProjection)):  ## iterate over the signal 

            sum1 += weighted_window(Ind, window_half_width, index, RealProjection[index]) * index  ## apply window (centered at centroid index) to signal
            sum2 += weighted_window(Ind, window_half_width, index, RealProjection[index])               ## to calculate centroid

        centroidInd2 = sum1/sum2
        centroidCoord2 = res * (centroidInd2 - (size/2))

        Ind2 = centroidInd2
        delta = abs(Ind2 - Ind)

        centroidInd1 = centroidInd2
        final_centroid_coord = centroidCoord2

        if (iteration == 25):       ## break if we hit 25 iterations as chances are we are stuck in a loop between two centroid index values! 
            
            break

    #print('iteration = {}'.format(iteration)) ## extra information 
    return(centroidInd2, final_centroid_coord)

## Peak Value 

## peak_val requires a "projectionFT" which is an array of Complex floats which represents the fourier transform of 
## the time domain signal, a "FieldOfView" which is a float (FOV is in cm) and the coil length in cm. Returns the index
## of the peak on the ProjectionFT graph, its coordinate in space, and its signal intensity value
def peak_val(ProjectionFT, FieldOfView, CoilLength):
    xsize = len(ProjectionFT)
    xres = FieldOfView / xsize
    
    mag = abs(ProjectionFT)
    peak = max(mag)
    peakInd = list(mag).index(peak)
    peakCoord = xres * (peakInd - (xsize/2))
    
    return peakInd, peakCoord, peak


# read_proj requires a string "file_path" which is the path to a .projections file
# it returns a series of lists, floats and arrays which are utilized to graph the projection file
# more information in the readRthRaw.py file under readProjections function and reconstructProjections function
def read_proj(file_path):
    xsize,ysize,zsize,fov,projNum,triggerTimes,respPhases,times,projComplex = readRthRaw.readProjections(file_path)
    fts = readRthRaw.reconstructProjections(projComplex,xsize,ysize)
    return fov,projNum,triggerTimes,respPhases,times,fts

## FH_to_xyz requires a list of four float values which it can then map back to x,y,z coordinates
## returns a 1D array of length 3, where x = arr[0], y = arr[1], z = arr[2]
## note the order of the list must be [excitation1, excitation2, excitation3, excitation4]
## readRthRaw collects projections in this order, more information on the excitations can be found in the tracking manual
## (getting started on track confluence page)
def FH_to_xyz(lst): 
    FH_linear_Transform_matrix = ((3**0.5)/4)*np.array([[-1,1,1,-1],[-1,1,-1,1],[-1,-1,1,1]])
    coord_matrix = np.array([[lst[0]], [lst[1]], [lst[2]], [lst[3]]])
    return np.dot(FH_linear_Transform_matrix, coord_matrix)


## Normality Tests for the Data Set: 

## The NormTest function takes a 1D array of data points, an int representing the significance level 
## (as a percentage, ex: NormTest(data, 20) for 20% significance) and an optional bool debug which will print info
## about the test. The function returns True if all three tests pass and returns False otherwise! 
## The p value can be thought of as: assuming the following data follows a normal distribution what is 
## the likelihood that we would see a given result. If the p value is very small then its not likely that our data follows a
## normal distribution. On the contrary, if the p_value is large (larger than our cut of significance value) then we can 
## safely assume that our data is normally distributed! 
## Common significance value cut-offs in literature are 5%, though I personally perfer using 10%-15% to be extra sure! 

def NormTest(data, significance = 10, debug=False):
    
    stat1,p1 = shapiro(data)             ## Run the normality tests from scipy package
    stat2,p2 = normaltest(data)
    results = anderson(data) 
    significance_float = significance / 100  ## calculate decimal value of significance 
    
    if (debug):
        print("significance: {}%".format(significance))
        print("shapiro: {}".format(p1))
        print("normal test: {}".format(p2))
        print("anderson: (statistic) {}".format(results.statistic))
    
    anderson_test_result = False
    shapiro_test_result = False
    normal_test_result = False
    
    for i in range(len(results.critical_values)):                                     ## anderson test does not produce
        sl, cv = results.significance_level[i], results.critical_values[i]            ## p values, it rather has cut off 
        if results.statistic < results.critical_values[i]:                            ## statistic values for each p value
            if (results.significance_level[i] == significance):
                anderson_test_result = True
            if (debug):
                print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
        else:
            if (debug):
                print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))

    
    if (p1 > significance_float):
        shapiro_test_result = True
    
    if (p2 > significance_float):
        normal_test_result = True
    
    if (anderson_test_result and shapiro_test_result and normal_test_result):
        return True
    else:
        if not(anderson_test_result): 
            print("anderson test failed")
        if not(shapiro_test_result):
            print("shapiro test failed")
        if not(normal_test_result):
            print("normal test failed")
        return False


## Testing the NormTest function: 
#data = np.random.randn(100)*5 + 50      ### The randomly generated data is X ~ Norm(Mean=50, Std=5)
#print(NormTest(data, 10, debug=True))

## Printing sample mean and sample Standard Deviation! 
#print(np.mean(data))
#print(np.std(data))

## Confidence Interval around the Mean:

## ConfIntervalMean takes a 1D array of data points, an optional list of floats corresponding to confidence percentages 
## (as decimals, ex. ConfIntervalMean(data, two_sided_significances = [0.70, 0.80, 0.90]) for 70%,80% and 90%)
## and an optional bool "debug" which prints additional information about the calculation
## the function returns an array or a value corresponding to the magnitude of the confidence interval.
## The confidence percentages and intervals can be thought of as: given a 90% confidence interval for a data set, if the
## experiment was done multiple times to recreate multiple similare data sets then 90% of those data sets would have the 
## true mean fall within their confidence interval. 

def ConfIntervalMean(data, two_sided_significances = [0.9, 0.95, 0.99], debug=False):
    
    n = len(data) ## number of samples
    sample_std = np.std(data, dtype=np.float64) ## calculates standard deviation (float64 is more accurate)
    sample_mean = np.mean(data)
    
    one_sided_significances = []                     ## the t.ppf function gives p(T<=t) which are one sided probabilities 
    for i in two_sided_significances:                ## so we must convert the two sided to a one sided probability to get 
        one_sided_significances.append((i + 1)/2)    ## the same cut off value! 
    
    t_scores = t.ppf(one_sided_significances, n)     ## This function gives us the t_scores which we will then transform to
                                                                     ## to get the magnitude of the intervals themselves
    magnitude_of_interval = t_scores * (sample_std / (n)**0.5)
    
    
    if(debug): 
        for i in range(len(two_sided_significances)): 
            print("The {}% significance interval is (+/-){} around the sample mean {}".format((two_sided_significances[i]*100), 
                                                                                    magnitude_of_interval[i], sample_mean))
    
    return magnitude_of_interval, sample_mean    


## Testing the ConfIntervalMean function: 
#print(ConfIntervalMean(data, debug=True))


## Confidence Interval around the Standard Deviation: 

## ConfIntervalStd takes a 1D array of data points, an optional list of floats corresponding to confidence percentages 
## (as decimals, ex. ConfIntervalStd(data, one_sided_significances = [0.70, 0.80, 0.90]) for 70%,80% and 90%)
## and an optional bool "debug" which prints additional information about the calculation
## the function returns an array or a value corresponding to the confidence interval.
## The confidence percentages and intervals can be thought of as: given a 90% confidence interval for a data set, if the
## experiment was done multiple times to recreate multiple similare data sets then 90% of those data sets would have the 
## true standard deviation fall within their confidence interval. 

def ConfIntervalStd(data, one_sided_significanes = [0.9, 0.95, 0.99], debug = False): 
    
    n = len(data)
    sample_var = np.var(data, dtype=np.float64)  ## calculates variance (Std^2) 
    
    invert_one_sided_significances = []              ## the chi2.ppf function gives p(U<=a) but we have probabilities for  
    for i in one_sided_significanes:                 ## p(a<=U) so we need to convert by doing (1 - p) = new_p
        invert_one_sided_significances.append(1 - i)    
    
    chi2_scores = chi2.ppf(invert_one_sided_significances, n)  
    max_one_side_interval = ((1/chi2_scores) * ((n-1) * sample_var))**0.5
   
    ## This function gives us the t_scores which we will then transform to get the magnitude of the intervals themselves
    
    if(debug): 
        for i in range(len(one_sided_significanes)): 
            print("The {}% significance interval is [0,{}]".format((one_sided_significanes[i]*100), max_one_side_interval[i]))
    
    return max_one_side_interval    


## Testing the ConfIntervalStd function:
#print(ConfIntervalStd(data, debug=True))



## Error in precision  

## ErrorPrecision takes a float mean, float std, optional float thresh_hold and optional bool debug. The mean and std are
## used to calculate the error in precision of Gaussian data! The thresh-hold tells you what level of accuracy to use when 
## constricting the interval! The bool "debug" prints additional information about the calculation

def ErrorPrecision(mean, std_int, mean_int, thresh_hold = .90, debug = False): 
    a = norm.ppf((thresh_hold / 2) + 0.5)              ## Z-score values are saved as less than values so we must
    half_width = a*std_int + mean_int                  ## transform our center based area to a one sided area
    interval = [mean - half_width, mean + half_width]  ## This is the final interval with 90% probability
    
    if (debug): 
        print("The probability thresh-hold is: {}".format(thresh_hold))
        print("The half width is: {}".format(half_width))
        print("The interval is: {} - {}".format(interval[0],interval[1]))
    
    return half_width, interval

## ErrorPrecision2 takes a float mean and float mean_int. 
def ErrorPrecision2(mean, mean_int): 
    half_width = mean_int                  
    interval = [mean - half_width, mean + half_width]  ## This is the final interval
    
    return half_width, interval


## Testing the ErrorPrecision function:
#mean = 50 
#std = 5
#print(ErrorPrecision(mean,std,debug = True))


## Error in accuracy 

## ErrorAccuracy takes a float mean and a float grount_truth and returns the absolute difference between the two! 

def ErrorAccuracy(mean, ground_truth, debug = False):
    Error = abs(ground_truth - mean)                  ## Just produce the difference between the ground truth and the mean! 
    
    if (debug):
        print("The error in accuracy is: {}".format(Error))
    
    return Error


## Testing the ErrorAccuracy function: 
#mean = 51.5 
#ground_truth = 50
#print(ErrorAccuracy(mean,ground_truth, debug=True))

















