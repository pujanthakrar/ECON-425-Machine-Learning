

import numpy as np
from pandas import read_csv

#### functions 

##### Function1: import data 
def download_data(fileLocation, fields):
    '''
    Downloads the data for this script into a pandas DataFrame. Uses columns indices provided
    '''

    frame = read_csv(
        fileLocation,
        
        # Specify the file encoding
        # Latin-1 is common for data from US sources
        encoding='latin-1',
        #encoding='utf-8',  # UTF-8 is also common

        # Specify the separator in the data
        sep=',',            # comma separated values

        # Ignore spaces after the separator
        skipinitialspace=True,

        # Generate row labels from each row number
        index_col=None,

        # No header names
        header=None,          # use the first line as headers

        usecols=fields
    )

    # Return the entire frame
    return frame

#### Function 2: transform data to numbers
def transtonumber(string,names):
    output = np.zeros((1,len(string)))
    for i in range(len(string)):
        for j in range(len(names)):
            if string[i] == names[j]:
                output[0][i] = j
    return output

#### Function 3: data normalization 
def rescaleNormalization(dataArray):
    min = dataArray.min()
    denom = dataArray.max() - min
    newValues = []
    for x in dataArray:
        newX = (x - min) / denom
        newValues.append(newX)
    return newValues 

        

