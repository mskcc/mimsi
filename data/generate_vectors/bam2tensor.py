'''
Bam2Tensor Conversion Class

@author: John Ziegler
Memorial Sloan Kettering Cancer Center 
Nov. 2018

zieglerj@mskcc.org

(c) 2018 Memorial Sloan Kettering Cancer Center.  This program is free software: you may use, redistribute, 
and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, 
either version 3 or later. See the LICENSE file for details

'''

import pysam
import numpy as np
from collections import deque


class Bam2Tensor(object):
    chromosome = '1'
    start = 1
    end = 100


    def __init__(self, bam_file, normal_bam_file, chromosome, start, end, coverage=50):
        self.bam_file = bam_file
        self.normal_bam_file = normal_bam_file
        self.chromosome = chromosome
        self.start = start
        self.end = end
        self.coverage = coverage

    def createIndividualBamTensor(self, read_iterator, cols):
        cols = self.end - self.start
        result_array = np.zeros((0, cols, 3))
        required_coverage = int(self.coverage)

        row_counter = 0
        try:
            for read in read_iterator:
                #skip the read if it doesn't totally overlap the region
                if(read.reference_start > self.start or read.reference_end < self.end):
                    continue

                # get the cigar tuples and covert to a list of cigar values
                cigar_tuples = read.cigartuples
                cigar_vals_for_read = []
                if cigar_tuples is None:
                    continue
                
                for cigar_tuple in cigar_tuples:
                    temp = [cigar_tuple[0]]*cigar_tuple[1]
                    cigar_vals_for_read += temp


                cigar_vals_for_read = np.array(cigar_vals_for_read)
                
                # pull out the cigar values for the region we're interested in
                # for example, if this read aligns to positions 100 - 110, and our region
                # of interest(roi) is 105 - 109 we need to get cigar_vals_for_read[5:9]... to get
                # the start and end index (5 and 9 in our example) we do:
                # start = max(0, roi_start - read_start)
                # end = len(civar_vals_for_read) - 1 - max(0, read_end - roi_end)
                read_start = read.reference_start
                read_end = read.reference_end
                cigar_start = max(0, self.start - read_start)
                cigar_end = len(cigar_vals_for_read) - max(0, read_end - self.end)


                ref_locations = range(self.start, self.end)
                aligned_pairs = read.get_aligned_pairs()
                aligned_cigar_pairs = filter(lambda x: x[1] in ref_locations, aligned_pairs)
                aligned_cigar_indicies = [x[0] for x in aligned_cigar_pairs]
                aligned_cigar_indicies = map(lambda x: -1 if x is None else x, aligned_cigar_indicies)

                aligned_cigar_indicies = np.array(aligned_cigar_indicies)
                cigar_vals_to_fill = cigar_vals_for_read[aligned_cigar_indicies]

               
                # fill and add new row
                new_row = np.zeros((1, cols, 3))
                for i in range(0, len(cigar_vals_to_fill)-1):
                    new_row[0,i,:] = [cigar_vals_to_fill[i], read.mapping_quality, int(read.is_reverse)]
               
                result_array = np.vstack((result_array, new_row))
                row_counter += 1

        except Exception as e:
            print("Exception in creating vector...")
            print(e)
            return None

        if row_counter < required_coverage:
            return None

        return result_array

    def createTensor(self):

        read_iterator = self.bam_file.fetch(self.chromosome, self.start, self.end)
        cols = self.end - self.start
        norm_read_iterator = self.normal_bam_file.fetch(self.chromosome, self.start, self.end)
        
        tumor_result = self.createIndividualBamTensor(read_iterator, cols)
        normal_result = self.createIndividualBamTensor(norm_read_iterator, cols)
        
        if tumor_result is None or normal_result is None:
            return None

        return (tumor_result, normal_result)
        
