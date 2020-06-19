'''
Bam2Tensor Conversion Class

@author: John Ziegler
Memorial Sloan Kettering Cancer Center 
May 2020 

zieglerj@mskcc.org

(c) 2018 Memorial Sloan Kettering Cancer Center.  This program is free software: you may use, redistribute, 
and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, 
either version 3 or later. See the LICENSE file for details

'''

import pysam
import numpy as np
import traceback


class Bam2Tensor(object):


    def __init__(self, bam_file, normal_bam_file, coverage=50):
        self.bam_file = bam_file
        self.normal_bam_file = normal_bam_file
        self.coverage = coverage


    def create_cigar_array(self, cigar_tuples):
        if len(cigar_tuples) == 1:
            return np.array([cigar_tuples[0][0]]*cigar_tuples[0][1])

        expanded = [[pair[0]]*pair[1] for pair in cigar_tuples]
        return np.concatenate(expanded).flat



    def createIndividualBamTensor(self, read_iterator, start, end, cols):
        required_coverage = int(self.coverage)
        results = list()

        try:
            for read in read_iterator:
                #skip the read if it doesn't totally overlap the region

                if read.reference_end is None:
                    continue

                read_start = read.reference_start
                read_end = read.reference_end
                is_reverse = int(read.is_reverse)
                mapping_q = read.mapping_quality

                if(read_start > start or read_end < end):
                    continue

                # get the cigar tuples and covert to a list of cigar values
                cigar_tuples = read.cigartuples
                if cigar_tuples is None:
                    continue

                
                cigar_vals_for_read = self.create_cigar_array(cigar_tuples)
                
                # pull out the cigar values for the region we're interested in
                # for example, if this read aligns to positions 100 - 110, and our region
                # of interest(roi) is 105 - 109 we need to get cigar_vals_for_read[5:9]... =
                cigar_start = start - read_start
                cigar_end = cigar_start + cols

                cigar_vals_to_fill = cigar_vals_for_read[cigar_start:cigar_end]
               
                # fill and add new row
                new_row = np.zeros((1, cols, 3))
                for i in range(0, len(cigar_vals_to_fill)):
                    new_row[0,i,:] = [cigar_vals_to_fill[i], mapping_q, is_reverse]
                
                results.append(new_row)
                row_counter += 1

        except Exception as e:
            print("Exception in creating vector...")
            print(traceback.format_exc())
            return None

        if row_counter < required_coverage:
            return None

        result_array = np.vstack(results)

        return result_array

    def createTensor(self, chromosome, start, end):
        cols = 0
        try:
            read_iterator = self.bam_file.fetch(chromosome, start, end)
            cols = end - start
            norm_read_iterator = self.normal_bam_file.fetch(chromosome, start, end)
            tumor_result = self.createIndividualBamTensor(read_iterator, start, end, cols)
            normal_result = self.createIndividualBamTensor(norm_read_iterator, start, end, cols)
            
            if tumor_result is None or normal_result is None:
                return None
        except Exception as e:
            print(e)
        return (tumor_result, normal_result)