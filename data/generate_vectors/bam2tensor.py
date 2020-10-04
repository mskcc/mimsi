"""
Bam2Tensor Conversion Class

@author: John Ziegler
Memorial Sloan Kettering Cancer Center 
May 2020 

zieglerj@mskcc.org

(c) 2018 Memorial Sloan Kettering Cancer Center.  This program is free software: you may use, redistribute, 
and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, 
either version 3 or later. See the LICENSE file for details

"""

import pysam
import numpy as np
import traceback
from collections import defaultdict


class Bam2Tensor(object):
    def __init__(self, bam_file, normal_bam_file, coverage=50):
        self.bam_file = bam_file
        self.normal_bam_file = normal_bam_file
        self.coverage = coverage
        self.size_select = size_select

    def create_cigar_array(self, cigar_tuples):
        if len(cigar_tuples) == 1:
            return np.array([cigar_tuples[0][0]] * cigar_tuples[0][1])

        expanded = [[pair[0]] * pair[1] for pair in cigar_tuples]
        return np.concatenate(expanded).flat

    def createIndividualBamTensor(self, read_iterator, start, end, cols):
        required_coverage = int(self.coverage)
        results = list()
        row_counter = 0

        try:
            for read1, read2 in read_iterator:
                for read in read1, read2:
                    # skip the read if it doesn't totally overlap the region

                    if read.reference_end is None:
                        continue

                    read_start = read.reference_start
                    read_end = read.reference_end
                    is_reverse = int(read.is_reverse)
                    mapping_q = read.mapping_quality

                    if read_start > start or read_end < end:
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
                        new_row[0, i, :] = [
                            cigar_vals_to_fill[i],
                            mapping_q,
                            is_reverse,
                        ]

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

    def read_pair_generator(
        self,
        bam,
        region_string=None,
        frag_size_filter=False,
        mifs1=80,
        mafs1=140,
        mifs2=200,
        mafs2=300,
    ):
        """
        Generate read pairs in a BAM file or within a region string.
        Reads are added to read_dict until a pair is found.
        """

        def is_frag_size(size):
            if frag_size_filter:
                return (size >= mifs1 and size <= mafs1) or (
                    size >= mifs2 and size <= mafs2
                )
            return True

        read_dict = defaultdict(lambda: [None, None])
        for read in bam.fetch(region=region_string):
            qname = read.query_name
            if qname not in read_dict:
                if read.is_read1:
                    read_dict[qname][0] = read
                else:
                    read_dict[qname][1] = read
            else:
                try:
                    if (
                        read.is_read1
                        and read.tlen == -read_dict[qname][1].tlen
                        and is_frag_size(abs(read.tlen))
                    ):
                        yield read, read_dict[qname][1]
                    else:
                        if read.tlen == -read_dict[qname][0].tlen and is_frag_size(
                            abs(read.tlen)
                        ):
                            yield read_dict[qname][0], read
                except AttributeError:
                    pass
                del read_dict[qname]

    def createTensor(self, chromosome, start, end):
        cols = 0
        tumor_result = None
        normal_result = None

        try:
            # read_iterator = self.bam_file.fetch(chromosome, start, end)
            read_iterator = self.read_pair_generator(
                self.bam_file, chromosome + ":" + str(start) + "-" + str(end), True
            )
            # norm_read_iterator = self.normal_bam_file.fetch(chromosome, start, end)
            norm_read_iterator = self.read_pair_generator(
                self.normal_bam_file, chromosome + ":" + str(start) + "-" + str(end)
            )
            cols = end - start
            tumor_result = self.createIndividualBamTensor(
                read_iterator, start, end, cols
            )
            normal_result = self.createIndividualBamTensor(
                norm_read_iterator, start, end, cols
            )

            if tumor_result is None or normal_result is None:
                return None
        except Exception as e:
            print(e)
        return (tumor_result, normal_result)
