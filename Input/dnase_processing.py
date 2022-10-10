import pyBigWig
import numpy as np
import pickle
from scipy.sparse import csr_matrix
from optparse import OptionParser

def DNase_processing():
    usage = 'usage: %prog [options] <bigWig_file>'
    parser = OptionParser(usage)
    (options, args) = parser.parse_args()
    if len(args) != 1:
        parser.error('Please justify the DNase-seq bigwig file path')
    else:
        dnase_file = args[0]
    chr_lens = [248956422, 242193529, 198295559, 190214555, 181538259, 170805979, 159345973,
                145138636, 138394717, 133797422, 135086622, 133275309, 114364328, 107043718,
                101991189, 90338345, 83257441, 80373285, 58617616, 64444167, 46709983, 50818468, 156040895]
    bw = pyBigWig.open(dnase_file)
    signals = {}
    for chrom, length in bw.chroms().items():
        try:
            if chrom == 'chrX':
                chr = 'X'
            else:
                chr = int(chrom[3:])
        except Exception:
            continue
        temp = np.zeros(length)
        intervals = bw.intervals(chrom)
        for interval in intervals:
            temp[interval[0]:interval[1]] = interval[2]
        if chr == 'X':
            seq_length = chr_lens[-1] // 1000 * 1000
        else:
            seq_length = chr_lens[chr - 1] // 1000 * 1000
        signals[chr] = csr_matrix(temp.astype('float32')[:seq_length])
        print(dnase_file, seq_length, np.mean(signals[chr]))
    with open(dnase_file.replace('bigWig', 'pickle'), 'wb') as file:
        pickle.dump(signals, file)

if __name__=='__main__':
    DNase_processing()
