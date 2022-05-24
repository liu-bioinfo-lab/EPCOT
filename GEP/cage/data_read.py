## the scripts below are slightly modified from https://github.com/calico/basenji/blob/master/bin/basenji_data_read.py

from optparse import OptionParser

import os
import sys

import h5py
import intervaltree
import numpy as np
import pandas as pd
try:
  import pyBigWig
except:
  pass
import scipy.interpolate

import collections




def main():
  # example usage: python data_read.py --cl A549
  parser = OptionParser()
  parser.add_option('-b', dest='blacklist_bed',
      help='Set blacklist nucleotides to a baseline value.')
  parser.add_option('--black_pct', dest='blacklist_pct',
      default=0.25, type='float',
      help='Clip blacklisted regions to this distribution value [Default: %default')
  parser.add_option('-c', dest='clip',
      default=10000, type='float',
      help='Clip values post-summary to a maximum [Default: %default]')
  parser.add_option('--clip_soft', dest='clip_soft',
      default=None, type='float',
      help='Soft clip values, applying sqrt to the execess above the threshold [Default: %default]')
  parser.add_option('--clip_pct', dest='clip_pct',
      default=0.9999, type='float',
      help='Clip extreme values to this distribution value [Default: %default')
  parser.add_option('--crop', dest='crop_bp',
      default=0, type='int',
      help='Crop bp off each end [Default: %default]')
  parser.add_option('-i', dest='interp_nan',
      default=False, action='store_true',
      help='Interpolate NaNs [Default: %default]')
  parser.add_option('-s', dest='scale',
      default=1., type='float',
      help='Scale values by [Default: %default]')
  parser.add_option('-u', dest='sum_stat',
      default='sum',
      help='Summary statistic to compute in windows [Default: %default]')
  parser.add_option('-w',dest='pool_width',
      default=1000, type='int',
      help='Average pooling width [Default: %default]')
  parser.add_option('--cl', dest='cl',
                    default='none')
  (options, args) = parser.parse_args()

  options.blacklist_bed='/nfs/turbo/umms-drjieliu/usr/zzh/KGbert/gene_exp/cage-seq/black_list.bed'
  print(options.blacklist_bed,options.cl,options.clip)
  # bigwig file output from bam_cov.py
  genome_cov_file='/scratch/drjieliu_root/drjieliu/zhenhaoz/CAGE-seq/%s_cage.bigWig'%(options.cl)
  # input 250kb regions
  seqs_bed_file='sequences.bed'
  seqs_cov_file='/nfs/turbo/umms-drjieliu/usr/zzh/KGbert/gene_exp/ct_cage/data/%s_seq_cov.h5'%(options.cl)
  ModelSeq = collections.namedtuple('ModelSeq', ['chr', 'start', 'end'])
  assert(options.crop_bp >= 0)

  # read model sequences
  model_seqs = []
  for line in open(seqs_bed_file):
    a = line.split()
    model_seqs.append(ModelSeq(a[0],int(a[1]),int(a[2])))

  # read blacklist regions
  black_chr_trees = read_blacklist(options.blacklist_bed)

  # compute dimensions
  num_seqs = len(model_seqs)
  seq_len_nt = model_seqs[0].end - model_seqs[0].start
  seq_len_nt -= 2*options.crop_bp
  target_length = seq_len_nt // options.pool_width
  assert(target_length > 0)

  # initialize sequences coverage file
  seqs_cov_open = h5py.File(seqs_cov_file, 'w')
  seqs_cov_open.create_dataset('targets', shape=(num_seqs, target_length), dtype='float16')

  # open genome coverage file
  genome_cov_open = CovFace(genome_cov_file)

  # for each model sequence
  for si in range(num_seqs):
    mseq = model_seqs[si]

    seq_cov_nt = genome_cov_open.read(mseq.chr, mseq.start, mseq.end)

    # determine baseline coverage
    baseline_cov = np.percentile(seq_cov_nt, 10)
    baseline_cov = np.nan_to_num(baseline_cov)

    # set blacklist to baseline
    if mseq.chr in black_chr_trees:
      for black_interval in black_chr_trees[mseq.chr][mseq.start:mseq.end]:
        # adjust for sequence indexes
        black_seq_start = black_interval.begin - mseq.start
        black_seq_end = black_interval.end - mseq.start
        seq_cov_nt[black_seq_start:black_seq_end] = baseline_cov

    nan_mask = np.isnan(seq_cov_nt)
    seq_cov_nt[nan_mask] = baseline_cov

    # crop
    if options.crop_bp > 0:
      seq_cov_nt = seq_cov_nt[options.crop_bp:-options.crop_bp]

    # sum pool
    seq_cov = seq_cov_nt.reshape(target_length, options.pool_width)
    if options.sum_stat == 'sum':
      seq_cov = seq_cov.sum(axis=1, dtype='float32')
    elif options.sum_stat == 'sum_sqrt':
      seq_cov = seq_cov.sum(axis=1, dtype='float32')
      seq_cov = seq_cov**0.75
    elif options.sum_stat in ['mean', 'avg']:
      seq_cov = seq_cov.mean(axis=1, dtype='float32')
    elif options.sum_stat == 'median':
      seq_cov = seq_cov.median(axis=1)
    elif options.sum_stat == 'max':
      seq_cov = seq_cov.max(axis=1)
    elif options.sum_stat == 'peak':
      seq_cov = seq_cov.mean(axis=1, dtype='float32')
      seq_cov = np.clip(np.sqrt(seq_cov*4), 0, 1)
    else:
      print('ERROR: Unrecognized summary statistic "%s".' % options.sum_stat,
            file=sys.stderr)
      exit(1)

    # clip
    if options.clip_soft is not None:
      clip_mask = (seq_cov > options.clip_soft)
      seq_cov[clip_mask] = options.clip_soft + np.sqrt(seq_cov[clip_mask] - options.clip_soft)
    if options.clip is not None:
      seq_cov = np.clip(seq_cov, 0, options.clip)

    # scale
    seq_cov = options.scale * seq_cov
    seqs_cov_open['targets'][si,:] = seq_cov.astype('float16')

  # close genome coverage file
  genome_cov_open.close()

  # close sequences coverage file
  seqs_cov_open.close()


def read_blacklist(blacklist_bed, black_buffer=20):
  """Construct interval trees of blacklist
     regions for each chromosome."""
  black_chr_trees = {}

  if blacklist_bed is not None and os.path.isfile(blacklist_bed):
    for line in open(blacklist_bed):
      a = line.split()
      chrm = a[0]
      start = max(0, int(a[1]) - black_buffer)
      end = int(a[2]) + black_buffer

      if chrm not in black_chr_trees:
        black_chr_trees[chrm] = intervaltree.IntervalTree()

      black_chr_trees[chrm][start:end] = True

  return black_chr_trees


class CovFace:
  def __init__(self, cov_file):
    self.cov_file = cov_file
    self.bigwig = False
    self.bed = False

    cov_ext = os.path.splitext(self.cov_file)[1].lower()
    if cov_ext == '.gz':
      cov_ext = os.path.splitext(self.cov_file[:-3])[1].lower()

    if cov_ext in ['.bed', '.narrowpeak']:
      self.bed = True
      self.preprocess_bed()

    elif cov_ext in ['.bw','.bigwig']:
      self.cov_open = pyBigWig.open(self.cov_file, 'r')
      self.bigwig = True

    elif cov_ext in ['.h5', '.hdf5', '.w5', '.wdf5']:
      self.cov_open = h5py.File(self.cov_file, 'r')

    else:
      print('Cannot identify coverage file extension "%s".' % cov_ext,
            file=sys.stderr)
      exit(1)

  def preprocess_bed(self):
    # read BED
    bed_df = pd.read_csv(self.cov_file, sep='\t',
      usecols=range(3), names=['chr','start','end'])

    # for each chromosome
    self.cov_open = {}
    for chrm in bed_df.chr.unique():
      bed_chr_df = bed_df[bed_df.chr==chrm]

      # find max pos
      pos_max = bed_chr_df.end.max()

      # initialize array
      self.cov_open[chrm] = np.zeros(pos_max, dtype='bool')

      # set peaks
      for peak in bed_chr_df.itertuples():
        self.cov_open[peak.chr][peak.start:peak.end] = 1


  def read(self, chrm, start, end):
    if self.bigwig:
      cov = self.cov_open.values(chrm, start, end, numpy=True).astype('float16')

    else:
      if chrm in self.cov_open:
        cov = self.cov_open[chrm][start:end]
        pad_zeros = end-start-len(cov)
        if pad_zeros > 0:
          cov_pad = np.zeros(pad_zeros, dtype='bool')
          cov = np.concatenate([cov, cov_pad])
      else:
        print("WARNING: %s doesn't see %s:%d-%d. Setting to all zeros." % \
          (self.cov_file, chrm, start, end), file=sys.stderr)
        cov = np.zeros(end-start, dtype='float16')

    return cov

  def close(self):
    if not self.bed:
      self.cov_open.close()

if __name__ == '__main__':
  main()
