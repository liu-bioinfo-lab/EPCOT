## Downstream model architecture

<img
  src="../Profiles/GEP.png"
  title=""
  style="display: inline-block; margin: 0 auto; max-width: 300px">
  
  
 ## Data
For RNA-seq GEP, we download the datasets of DeepChrome and AttentiveChrome from [https://zenodo.org/record/2652278#.Yow84ZPMKAk](https://zenodo.org/record/2652278#.Yow84ZPMKAk), and GC-MERGE data from [https://github.com/rsinghlab/GC-MERGE/tree/main/src/data](https://github.com/rsinghlab/GC-MERGE/tree/main/src/data).

For CAGE-seq GEP, we download CAGE-seq bam files from ENCODE and use samtools to merge replicates, and then use [bam_cov.py](https://github.com/calico/basenji/blob/master/bin/bam_cov.py) and [data_read.py](https://github.com/liu-bioinfo-lab/EPCOT/blob/main/GEP/cage/data_read.py)  to convert the bam files into bigWig and process the data.

## Download trained downstream models

The model to predict 1kb-reolustion CAGE-seq with transformer layers in the downstream model can be downloaded using following command lines. This model is trained on four cell types: GM12878, K562, HUVEC, and IMR-90.
```
pip install gdown
gdown 1eP-ruOcywlGeQIRuVwWk_QFc4z9j4-jF --output models/cage_transformer.pt
```

## train from scratch

Please refer to the codes []()
