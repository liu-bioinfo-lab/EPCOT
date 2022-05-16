# EPCOT data inputs

EPCOT takes inputs of one-hot representations of DNA sequences and DNase-seq profiles. For DNA sequence, we use the reference genome hg38 whose fasta file is downloaded from [UCSC genome browser](http://hgdownload.cse.ucsc.edu/goldenPath/hg38/bigZips/). You can use [reference_genome.py](https://github.com/zzh24zzh/EPCOT/blob/master/Data/reference_genome.py) to transform the fasta file to one-hot matrices.
## Processing DNase-seq
### Dependencies
* samtools (1.11)
* deeptools (3.5.1)


Since the only cell-type specific inputs of EPCOT are DNase-seq profiles, we use [samtools](https://github.com/samtools/samtools) and [deepTools](https://github.com/deeptools/deepTools) bamCoverage's RPGC normalization to generate normalized bigWig files from bam files. The 'effectiveGenomeSize' in RPGC is calculated using [unique-kmers.py](https://github.com/dib-lab/khmer/blob/master/scripts/unique-kmers.py). 

An example of GM12878 with ENCODE accession numbers [ENCFF020WZB](https://www.encodeproject.org/experiments/ENCSR000EMT/) and [ENCFF729UYK](https://www.encodeproject.org/experiments/ENCSR000EMT/), and mapped read length of 36 is provided below.
```
#download bam files and blacklist
wget -O GM12878_rep1.bam https://www.encodeproject.org/files/ENCFF020WZB/@@download/ENCFF020WZB.bam
wget -O GM12878_rep2.bam https://www.encodeproject.org/files/ENCFF729UYK/@@download/ENCFF729UYK.bam
wget -O black_list.bed.gz https://www.encodeproject.org/files/ENCFF356LFX/@@download/ENCFF356LFX.bed.gz
gunzip black_list.bed.gz

#merge and index bam
samtools merge GM12878.bam GM12878_rep*.bam
samtools index GM12878.bam

#generalize normalized coverage track
bamCoverage --bam GM12878.bam -o GM12878_dnase.bigWig --outFileFormat bigwig --normalizeUsing RPGC --effectiveGenomeSize 2559804523 
--ignoreForNormalization chrX chrM --Offset 1 --binSize 1 --numberOfProcessors 24 --blackListFileName black_list.bed --skipNonCoveredRegions

# transform bigWig to numpy arrays saved as dictionary structure where the keys are chromosomes
python dnase_processing.py GM12878_dnase.bigWig
```
If you use our trained model to perform cross-cell type prediction on your own DNase-seq data, please check if the normalized DNase has similar distribution to the DNase of our training cell lines.
