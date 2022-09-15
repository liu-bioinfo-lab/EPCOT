# EPCOT


EPCOT (comprehensively predicting <ins>EP</ins>igenome, <ins>C</ins>hromatin <ins>O</ins>rganization and <ins>T</ins>ranscription) is a comprehensive model to jointly predict epigenomic features, gene expression, high-resolution chromatin contact maps, and enhancer activities from DNA sequence and cell-type specific chromatin accessibility data. 

<img
  src="Profiles/model.png"
  title=""
  style="display: inline-block; margin: 0 auto; max-width: 300px">
  
 ## Dependencies
* einops (0.3.2)
* kipoiseq (0.5.2)
* numpy (1.19.5)
* torch (1.10.1)
* scipy (1.7.3)
* scikit-learn (1.0.2)

You can use ```conda``` and ```pip``` to install the required packages
```
conda create -n epcot python==3.9
conda activate epcot
pip install -r requirements.txt
```
  

 ## Usage
 
### Prepare inputs to EPCOT
Please go to the directory [Data/](https://github.com/liu-bioinfo-lab/EPCOT/tree/main/Data) for how to generate the inputs to EPCOT (one-hot repsentations of DNA sequences and normalized DNase-seq). All the data and codes used in EPCOT are in reference genome hg38.

### Download the pre-training model and downstream models
You can download our pre-training model trained on DNA sequence and DNase-seq or ATAC-seq from Google Drive
```
pip install gdown
gdown 1_YfpNSv-2ABQV2qSyBxem-y7aJFyRNzz --output pretrain_dnase.pt

### we also provide the pre-training model trained on ATAC-seq using the same four cell lines with DNase-seq
gdown 1aMb3kVmaWZPUzqKmfZs9xWT-QLUGjQQd --output pretrain_atac.pt
```

For the trained downstream models and how to train downstream models from scratch, you can go to each correspoding directory [GEP/](https://github.com/liu-bioinfo-lab/EPCOT/tree/main/GEP), [COP/](https://github.com/liu-bioinfo-lab/EPCOT/tree/main/COP), and [EAP/](https://github.com/liu-bioinfo-lab/EPCOT/tree/main/EAP).

### Tutorial
We prepare a Google Colab Notebook [EPCOT_usage.ipynb](https://github.com/liu-bioinfo-lab/EPCOT/tree/main/EPCOT_usage.ipynb) to introduce how to use EPCOT to predict multiple modalities, and provide a Google Colab Notebook [sequence_pattern.ipynb](https://github.com/liu-bioinfo-lab/EPCOT/blob/main/examples/sequence_pattern.ipynb) to introduce how to generate sequence patterns for TFs of interest.

### Documents and webpages
We prepare a webpage of our TF sequence binding patterns along with Tomtom motif comparison results, please see [epcot.github.io](https://zzh24zzh.github.io/epcot.github.io/) to search for TFs of interest, and we also summarize the results in an EXCEL file [motif_comparison_summary.xls](https://github.com/liu-bioinfo-lab/EPCOT/tree/main/motif_comparison_summary.xls)
