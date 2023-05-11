# EPCOT

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7485616.svg)](https://doi.org/10.5281/zenodo.7485616)
<!-- [![figshare](https://a11ybadges.com/badge?logo=figshare)](https://doi.org/10.6084/m9.figshare.22731623.v1) -->


EPCOT (comprehensively predicting <ins>EP</ins>igenome, <ins>C</ins>hromatin <ins>O</ins>rganization and <ins>T</ins>ranscription) is a comprehensive model to jointly predict epigenomic features, gene expression, high-resolution chromatin contact maps, and enhancer activities from DNA sequence and cell-type specific chromatin accessibility data. 

We have developed resources to assist users in predicting other genomic modalities from ATAC-seq. These include a Google Colab notebook
<a target="_blank" href="https://colab.research.google.com/github/liu-bioinfo-lab/EPCOT/blob/main/gradio.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
and a webpage [https://liu-bioinfo-lab.github.io/EPCOT_APP.github.io/](https://liu-bioinfo-lab.github.io/EPCOT_APP.github.io/).



<img
  src="Data/model.png"
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
Please go to the directory [Input/](https://github.com/liu-bioinfo-lab/EPCOT/tree/main/Input) for how to generate the inputs to EPCOT (one-hot repsentations of DNA sequences and normalized DNase-seq).  All the human data used in EPCOT are in reference genome hg38 and the data processing codes are also for hg38 version.

### Download the pre-training model and downstream models
You can download EPCOT models trained on DNA sequence and DNase-seq or ATAC-seq from [Google Drive](https://drive.google.com/drive/folders/1gsveyTgYwlXK5Ntnx5nLKSzIW3JvxLse?usp=share_link) or [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7485616.svg)](https://doi.org/10.5281/zenodo.7485616)


For the trained downstream models and how to train downstream models from scratch, you can go to each correspoding directory [GEP/](https://github.com/liu-bioinfo-lab/EPCOT/tree/main/GEP), [COP/](https://github.com/liu-bioinfo-lab/EPCOT/tree/main/COP), and [EAP/](https://github.com/liu-bioinfo-lab/EPCOT/tree/main/EAP).


### Tutorial
We prepare a Google Colab Notebook [EPCOT_usage.ipynb](https://github.com/liu-bioinfo-lab/EPCOT/tree/main/EPCOT_usage.ipynb) to introduce how to use EPCOT to predict multiple modalities, and provide a Google Colab Notebook [sequence_pattern.ipynb](https://github.com/liu-bioinfo-lab/EPCOT/blob/main/Data/sequence_pattern.ipynb) to introduce how to generate sequence patterns for TFs of interest.

### Documents and webpages
We prepare a [GitHub page](https://zzh24zzh.github.io/epcot.github.io/) to share our TF sequence binding patterns along with Tomtom motif comparison results, and we also summarize the results in an EXCEL file [motif_comparison_summary.xls](https://github.com/liu-bioinfo-lab/EPCOT/blob/main/Data/motif_comparison_summary.xls).
