# Count variants in each sample
This Python Jupyter notebook counts occurrences of each barcode in each sample from Illumina barcode sequencing, and adds these counts to the codon variant table.

## Set up analysis
### Import Python modules.
Use [plotnine](https://plotnine.readthedocs.io/en/stable/) for ggplot2-like plotting.

The analysis relies heavily on the Bloom lab's [dms_variants](https://jbloomlab.github.io/dms_variants) package:


```python
import glob
import itertools
import multiprocessing
import multiprocessing.pool
import os
import warnings

import alignparse
import alignparse.targets

import dms_variants.codonvarianttable
from dms_variants.constants import CBPALETTE
import dms_variants.illuminabarcodeparser
import dms_variants.utils
import dms_variants.plotnine_themes

from IPython.display import display, HTML

import pandas as pd

from plotnine import *

import yaml
```

Set [plotnine](https://plotnine.readthedocs.io/en/stable/) theme to the gray-grid one defined in `dms_variants`:


```python
theme_set(dms_variants.plotnine_themes.theme_graygrid())
```

Versions of key software:


```python
print(f"Using alignparse version {alignparse.__version__}")
print(f"Using dms_variants version {dms_variants.__version__}")
```

    Using alignparse version 0.2.4
    Using dms_variants version 0.8.9


Ignore warnings that clutter output:


```python
warnings.simplefilter('ignore')
```

### Parameters for notebook
Read the configuration file:


```python
with open('config.yaml') as f:
    config = yaml.safe_load(f)
```

Make output directory if needed:


```python
os.makedirs(config['counts_dir'], exist_ok=True)
os.makedirs(config['figs_dir'], exist_ok=True)
```

## Input variant tables
Initialize the table of barcode-variant pairs from the respective `process_ccs` notebooks for each background.


```python
variants = pd.read_csv(config['codon_variant_table_file'], na_filter=None)

display(HTML(variants.head().to_html(index=False)))
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>target</th>
      <th>library</th>
      <th>barcode</th>
      <th>variant_call_support</th>
      <th>codon_substitutions</th>
      <th>aa_substitutions</th>
      <th>n_codon_substitutions</th>
      <th>n_aa_substitutions</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>CGG_naive</td>
      <td>lib1</td>
      <td>AAAAAAAAAACACCGG</td>
      <td>6</td>
      <td>GGC119GGT TTA200ACT</td>
      <td>L200T</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <td>CGG_naive</td>
      <td>lib1</td>
      <td>AAAAAAAAAACATGAG</td>
      <td>1</td>
      <td>CAG16TGG</td>
      <td>Q16W</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>CGG_naive</td>
      <td>lib1</td>
      <td>AAAAAAAAAAGCGACG</td>
      <td>1</td>
      <td>GTG156CAT</td>
      <td>V156H</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>CGG_naive</td>
      <td>lib1</td>
      <td>AAAAAAAAAAGGAAAG</td>
      <td>6</td>
      <td>GTG110GGT</td>
      <td>V110G</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>CGG_naive</td>
      <td>lib1</td>
      <td>AAAAAAAAAATATAGA</td>
      <td>1</td>
      <td>TAC47CCA</td>
      <td>Y47P</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>


Pull out a target sequence for matching to the barcode and flanking sequence regions. Note, in this pipeline this is ok because our different backgrounds don't have differing flanks or other features within the actual N16 region covered in Illumina sequencing. If ever placing in-line barcodes here in the future, we would need to modify this.


```python
# get wildtype gene sequence for primary target
targets = alignparse.targets.Targets(seqsfile=config['amplicon'],
                                     feature_parse_specs=config['feature_parse_specs'])
```

## Setup to parse barcodes
Read data frame with list of all barcode runs.


```python
print(f"Reading list of barcode runs from {config['barcode_runs']}")

barcode_runs = (pd.read_csv(config['barcode_runs'])
                .assign(R1=lambda x: x['R1'].str.split('; '))
                )
      
display(HTML(barcode_runs.to_html(index=False)))
```

    Reading list of barcode runs from data/barcode_runs.csv



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>library</th>
      <th>sample</th>
      <th>sample_type</th>
      <th>sort_bin</th>
      <th>concentration</th>
      <th>date</th>
      <th>number_cells</th>
      <th>R1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>lib1</td>
      <td>SortSeq_bin1</td>
      <td>SortSeq</td>
      <td>1</td>
      <td>NaN</td>
      <td>210621</td>
      <td>210800</td>
      <td>[/shared/ngs/illumina/tstarr/210629_D00300_1271_BHM55FBCX3/Unaligned/Project_tstarr/CGGexpr_lib1_bin1-1_S73_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210629_D00300_1271_BHM55FBCX3/Unaligned/Project_tstarr/CGGexpr_lib1_bin1-2_S74_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210629_D00300_1271_BHM55FBCX3/Unaligned/Project_tstarr/CGGexpr_lib1_bin1-3_S75_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>SortSeq_bin2</td>
      <td>SortSeq</td>
      <td>2</td>
      <td>NaN</td>
      <td>210621</td>
      <td>1984000</td>
      <td>[/shared/ngs/illumina/tstarr/210629_D00300_1271_BHM55FBCX3/Unaligned/Project_tstarr/CGGexpr_lib1_bin2-1_S76_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210629_D00300_1271_BHM55FBCX3/Unaligned/Project_tstarr/CGGexpr_lib1_bin2-2_S77_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210629_D00300_1271_BHM55FBCX3/Unaligned/Project_tstarr/CGGexpr_lib1_bin2-3_S78_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>SortSeq_bin3</td>
      <td>SortSeq</td>
      <td>3</td>
      <td>NaN</td>
      <td>210621</td>
      <td>2940000</td>
      <td>[/shared/ngs/illumina/tstarr/210629_D00300_1271_BHM55FBCX3/Unaligned/Project_tstarr/CGGexpr_lib1_bin3-1_S79_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210629_D00300_1271_BHM55FBCX3/Unaligned/Project_tstarr/CGGexpr_lib1_bin3-2_S80_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210629_D00300_1271_BHM55FBCX3/Unaligned/Project_tstarr/CGGexpr_lib1_bin3_S81_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>SortSeq_bin4</td>
      <td>SortSeq</td>
      <td>4</td>
      <td>NaN</td>
      <td>210621</td>
      <td>3575000</td>
      <td>[/shared/ngs/illumina/tstarr/210629_D00300_1271_BHM55FBCX3/Unaligned/Project_tstarr/CGGexpr_lib1_bin4-1_S82_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210629_D00300_1271_BHM55FBCX3/Unaligned/Project_tstarr/CGGexpr_lib1_bin4-2_S83_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210629_D00300_1271_BHM55FBCX3/Unaligned/Project_tstarr/CGGexpr_lib1_bin4-3_S84_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>SortSeq_bin1</td>
      <td>SortSeq</td>
      <td>1</td>
      <td>NaN</td>
      <td>210621</td>
      <td>275500</td>
      <td>[/shared/ngs/illumina/tstarr/210629_D00300_1271_BHM55FBCX3/Unaligned/Project_tstarr/CGGexpr_lib2_bin1-1_S85_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210629_D00300_1271_BHM55FBCX3/Unaligned/Project_tstarr/CGGexpr_lib2_bin1-2_S86_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210629_D00300_1271_BHM55FBCX3/Unaligned/Project_tstarr/CGGexpr_lib2_bin1-3_S87_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>SortSeq_bin2</td>
      <td>SortSeq</td>
      <td>2</td>
      <td>NaN</td>
      <td>210621</td>
      <td>2146000</td>
      <td>[/shared/ngs/illumina/tstarr/210629_D00300_1271_BHM55FBCX3/Unaligned/Project_tstarr/CGGexpr_lib2_bin2-1_S88_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210629_D00300_1271_BHM55FBCX3/Unaligned/Project_tstarr/CGGexpr_lib2_bin2-2_S89_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210629_D00300_1271_BHM55FBCX3/Unaligned/Project_tstarr/CGGexpr_lib2_bin2-3_S90_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>SortSeq_bin3</td>
      <td>SortSeq</td>
      <td>3</td>
      <td>NaN</td>
      <td>210621</td>
      <td>2128000</td>
      <td>[/shared/ngs/illumina/tstarr/210629_D00300_1271_BHM55FBCX3/Unaligned/Project_tstarr/CGGexpr_lib2_bin3-1_S91_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210629_D00300_1271_BHM55FBCX3/Unaligned/Project_tstarr/CGGexpr_lib2_bin3-2_S92_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210629_D00300_1271_BHM55FBCX3/Unaligned/Project_tstarr/CGGexpr_lib2_bin3-3_S93_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>SortSeq_bin4</td>
      <td>SortSeq</td>
      <td>4</td>
      <td>NaN</td>
      <td>210621</td>
      <td>3850000</td>
      <td>[/shared/ngs/illumina/tstarr/210629_D00300_1271_BHM55FBCX3/Unaligned/Project_tstarr/CGGexpr_lib2_bin4-1_S94_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210629_D00300_1271_BHM55FBCX3/Unaligned/Project_tstarr/CGGexpr_lib2_bin4-2_S95_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210629_D00300_1271_BHM55FBCX3/Unaligned/Project_tstarr/CGGexpr_lib2_bin4-3_S96_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>PSR_01_bin1</td>
      <td>PSR</td>
      <td>1</td>
      <td>1.0</td>
      <td>210623</td>
      <td>166552</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/PSR_lib1_s1_b1_S1_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/PSR_lib1_s1_b1_S1_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>PSR_01_bin2</td>
      <td>PSR</td>
      <td>2</td>
      <td>1.0</td>
      <td>210623</td>
      <td>1321552</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/PSR_lib1_s1_b2_S2_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/PSR_lib1_s1_b2_S2_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>PSR_01_bin3</td>
      <td>PSR</td>
      <td>3</td>
      <td>1.0</td>
      <td>210623</td>
      <td>2584631</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/PSR_lib1_s1_b3_S3_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/PSR_lib1_s1_b3_S3_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>PSR_01_bin4</td>
      <td>PSR</td>
      <td>4</td>
      <td>1.0</td>
      <td>210623</td>
      <td>1056008</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/PSR_lib1_s1_b4_S4_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/PSR_lib1_s1_b4_S4_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>PSR_02_bin1</td>
      <td>PSR</td>
      <td>1</td>
      <td>2.0</td>
      <td>210623</td>
      <td>226835</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/PSR_lib1_s2_b1_S5_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/PSR_lib1_s2_b1_S5_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>PSR_02_bin2</td>
      <td>PSR</td>
      <td>2</td>
      <td>2.0</td>
      <td>210623</td>
      <td>2109671</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/PSR_lib1_s2_b2_S6_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/PSR_lib1_s2_b2_S6_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>PSR_02_bin3</td>
      <td>PSR</td>
      <td>3</td>
      <td>2.0</td>
      <td>210623</td>
      <td>2116370</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/PSR_lib1_s2_b3_S7_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/PSR_lib1_s2_b3_S7_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>PSR_02_bin4</td>
      <td>PSR</td>
      <td>4</td>
      <td>2.0</td>
      <td>210623</td>
      <td>652024</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/PSR_lib1_s2_b4_S8_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/PSR_lib1_s2_b4_S8_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>PSR_03_bin1</td>
      <td>PSR</td>
      <td>1</td>
      <td>3.0</td>
      <td>210623</td>
      <td>638294</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/PSR_lib1_s3_b1_S9_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/PSR_lib1_s3_b1_S9_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>PSR_03_bin2</td>
      <td>PSR</td>
      <td>2</td>
      <td>3.0</td>
      <td>210623</td>
      <td>2990094</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/PSR_lib1_s3_b2_S10_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/PSR_lib1_s3_b2_S10_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>PSR_03_bin3</td>
      <td>PSR</td>
      <td>3</td>
      <td>3.0</td>
      <td>210623</td>
      <td>1205131</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/PSR_lib1_s3_b3_S11_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/PSR_lib1_s3_b3_S11_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>PSR_03_bin4</td>
      <td>PSR</td>
      <td>4</td>
      <td>3.0</td>
      <td>210623</td>
      <td>328641</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/PSR_lib1_s3_b4_S12_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/PSR_lib1_s3_b4_S12_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>PSR_04_bin1</td>
      <td>PSR</td>
      <td>1</td>
      <td>4.0</td>
      <td>210623</td>
      <td>1810068</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/PSR_lib1_s4_b1_S13_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/PSR_lib1_s4_b1_S13_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>PSR_04_bin2</td>
      <td>PSR</td>
      <td>2</td>
      <td>4.0</td>
      <td>210623</td>
      <td>2620843</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/PSR_lib1_s4_b2_S14_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/PSR_lib1_s4_b2_S14_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>PSR_04_bin3</td>
      <td>PSR</td>
      <td>3</td>
      <td>4.0</td>
      <td>210623</td>
      <td>530641</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/PSR_lib1_s4_b3_S15_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/PSR_lib1_s4_b3_S15_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>PSR_04_bin4</td>
      <td>PSR</td>
      <td>4</td>
      <td>4.0</td>
      <td>210623</td>
      <td>118642</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/PSR_lib1_s4_b4_S16_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/PSR_lib1_s4_b4_S16_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>PSR_05_bin1</td>
      <td>PSR</td>
      <td>1</td>
      <td>5.0</td>
      <td>210623</td>
      <td>4778551</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/PSR_lib1_s5_b1_S17_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/PSR_lib1_s5_b1_S17_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>PSR_05_bin2</td>
      <td>PSR</td>
      <td>2</td>
      <td>5.0</td>
      <td>210623</td>
      <td>353774</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/PSR_lib1_s5_b2_S18_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/PSR_lib1_s5_b2_S18_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>PSR_05_bin3</td>
      <td>PSR</td>
      <td>3</td>
      <td>5.0</td>
      <td>210623</td>
      <td>931</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/PSR_lib1_s5_b3_S19_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/PSR_lib1_s5_b3_S19_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>PSR_05_bin4</td>
      <td>PSR</td>
      <td>4</td>
      <td>5.0</td>
      <td>210623</td>
      <td>86</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/PSR_lib1_s5_b4_S20_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/PSR_lib1_s5_b4_S20_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>PSR_01_bin1</td>
      <td>PSR</td>
      <td>1</td>
      <td>1.0</td>
      <td>210623</td>
      <td>205712</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/PSR_lib2_s1_b1_S21_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/PSR_lib2_s1_b1_S21_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>PSR_01_bin2</td>
      <td>PSR</td>
      <td>2</td>
      <td>1.0</td>
      <td>210623</td>
      <td>1052179</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/PSR_lib2_s1_b2_S22_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/PSR_lib2_s1_b2_S22_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>PSR_01_bin3</td>
      <td>PSR</td>
      <td>3</td>
      <td>1.0</td>
      <td>210623</td>
      <td>2587244</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/PSR_lib2_s1_b3_S23_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/PSR_lib2_s1_b3_S23_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>PSR_01_bin4</td>
      <td>PSR</td>
      <td>4</td>
      <td>1.0</td>
      <td>210623</td>
      <td>1258113</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/PSR_lib2_s1_b4_S24_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/PSR_lib2_s1_b4_S24_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>PSR_02_bin1</td>
      <td>PSR</td>
      <td>1</td>
      <td>2.0</td>
      <td>210623</td>
      <td>92181</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/PSR_lib2_s2_b1_S25_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/PSR_lib2_s2_b1_S25_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>PSR_02_bin2</td>
      <td>PSR</td>
      <td>2</td>
      <td>2.0</td>
      <td>210623</td>
      <td>1595278</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/PSR_lib2_s2_b2_S26_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/PSR_lib2_s2_b2_S26_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>PSR_02_bin3</td>
      <td>PSR</td>
      <td>3</td>
      <td>2.0</td>
      <td>210623</td>
      <td>2631834</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/PSR_lib2_s2_b3_S27_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/PSR_lib2_s2_b3_S27_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>PSR_02_bin4</td>
      <td>PSR</td>
      <td>4</td>
      <td>2.0</td>
      <td>210623</td>
      <td>809181</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/PSR_lib2_s2_b4_S28_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/PSR_lib2_s2_b4_S28_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>PSR_03_bin1</td>
      <td>PSR</td>
      <td>1</td>
      <td>3.0</td>
      <td>210623</td>
      <td>212570</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/PSR_lib2_s3_b1_S29_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/PSR_lib2_s3_b1_S29_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>PSR_03_bin2</td>
      <td>PSR</td>
      <td>2</td>
      <td>3.0</td>
      <td>210623</td>
      <td>2734623</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/PSR_lib2_s3_b2_S30_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/PSR_lib2_s3_b2_S30_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>PSR_03_bin3</td>
      <td>PSR</td>
      <td>3</td>
      <td>3.0</td>
      <td>210623</td>
      <td>2024807</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/PSR_lib2_s3_b3_S31_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/PSR_lib2_s3_b3_S31_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>PSR_03_bin4</td>
      <td>PSR</td>
      <td>4</td>
      <td>3.0</td>
      <td>210623</td>
      <td>394498</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/PSR_lib2_s3_b4_S32_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/PSR_lib2_s3_b4_S32_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>PSR_04_bin1</td>
      <td>PSR</td>
      <td>1</td>
      <td>4.0</td>
      <td>210623</td>
      <td>522992</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/PSR_lib2_s4_b1_S33_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/PSR_lib2_s4_b1_S33_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>PSR_04_bin2</td>
      <td>PSR</td>
      <td>2</td>
      <td>4.0</td>
      <td>210623</td>
      <td>3416342</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/PSR_lib2_s4_b2_S34_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/PSR_lib2_s4_b2_S34_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>PSR_04_bin3</td>
      <td>PSR</td>
      <td>3</td>
      <td>4.0</td>
      <td>210623</td>
      <td>1023529</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/PSR_lib2_s4_b3_S35_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/PSR_lib2_s4_b3_S35_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>PSR_04_bin4</td>
      <td>PSR</td>
      <td>4</td>
      <td>4.0</td>
      <td>210623</td>
      <td>137633</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/PSR_lib2_s4_b4_S36_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/PSR_lib2_s4_b4_S36_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>PSR_05_bin1</td>
      <td>PSR</td>
      <td>1</td>
      <td>5.0</td>
      <td>210623</td>
      <td>4937652</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/PSR_lib2_s5_b1_S37_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/PSR_lib2_s5_b1_S37_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>PSR_05_bin2</td>
      <td>PSR</td>
      <td>2</td>
      <td>5.0</td>
      <td>210623</td>
      <td>852436</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/PSR_lib2_s5_b2_S38_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/PSR_lib2_s5_b2_S38_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>PSR_05_bin3</td>
      <td>PSR</td>
      <td>3</td>
      <td>5.0</td>
      <td>210623</td>
      <td>1912</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/PSR_lib2_s5_b3_S39_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/PSR_lib2_s5_b3_S39_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>PSR_05_bin4</td>
      <td>PSR</td>
      <td>4</td>
      <td>5.0</td>
      <td>210623</td>
      <td>123</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/PSR_lib2_s5_b4_S40_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/PSR_lib2_s5_b4_S40_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>TiteSeq_01_bin1</td>
      <td>TiteSeq</td>
      <td>1</td>
      <td>1.0</td>
      <td>210624</td>
      <td>9967</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/210624_s01-b1_S53_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/210624_s01-b1_S53_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>TiteSeq_01_bin2</td>
      <td>TiteSeq</td>
      <td>2</td>
      <td>1.0</td>
      <td>210624</td>
      <td>100694</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/210624_s01-b2_S54_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/210624_s01-b2_S54_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>TiteSeq_01_bin3</td>
      <td>TiteSeq</td>
      <td>3</td>
      <td>1.0</td>
      <td>210624</td>
      <td>910229</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/210624_s01-b3_S55_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/210624_s01-b3_S55_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>TiteSeq_01_bin4</td>
      <td>TiteSeq</td>
      <td>4</td>
      <td>1.0</td>
      <td>210624</td>
      <td>4152079</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/210624_s01-b4_S56_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/210624_s01-b4_S56_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>TiteSeq_02_bin1</td>
      <td>TiteSeq</td>
      <td>1</td>
      <td>2.0</td>
      <td>210624</td>
      <td>78772</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/210624_s02-b1_S57_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/210624_s02-b1_S57_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>TiteSeq_02_bin2</td>
      <td>TiteSeq</td>
      <td>2</td>
      <td>2.0</td>
      <td>210624</td>
      <td>363643</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/210624_s02-b2_S58_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/210624_s02-b2_S58_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>TiteSeq_02_bin3</td>
      <td>TiteSeq</td>
      <td>3</td>
      <td>2.0</td>
      <td>210624</td>
      <td>1099977</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/210624_s02-b3_S59_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/210624_s02-b3_S59_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>TiteSeq_02_bin4</td>
      <td>TiteSeq</td>
      <td>4</td>
      <td>2.0</td>
      <td>210624</td>
      <td>3597621</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/210624_s02-b4_S60_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/210624_s02-b4_S60_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>TiteSeq_03_bin1</td>
      <td>TiteSeq</td>
      <td>1</td>
      <td>3.0</td>
      <td>210624</td>
      <td>165484</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/210624_s03-b1_S61_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/210624_s03-b1_S61_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>TiteSeq_03_bin2</td>
      <td>TiteSeq</td>
      <td>2</td>
      <td>3.0</td>
      <td>210624</td>
      <td>563920</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/210624_s03-b2_S62_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/210624_s03-b2_S62_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>TiteSeq_03_bin3</td>
      <td>TiteSeq</td>
      <td>3</td>
      <td>3.0</td>
      <td>210624</td>
      <td>1517930</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/210624_s03-b3_S63_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/210624_s03-b3_S63_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>TiteSeq_03_bin4</td>
      <td>TiteSeq</td>
      <td>4</td>
      <td>3.0</td>
      <td>210624</td>
      <td>2863005</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/210624_s03-b4_S64_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/210624_s03-b4_S64_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>TiteSeq_04_bin1</td>
      <td>TiteSeq</td>
      <td>1</td>
      <td>4.0</td>
      <td>210624</td>
      <td>292375</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/210624_s04-b1_S65_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/210624_s04-b1_S65_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>TiteSeq_04_bin2</td>
      <td>TiteSeq</td>
      <td>2</td>
      <td>4.0</td>
      <td>210624</td>
      <td>715622</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/210624_s04-b2_S66_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/210624_s04-b2_S66_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>TiteSeq_04_bin3</td>
      <td>TiteSeq</td>
      <td>3</td>
      <td>4.0</td>
      <td>210624</td>
      <td>2461781</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/210624_s04-b3_S67_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/210624_s04-b3_S67_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>TiteSeq_04_bin4</td>
      <td>TiteSeq</td>
      <td>4</td>
      <td>4.0</td>
      <td>210624</td>
      <td>1745862</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/210624_s04-b4_S68_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/210624_s04-b4_S68_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>TiteSeq_05_bin1</td>
      <td>TiteSeq</td>
      <td>1</td>
      <td>5.0</td>
      <td>210624</td>
      <td>410552</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/210624_s05-b1_S69_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/210624_s05-b1_S69_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>TiteSeq_05_bin2</td>
      <td>TiteSeq</td>
      <td>2</td>
      <td>5.0</td>
      <td>210624</td>
      <td>1513776</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/210624_s05-b2_S70_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/210624_s05-b2_S70_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>TiteSeq_05_bin3</td>
      <td>TiteSeq</td>
      <td>3</td>
      <td>5.0</td>
      <td>210624</td>
      <td>3204858</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/210624_s05-b3_S71_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/210624_s05-b3_S71_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>TiteSeq_05_bin4</td>
      <td>TiteSeq</td>
      <td>4</td>
      <td>5.0</td>
      <td>210624</td>
      <td>48951</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/210624_s05-b4_S72_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/210624_s05-b4_S72_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>TiteSeq_06_bin1</td>
      <td>TiteSeq</td>
      <td>1</td>
      <td>6.0</td>
      <td>210624</td>
      <td>1352339</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/210624_s06-b1_S73_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/210624_s06-b1_S73_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>TiteSeq_06_bin2</td>
      <td>TiteSeq</td>
      <td>2</td>
      <td>6.0</td>
      <td>210624</td>
      <td>3598153</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/210624_s06-b2_S74_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/210624_s06-b2_S74_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>TiteSeq_06_bin3</td>
      <td>TiteSeq</td>
      <td>3</td>
      <td>6.0</td>
      <td>210624</td>
      <td>98238</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/210624_s06-b3_S75_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/210624_s06-b3_S75_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>TiteSeq_06_bin4</td>
      <td>TiteSeq</td>
      <td>4</td>
      <td>6.0</td>
      <td>210624</td>
      <td>17</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/210624_s06-b4_S76_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/210624_s06-b4_S76_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>TiteSeq_07_bin1</td>
      <td>TiteSeq</td>
      <td>1</td>
      <td>7.0</td>
      <td>210624</td>
      <td>4214830</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/210624_s07-b1_S77_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/210624_s07-b1_S77_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>TiteSeq_07_bin2</td>
      <td>TiteSeq</td>
      <td>2</td>
      <td>7.0</td>
      <td>210624</td>
      <td>814937</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/210624_s07-b2_S78_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/210624_s07-b2_S78_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>TiteSeq_07_bin3</td>
      <td>TiteSeq</td>
      <td>3</td>
      <td>7.0</td>
      <td>210624</td>
      <td>321</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/210624_s07-b3_S79_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/210624_s07-b3_S79_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>TiteSeq_07_bin4</td>
      <td>TiteSeq</td>
      <td>4</td>
      <td>7.0</td>
      <td>210624</td>
      <td>4</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/210624_s07-b4_S80_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/210624_s07-b4_S80_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>TiteSeq_08_bin1</td>
      <td>TiteSeq</td>
      <td>1</td>
      <td>8.0</td>
      <td>210624</td>
      <td>4879499</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/210624_s08-b1_S81_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/210624_s08-b1_S81_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>TiteSeq_08_bin2</td>
      <td>TiteSeq</td>
      <td>2</td>
      <td>8.0</td>
      <td>210624</td>
      <td>166605</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/210624_s08-b2_S82_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/210624_s08-b2_S82_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>TiteSeq_08_bin3</td>
      <td>TiteSeq</td>
      <td>3</td>
      <td>8.0</td>
      <td>210624</td>
      <td>110</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/210624_s08-b3_S83_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/210624_s08-b3_S83_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>TiteSeq_08_bin4</td>
      <td>TiteSeq</td>
      <td>4</td>
      <td>8.0</td>
      <td>210624</td>
      <td>34</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/210624_s08-b4_S84_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/210624_s08-b4_S84_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>TiteSeq_09_bin1</td>
      <td>TiteSeq</td>
      <td>1</td>
      <td>9.0</td>
      <td>210624</td>
      <td>4994518</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/210624_s09-b1_S85_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/210624_s09-b1_S85_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>TiteSeq_09_bin2</td>
      <td>TiteSeq</td>
      <td>2</td>
      <td>9.0</td>
      <td>210624</td>
      <td>97766</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/210624_s09-b2_S86_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/210624_s09-b2_S86_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>TiteSeq_09_bin3</td>
      <td>TiteSeq</td>
      <td>3</td>
      <td>9.0</td>
      <td>210624</td>
      <td>156</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/210624_s09-b3_S87_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/210624_s09-b3_S87_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>TiteSeq_09_bin4</td>
      <td>TiteSeq</td>
      <td>4</td>
      <td>9.0</td>
      <td>210624</td>
      <td>57</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/210624_s09-b4_S88_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/210624_s09-b4_S88_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>TiteSeq_01_bin1</td>
      <td>TiteSeq</td>
      <td>1</td>
      <td>1.0</td>
      <td>210624</td>
      <td>5023</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/210624_s10-b1_S89_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/210624_s10-b1_S89_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>TiteSeq_01_bin2</td>
      <td>TiteSeq</td>
      <td>2</td>
      <td>1.0</td>
      <td>210624</td>
      <td>85666</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/210624_s10-b2_S90_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/210624_s10-b2_S90_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>TiteSeq_01_bin3</td>
      <td>TiteSeq</td>
      <td>3</td>
      <td>1.0</td>
      <td>210624</td>
      <td>910355</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/210624_s10-b3_S91_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/210624_s10-b3_S91_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>TiteSeq_01_bin4</td>
      <td>TiteSeq</td>
      <td>4</td>
      <td>1.0</td>
      <td>210624</td>
      <td>4087566</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/210624_s10-b4_S92_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/210624_s10-b4_S92_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>TiteSeq_02_bin1</td>
      <td>TiteSeq</td>
      <td>1</td>
      <td>2.0</td>
      <td>210624</td>
      <td>64354</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/210624_s11-b1_S93_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/210624_s11-b1_S93_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>TiteSeq_02_bin2</td>
      <td>TiteSeq</td>
      <td>2</td>
      <td>2.0</td>
      <td>210624</td>
      <td>377819</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/210624_s11-b2_S94_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/210624_s11-b2_S94_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>TiteSeq_02_bin3</td>
      <td>TiteSeq</td>
      <td>3</td>
      <td>2.0</td>
      <td>210624</td>
      <td>1149394</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/210624_s11-b3_S95_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/210624_s11-b3_S95_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>TiteSeq_02_bin4</td>
      <td>TiteSeq</td>
      <td>4</td>
      <td>2.0</td>
      <td>210624</td>
      <td>3595640</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/210624_s11-b4_S96_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/210624_s11-b4_S96_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>TiteSeq_03_bin1</td>
      <td>TiteSeq</td>
      <td>1</td>
      <td>3.0</td>
      <td>210624</td>
      <td>146993</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/210624_s12-b1_S97_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/210624_s12-b1_S97_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>TiteSeq_03_bin2</td>
      <td>TiteSeq</td>
      <td>2</td>
      <td>3.0</td>
      <td>210624</td>
      <td>585863</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/210624_s12-b2_S98_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/210624_s12-b2_S98_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>TiteSeq_03_bin3</td>
      <td>TiteSeq</td>
      <td>3</td>
      <td>3.0</td>
      <td>210624</td>
      <td>1546007</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/210624_s12-b3_S99_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/210624_s12-b3_S99_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>TiteSeq_03_bin4</td>
      <td>TiteSeq</td>
      <td>4</td>
      <td>3.0</td>
      <td>210624</td>
      <td>2888203</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/210624_s12-b4_S100_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/210624_s12-b4_S100_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>TiteSeq_04_bin1</td>
      <td>TiteSeq</td>
      <td>1</td>
      <td>4.0</td>
      <td>210624</td>
      <td>229759</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/210624_s13-b1_S101_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/210624_s13-b1_S101_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>TiteSeq_04_bin2</td>
      <td>TiteSeq</td>
      <td>2</td>
      <td>4.0</td>
      <td>210624</td>
      <td>754199</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/210624_s13-b2_S102_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/210624_s13-b2_S102_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>TiteSeq_04_bin3</td>
      <td>TiteSeq</td>
      <td>3</td>
      <td>4.0</td>
      <td>210624</td>
      <td>2383948</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/210624_s13-b3_S103_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/210624_s13-b3_S103_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>TiteSeq_04_bin4</td>
      <td>TiteSeq</td>
      <td>4</td>
      <td>4.0</td>
      <td>210624</td>
      <td>1851348</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/210624_s13-b4_S104_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/210624_s13-b4_S104_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>TiteSeq_05_bin1</td>
      <td>TiteSeq</td>
      <td>1</td>
      <td>5.0</td>
      <td>210624</td>
      <td>330918</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/210624_s14-b1_S105_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/210624_s14-b1_S105_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>TiteSeq_05_bin2</td>
      <td>TiteSeq</td>
      <td>2</td>
      <td>5.0</td>
      <td>210624</td>
      <td>1498361</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/210624_s14-b2_S106_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/210624_s14-b2_S106_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>TiteSeq_05_bin3</td>
      <td>TiteSeq</td>
      <td>3</td>
      <td>5.0</td>
      <td>210624</td>
      <td>3161208</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/210624_s14-b3_S107_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/210624_s14-b3_S107_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>TiteSeq_05_bin4</td>
      <td>TiteSeq</td>
      <td>4</td>
      <td>5.0</td>
      <td>210624</td>
      <td>54642</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/210624_s14-b4_S108_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/210624_s14-b4_S108_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>TiteSeq_06_bin1</td>
      <td>TiteSeq</td>
      <td>1</td>
      <td>6.0</td>
      <td>210624</td>
      <td>1260056</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/210624_s15-b1_S109_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/210624_s15-b1_S109_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>TiteSeq_06_bin2</td>
      <td>TiteSeq</td>
      <td>2</td>
      <td>6.0</td>
      <td>210624</td>
      <td>3701958</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/210624_s15-b2_S110_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/210624_s15-b2_S110_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>TiteSeq_06_bin3</td>
      <td>TiteSeq</td>
      <td>3</td>
      <td>6.0</td>
      <td>210624</td>
      <td>90277</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/210624_s15-b3_S111_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/210624_s15-b3_S111_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>TiteSeq_06_bin4</td>
      <td>TiteSeq</td>
      <td>4</td>
      <td>6.0</td>
      <td>210624</td>
      <td>19</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/210624_s15-b4_S112_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/210624_s15-b4_S112_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>TiteSeq_07_bin1</td>
      <td>TiteSeq</td>
      <td>1</td>
      <td>7.0</td>
      <td>210624</td>
      <td>4182689</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/210624_s16-b1_S41_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/210624_s16-b1_S41_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>TiteSeq_07_bin2</td>
      <td>TiteSeq</td>
      <td>2</td>
      <td>7.0</td>
      <td>210624</td>
      <td>830786</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/210624_s16-b2_S42_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/210624_s16-b2_S42_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>TiteSeq_07_bin3</td>
      <td>TiteSeq</td>
      <td>3</td>
      <td>7.0</td>
      <td>210624</td>
      <td>273</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/210624_s26-b3_S43_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/210624_s26-b3_S43_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>TiteSeq_07_bin4</td>
      <td>TiteSeq</td>
      <td>4</td>
      <td>7.0</td>
      <td>210624</td>
      <td>20</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/210624_s16-b4_S44_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/210624_s16-b4_S44_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>TiteSeq_08_bin1</td>
      <td>TiteSeq</td>
      <td>1</td>
      <td>8.0</td>
      <td>210624</td>
      <td>4893728</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/210624_s17-b1_S45_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/210624_s17-b1_S45_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>TiteSeq_08_bin2</td>
      <td>TiteSeq</td>
      <td>2</td>
      <td>8.0</td>
      <td>210624</td>
      <td>154510</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/210624_s17-b2_S46_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/210624_s17-b2_S46_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>TiteSeq_08_bin3</td>
      <td>TiteSeq</td>
      <td>3</td>
      <td>8.0</td>
      <td>210624</td>
      <td>83</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/210624_s17-b3_S47_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/210624_s17-b3_S47_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>TiteSeq_08_bin4</td>
      <td>TiteSeq</td>
      <td>4</td>
      <td>8.0</td>
      <td>210624</td>
      <td>23</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/210624_s17-b4_S48_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/210624_s17-b4_S48_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>TiteSeq_09_bin1</td>
      <td>TiteSeq</td>
      <td>1</td>
      <td>9.0</td>
      <td>210624</td>
      <td>5016672</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/210624_s18-b1_S49_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/210624_s18-b1_S49_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>TiteSeq_09_bin2</td>
      <td>TiteSeq</td>
      <td>2</td>
      <td>9.0</td>
      <td>210624</td>
      <td>86684</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/210624_s18-b2_S50_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/210624_s18-b2_S50_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>TiteSeq_09_bin3</td>
      <td>TiteSeq</td>
      <td>3</td>
      <td>9.0</td>
      <td>210624</td>
      <td>50</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/210624_s18-b3_S51_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/210624_s18-b3_S51_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>TiteSeq_09_bin4</td>
      <td>TiteSeq</td>
      <td>4</td>
      <td>9.0</td>
      <td>210624</td>
      <td>3</td>
      <td>[/shared/ngs/illumina/tstarr/210728_D00300_1288_AHMGMTBCX3/Unaligned/Project_tstarr/210624_s18-b4_S52_R1_001.fastq.gz, /shared/ngs/illumina/tstarr/210701_D00300_1272_BHM532BCX3/Unaligned/Project_tstarr/210624_s18-b4_S52_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>TuGG_TiteSeq_01_bin1</td>
      <td>TuGG_TiteSeq</td>
      <td>1</td>
      <td>1.0</td>
      <td>210624</td>
      <td>95588</td>
      <td>[/shared/ngs/illumina/tstarr/220428_VH00699_111_AAATNWKM5/Unaligned/Project_tstarr/220420_s01-b1_S1_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>TuGG_TiteSeq_01_bin2</td>
      <td>TuGG_TiteSeq</td>
      <td>2</td>
      <td>1.0</td>
      <td>210624</td>
      <td>4168050</td>
      <td>[/shared/ngs/illumina/tstarr/220428_VH00699_111_AAATNWKM5/Unaligned/Project_tstarr/220420_s01-b2_S2_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>TuGG_TiteSeq_01_bin3</td>
      <td>TuGG_TiteSeq</td>
      <td>3</td>
      <td>1.0</td>
      <td>210624</td>
      <td>4184690</td>
      <td>[/shared/ngs/illumina/tstarr/220428_VH00699_111_AAATNWKM5/Unaligned/Project_tstarr/220420_s01-b3_S3_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>TuGG_TiteSeq_01_bin4</td>
      <td>TuGG_TiteSeq</td>
      <td>4</td>
      <td>1.0</td>
      <td>210624</td>
      <td>1183718</td>
      <td>[/shared/ngs/illumina/tstarr/220428_VH00699_111_AAATNWKM5/Unaligned/Project_tstarr/220420_s01-b4_S4_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>TuGG_TiteSeq_02_bin1</td>
      <td>TuGG_TiteSeq</td>
      <td>1</td>
      <td>2.0</td>
      <td>210624</td>
      <td>3314375</td>
      <td>[/shared/ngs/illumina/tstarr/220428_VH00699_111_AAATNWKM5/Unaligned/Project_tstarr/220420_s02-b1_S5_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>TuGG_TiteSeq_02_bin2</td>
      <td>TuGG_TiteSeq</td>
      <td>2</td>
      <td>2.0</td>
      <td>210624</td>
      <td>5019870</td>
      <td>[/shared/ngs/illumina/tstarr/220428_VH00699_111_AAATNWKM5/Unaligned/Project_tstarr/220420_s02-b2_S6_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>TuGG_TiteSeq_02_bin3</td>
      <td>TuGG_TiteSeq</td>
      <td>3</td>
      <td>2.0</td>
      <td>210624</td>
      <td>648425</td>
      <td>[/shared/ngs/illumina/tstarr/220428_VH00699_111_AAATNWKM5/Unaligned/Project_tstarr/220420_s02-b3_S7_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>TuGG_TiteSeq_02_bin4</td>
      <td>TuGG_TiteSeq</td>
      <td>4</td>
      <td>2.0</td>
      <td>210624</td>
      <td>157814</td>
      <td>[/shared/ngs/illumina/tstarr/220428_VH00699_111_AAATNWKM5/Unaligned/Project_tstarr/220420_s02-b4_S8_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>TuGG_TiteSeq_03_bin1</td>
      <td>TuGG_TiteSeq</td>
      <td>1</td>
      <td>3.0</td>
      <td>210624</td>
      <td>3125305</td>
      <td>[/shared/ngs/illumina/tstarr/220428_VH00699_111_AAATNWKM5/Unaligned/Project_tstarr/220420_s03-b1_S9_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>TuGG_TiteSeq_03_bin2</td>
      <td>TuGG_TiteSeq</td>
      <td>2</td>
      <td>3.0</td>
      <td>210624</td>
      <td>5256079</td>
      <td>[/shared/ngs/illumina/tstarr/220428_VH00699_111_AAATNWKM5/Unaligned/Project_tstarr/220420_s03-b2_S10_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>TuGG_TiteSeq_03_bin3</td>
      <td>TuGG_TiteSeq</td>
      <td>3</td>
      <td>3.0</td>
      <td>210624</td>
      <td>953327</td>
      <td>[/shared/ngs/illumina/tstarr/220428_VH00699_111_AAATNWKM5/Unaligned/Project_tstarr/220420_s03-b3_S11_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>TuGG_TiteSeq_03_bin4</td>
      <td>TuGG_TiteSeq</td>
      <td>4</td>
      <td>3.0</td>
      <td>210624</td>
      <td>120068</td>
      <td>[/shared/ngs/illumina/tstarr/220428_VH00699_111_AAATNWKM5/Unaligned/Project_tstarr/220420_s03-b4_S12_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>TuGG_TiteSeq_04_bin1</td>
      <td>TuGG_TiteSeq</td>
      <td>1</td>
      <td>4.0</td>
      <td>210624</td>
      <td>7469770</td>
      <td>[/shared/ngs/illumina/tstarr/220428_VH00699_111_AAATNWKM5/Unaligned/Project_tstarr/220420_s04-b1_S13_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>TuGG_TiteSeq_04_bin2</td>
      <td>TuGG_TiteSeq</td>
      <td>2</td>
      <td>4.0</td>
      <td>210624</td>
      <td>760060</td>
      <td>[/shared/ngs/illumina/tstarr/220428_VH00699_111_AAATNWKM5/Unaligned/Project_tstarr/220420_s04-b2_S14_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>TuGG_TiteSeq_04_bin3</td>
      <td>TuGG_TiteSeq</td>
      <td>3</td>
      <td>4.0</td>
      <td>210624</td>
      <td>36335</td>
      <td>[/shared/ngs/illumina/tstarr/220428_VH00699_111_AAATNWKM5/Unaligned/Project_tstarr/220420_s04-b3_S15_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>TuGG_TiteSeq_04_bin4</td>
      <td>TuGG_TiteSeq</td>
      <td>4</td>
      <td>4.0</td>
      <td>210624</td>
      <td>17293</td>
      <td>[/shared/ngs/illumina/tstarr/220428_VH00699_111_AAATNWKM5/Unaligned/Project_tstarr/220420_s04-b4_S16_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>TuGG_TiteSeq_05_bin1</td>
      <td>TuGG_TiteSeq</td>
      <td>1</td>
      <td>5.0</td>
      <td>210624</td>
      <td>6294823</td>
      <td>[/shared/ngs/illumina/tstarr/220428_VH00699_111_AAATNWKM5/Unaligned/Project_tstarr/220420_s05-b1_S17_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>TuGG_TiteSeq_05_bin2</td>
      <td>TuGG_TiteSeq</td>
      <td>2</td>
      <td>5.0</td>
      <td>210624</td>
      <td>151581</td>
      <td>[/shared/ngs/illumina/tstarr/220428_VH00699_111_AAATNWKM5/Unaligned/Project_tstarr/220420_s05-b2_S18_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>TuGG_TiteSeq_05_bin3</td>
      <td>TuGG_TiteSeq</td>
      <td>3</td>
      <td>5.0</td>
      <td>210624</td>
      <td>3155</td>
      <td>[/shared/ngs/illumina/tstarr/220428_VH00699_111_AAATNWKM5/Unaligned/Project_tstarr/220420_s05-b3_S19_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>TuGG_TiteSeq_05_bin4</td>
      <td>TuGG_TiteSeq</td>
      <td>4</td>
      <td>5.0</td>
      <td>210624</td>
      <td>1286</td>
      <td>[/shared/ngs/illumina/tstarr/220428_VH00699_111_AAATNWKM5/Unaligned/Project_tstarr/220420_s05-b4_S20_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>TuGG_TiteSeq_06_bin1</td>
      <td>TuGG_TiteSeq</td>
      <td>1</td>
      <td>6.0</td>
      <td>210624</td>
      <td>8590833</td>
      <td>[/shared/ngs/illumina/tstarr/220428_VH00699_111_AAATNWKM5/Unaligned/Project_tstarr/220420_s06-b1_S21_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>TuGG_TiteSeq_06_bin2</td>
      <td>TuGG_TiteSeq</td>
      <td>2</td>
      <td>6.0</td>
      <td>210624</td>
      <td>180075</td>
      <td>[/shared/ngs/illumina/tstarr/220428_VH00699_111_AAATNWKM5/Unaligned/Project_tstarr/220420_s06-b2_S22_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>TuGG_TiteSeq_06_bin3</td>
      <td>TuGG_TiteSeq</td>
      <td>3</td>
      <td>6.0</td>
      <td>210624</td>
      <td>2343</td>
      <td>[/shared/ngs/illumina/tstarr/220428_VH00699_111_AAATNWKM5/Unaligned/Project_tstarr/220420_s06-b3_S23_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>TuGG_TiteSeq_06_bin4</td>
      <td>TuGG_TiteSeq</td>
      <td>4</td>
      <td>6.0</td>
      <td>210624</td>
      <td>912</td>
      <td>[/shared/ngs/illumina/tstarr/220428_VH00699_111_AAATNWKM5/Unaligned/Project_tstarr/220420_s06-b4_S24_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>TuGG_TiteSeq_07_bin1</td>
      <td>TuGG_TiteSeq</td>
      <td>1</td>
      <td>7.0</td>
      <td>210624</td>
      <td>8613053</td>
      <td>[/shared/ngs/illumina/tstarr/220428_VH00699_111_AAATNWKM5/Unaligned/Project_tstarr/220420_s07-b1_S25_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>TuGG_TiteSeq_07_bin2</td>
      <td>TuGG_TiteSeq</td>
      <td>2</td>
      <td>7.0</td>
      <td>210624</td>
      <td>186532</td>
      <td>[/shared/ngs/illumina/tstarr/220428_VH00699_111_AAATNWKM5/Unaligned/Project_tstarr/220420_s07-b2_S26_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>TuGG_TiteSeq_07_bin3</td>
      <td>TuGG_TiteSeq</td>
      <td>3</td>
      <td>7.0</td>
      <td>210624</td>
      <td>690</td>
      <td>[/shared/ngs/illumina/tstarr/220428_VH00699_111_AAATNWKM5/Unaligned/Project_tstarr/220420_s07-b3_S27_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>TuGG_TiteSeq_07_bin4</td>
      <td>TuGG_TiteSeq</td>
      <td>4</td>
      <td>7.0</td>
      <td>210624</td>
      <td>684</td>
      <td>[/shared/ngs/illumina/tstarr/220428_VH00699_111_AAATNWKM5/Unaligned/Project_tstarr/220420_s07-b4_S28_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>TuGG_TiteSeq_09_bin1</td>
      <td>TuGG_TiteSeq</td>
      <td>1</td>
      <td>9.0</td>
      <td>210624</td>
      <td>8975223</td>
      <td>[/shared/ngs/illumina/tstarr/220428_VH00699_111_AAATNWKM5/Unaligned/Project_tstarr/220420_s09-b1_S29_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>TuGG_TiteSeq_09_bin2</td>
      <td>TuGG_TiteSeq</td>
      <td>2</td>
      <td>9.0</td>
      <td>210624</td>
      <td>268793</td>
      <td>[/shared/ngs/illumina/tstarr/220428_VH00699_111_AAATNWKM5/Unaligned/Project_tstarr/220420_s09-b2_S30_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>TuGG_TiteSeq_09_bin3</td>
      <td>TuGG_TiteSeq</td>
      <td>3</td>
      <td>9.0</td>
      <td>210624</td>
      <td>697</td>
      <td>[/shared/ngs/illumina/tstarr/220428_VH00699_111_AAATNWKM5/Unaligned/Project_tstarr/220420_s09-b3_S31_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>TuGG_TiteSeq_09_bin4</td>
      <td>TuGG_TiteSeq</td>
      <td>4</td>
      <td>9.0</td>
      <td>210624</td>
      <td>727</td>
      <td>[/shared/ngs/illumina/tstarr/220428_VH00699_111_AAATNWKM5/Unaligned/Project_tstarr/220420_s09-b4_S32_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>TuGG_TiteSeq_01_bin1</td>
      <td>TuGG_TiteSeq</td>
      <td>1</td>
      <td>1.0</td>
      <td>210624</td>
      <td>115152</td>
      <td>[/shared/ngs/illumina/tstarr/220428_VH00699_111_AAATNWKM5/Unaligned/Project_tstarr/220420_s10-b1_S33_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>TuGG_TiteSeq_01_bin2</td>
      <td>TuGG_TiteSeq</td>
      <td>2</td>
      <td>1.0</td>
      <td>210624</td>
      <td>5524687</td>
      <td>[/shared/ngs/illumina/tstarr/220428_VH00699_111_AAATNWKM5/Unaligned/Project_tstarr/220420_s10-b2_S34_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>TuGG_TiteSeq_01_bin3</td>
      <td>TuGG_TiteSeq</td>
      <td>3</td>
      <td>1.0</td>
      <td>210624</td>
      <td>3023595</td>
      <td>[/shared/ngs/illumina/tstarr/220428_VH00699_111_AAATNWKM5/Unaligned/Project_tstarr/220420_s10-b3_S35_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>TuGG_TiteSeq_01_bin4</td>
      <td>TuGG_TiteSeq</td>
      <td>4</td>
      <td>1.0</td>
      <td>210624</td>
      <td>669346</td>
      <td>[/shared/ngs/illumina/tstarr/220428_VH00699_111_AAATNWKM5/Unaligned/Project_tstarr/220420_s10-b4_S36_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>TuGG_TiteSeq_02_bin1</td>
      <td>TuGG_TiteSeq</td>
      <td>1</td>
      <td>2.0</td>
      <td>210624</td>
      <td>3655690</td>
      <td>[/shared/ngs/illumina/tstarr/220428_VH00699_111_AAATNWKM5/Unaligned/Project_tstarr/220420_s11-b1_S37_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>TuGG_TiteSeq_02_bin2</td>
      <td>TuGG_TiteSeq</td>
      <td>2</td>
      <td>2.0</td>
      <td>210624</td>
      <td>5318467</td>
      <td>[/shared/ngs/illumina/tstarr/220428_VH00699_111_AAATNWKM5/Unaligned/Project_tstarr/220420_s11-b2_S38_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>TuGG_TiteSeq_02_bin3</td>
      <td>TuGG_TiteSeq</td>
      <td>3</td>
      <td>2.0</td>
      <td>210624</td>
      <td>484880</td>
      <td>[/shared/ngs/illumina/tstarr/220428_VH00699_111_AAATNWKM5/Unaligned/Project_tstarr/220420_s11-b3_S39_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>TuGG_TiteSeq_02_bin4</td>
      <td>TuGG_TiteSeq</td>
      <td>4</td>
      <td>2.0</td>
      <td>210624</td>
      <td>136565</td>
      <td>[/shared/ngs/illumina/tstarr/220428_VH00699_111_AAATNWKM5/Unaligned/Project_tstarr/220420_s11-b4_S40_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>TuGG_TiteSeq_03_bin1</td>
      <td>TuGG_TiteSeq</td>
      <td>1</td>
      <td>3.0</td>
      <td>210624</td>
      <td>2973732</td>
      <td>[/shared/ngs/illumina/tstarr/220428_VH00699_111_AAATNWKM5/Unaligned/Project_tstarr/220420_s12-b1_S41_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>TuGG_TiteSeq_03_bin2</td>
      <td>TuGG_TiteSeq</td>
      <td>2</td>
      <td>3.0</td>
      <td>210624</td>
      <td>5603928</td>
      <td>[/shared/ngs/illumina/tstarr/220428_VH00699_111_AAATNWKM5/Unaligned/Project_tstarr/220420_s12-b2_S42_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>TuGG_TiteSeq_03_bin3</td>
      <td>TuGG_TiteSeq</td>
      <td>3</td>
      <td>3.0</td>
      <td>210624</td>
      <td>830237</td>
      <td>[/shared/ngs/illumina/tstarr/220428_VH00699_111_AAATNWKM5/Unaligned/Project_tstarr/220420_s12-b3_S43_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>TuGG_TiteSeq_03_bin4</td>
      <td>TuGG_TiteSeq</td>
      <td>4</td>
      <td>3.0</td>
      <td>210624</td>
      <td>77414</td>
      <td>[/shared/ngs/illumina/tstarr/220428_VH00699_111_AAATNWKM5/Unaligned/Project_tstarr/220420_s12-b4_S44_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>TuGG_TiteSeq_04_bin1</td>
      <td>TuGG_TiteSeq</td>
      <td>1</td>
      <td>4.0</td>
      <td>210624</td>
      <td>9411050</td>
      <td>[/shared/ngs/illumina/tstarr/220428_VH00699_111_AAATNWKM5/Unaligned/Project_tstarr/220420_s13-b1_S45_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>TuGG_TiteSeq_04_bin2</td>
      <td>TuGG_TiteSeq</td>
      <td>2</td>
      <td>4.0</td>
      <td>210624</td>
      <td>868830</td>
      <td>[/shared/ngs/illumina/tstarr/220428_VH00699_111_AAATNWKM5/Unaligned/Project_tstarr/220420_s13-b2_S46_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>TuGG_TiteSeq_04_bin3</td>
      <td>TuGG_TiteSeq</td>
      <td>3</td>
      <td>4.0</td>
      <td>210624</td>
      <td>24669</td>
      <td>[/shared/ngs/illumina/tstarr/220428_VH00699_111_AAATNWKM5/Unaligned/Project_tstarr/220420_s13-b3_S47_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>TuGG_TiteSeq_04_bin4</td>
      <td>TuGG_TiteSeq</td>
      <td>4</td>
      <td>4.0</td>
      <td>210624</td>
      <td>15997</td>
      <td>[/shared/ngs/illumina/tstarr/220428_VH00699_111_AAATNWKM5/Unaligned/Project_tstarr/220420_s13-b4_S48_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>TuGG_TiteSeq_05_bin1</td>
      <td>TuGG_TiteSeq</td>
      <td>1</td>
      <td>5.0</td>
      <td>210624</td>
      <td>9099859</td>
      <td>[/shared/ngs/illumina/tstarr/220428_VH00699_111_AAATNWKM5/Unaligned/Project_tstarr/220420_s14-b1_S49_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>TuGG_TiteSeq_05_bin2</td>
      <td>TuGG_TiteSeq</td>
      <td>2</td>
      <td>5.0</td>
      <td>210624</td>
      <td>230472</td>
      <td>[/shared/ngs/illumina/tstarr/220428_VH00699_111_AAATNWKM5/Unaligned/Project_tstarr/220420_s14-b2_S50_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>TuGG_TiteSeq_05_bin3</td>
      <td>TuGG_TiteSeq</td>
      <td>3</td>
      <td>5.0</td>
      <td>210624</td>
      <td>12450</td>
      <td>[/shared/ngs/illumina/tstarr/220428_VH00699_111_AAATNWKM5/Unaligned/Project_tstarr/220420_s14-b3_S51_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>TuGG_TiteSeq_05_bin4</td>
      <td>TuGG_TiteSeq</td>
      <td>4</td>
      <td>5.0</td>
      <td>210624</td>
      <td>5709</td>
      <td>[/shared/ngs/illumina/tstarr/220428_VH00699_111_AAATNWKM5/Unaligned/Project_tstarr/220420_s14-b4_S52_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>TuGG_TiteSeq_06_bin1</td>
      <td>TuGG_TiteSeq</td>
      <td>1</td>
      <td>6.0</td>
      <td>210624</td>
      <td>9243226</td>
      <td>[/shared/ngs/illumina/tstarr/220428_VH00699_111_AAATNWKM5/Unaligned/Project_tstarr/220420_s15-b1_S53_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>TuGG_TiteSeq_06_bin2</td>
      <td>TuGG_TiteSeq</td>
      <td>2</td>
      <td>6.0</td>
      <td>210624</td>
      <td>207411</td>
      <td>[/shared/ngs/illumina/tstarr/220428_VH00699_111_AAATNWKM5/Unaligned/Project_tstarr/220420_s15-b2_S54_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>TuGG_TiteSeq_06_bin3</td>
      <td>TuGG_TiteSeq</td>
      <td>3</td>
      <td>6.0</td>
      <td>210624</td>
      <td>4539</td>
      <td>[/shared/ngs/illumina/tstarr/220428_VH00699_111_AAATNWKM5/Unaligned/Project_tstarr/220420_s15-b3_S55_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>TuGG_TiteSeq_06_bin4</td>
      <td>TuGG_TiteSeq</td>
      <td>4</td>
      <td>6.0</td>
      <td>210624</td>
      <td>1069</td>
      <td>[/shared/ngs/illumina/tstarr/220428_VH00699_111_AAATNWKM5/Unaligned/Project_tstarr/220420_s15-b4_S56_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>TuGG_TiteSeq_07_bin1</td>
      <td>TuGG_TiteSeq</td>
      <td>1</td>
      <td>7.0</td>
      <td>210624</td>
      <td>9230106</td>
      <td>[/shared/ngs/illumina/tstarr/220428_VH00699_111_AAATNWKM5/Unaligned/Project_tstarr/220420_s16-b1_S57_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>TuGG_TiteSeq_07_bin2</td>
      <td>TuGG_TiteSeq</td>
      <td>2</td>
      <td>7.0</td>
      <td>210624</td>
      <td>177944</td>
      <td>[/shared/ngs/illumina/tstarr/220428_VH00699_111_AAATNWKM5/Unaligned/Project_tstarr/220420_s16-b2_S58_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>TuGG_TiteSeq_07_bin3</td>
      <td>TuGG_TiteSeq</td>
      <td>3</td>
      <td>7.0</td>
      <td>210624</td>
      <td>502</td>
      <td>[/shared/ngs/illumina/tstarr/220428_VH00699_111_AAATNWKM5/Unaligned/Project_tstarr/220420_s16-b3_S59_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>TuGG_TiteSeq_07_bin4</td>
      <td>TuGG_TiteSeq</td>
      <td>4</td>
      <td>7.0</td>
      <td>210624</td>
      <td>684</td>
      <td>[/shared/ngs/illumina/tstarr/220428_VH00699_111_AAATNWKM5/Unaligned/Project_tstarr/220420_s16-b4_S60_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>TuGG_TiteSeq_09_bin1</td>
      <td>TuGG_TiteSeq</td>
      <td>1</td>
      <td>9.0</td>
      <td>210624</td>
      <td>9196329</td>
      <td>[/shared/ngs/illumina/tstarr/220428_VH00699_111_AAATNWKM5/Unaligned/Project_tstarr/220420_s18-b1_S61_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>TuGG_TiteSeq_09_bin2</td>
      <td>TuGG_TiteSeq</td>
      <td>2</td>
      <td>9.0</td>
      <td>210624</td>
      <td>279088</td>
      <td>[/shared/ngs/illumina/tstarr/220428_VH00699_111_AAATNWKM5/Unaligned/Project_tstarr/220420_s18-b2_S62_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>TuGG_TiteSeq_09_bin3</td>
      <td>TuGG_TiteSeq</td>
      <td>3</td>
      <td>9.0</td>
      <td>210624</td>
      <td>618</td>
      <td>[/shared/ngs/illumina/tstarr/220428_VH00699_111_AAATNWKM5/Unaligned/Project_tstarr/220420_s18-b3_S63_R1_001.fastq.gz]</td>
    </tr>
    <tr>
      <td>lib2</td>
      <td>TuGG_TiteSeq_09_bin4</td>
      <td>TuGG_TiteSeq</td>
      <td>4</td>
      <td>9.0</td>
      <td>210624</td>
      <td>731</td>
      <td>[/shared/ngs/illumina/tstarr/220428_VH00699_111_AAATNWKM5/Unaligned/Project_tstarr/220420_s18-b4_S64_R1_001.fastq.gz]</td>
    </tr>
  </tbody>
</table>


Make sure library / sample combinations are unique:


```python
assert len(barcode_runs) == len(barcode_runs.groupby(['library', 'sample']))
```

Make sure the the libraries for which we have barcode runs are all in our variant table:


```python
unknown_libs = set(barcode_runs['library']) - set(variants['library'])
if unknown_libs:
    raise ValueError(f"Libraries with barcode runs not in variant table: {unknown_libs}")
```

Now we initialize an [IlluminaBarcodeParser](https://jbloomlab.github.io/dms_variants/dms_variants.illuminabarcodeparser.html#dms_variants.illuminabarcodeparser.IlluminaBarcodeParser) for each library.

First, get the length of the barcode from the alignment target after making sure the same length for all targets:


```python
bclen = len(targets.targets[0].get_feature('barcode').seq)

assert (bclen == len(target.get_feature('barcode').seq) for target in targets.targets)

print(f"Barcodes of length {bclen}")
```

    Barcodes of length 16


The other barcode parsing params come from the config file:


```python
parser_params = config['illumina_barcode_parser_params']

display(HTML(
    pd.Series(parser_params, name='value')
    .rename_axis(index='parameter')
    .reset_index()
    .to_html(index=False)
    ))
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>parameter</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>upstream</td>
      <td>GGCCGC</td>
    </tr>
    <tr>
      <td>downstream</td>
      <td></td>
    </tr>
    <tr>
      <td>minq</td>
      <td>20</td>
    </tr>
    <tr>
      <td>upstream_mismatch</td>
      <td>1</td>
    </tr>
    <tr>
      <td>downstream_mismatch</td>
      <td>0</td>
    </tr>
  </tbody>
</table>


The parser needs to know the set of valid barcodes, which are stored in the variant table and are different for each library.
So we create a different parser for each library using these valid barcode sets:


```python
# create dict keyed by library, value is parser for library
parsers = {lib: dms_variants.illuminabarcodeparser.IlluminaBarcodeParser(
                    bclen=bclen,
                    valid_barcodes=variants.loc[variants['library']==lib]['barcode'],
                    **parser_params)
           for lib in set(variants['library'])}

print('Number of valid barcodes searched for by each parser:')
display(HTML(
    pd.DataFrame([(lib, len(p.valid_barcodes)) for lib, p in parsers.items()],
                 columns=['library', 'number of valid barcodes'])
    .to_html(index=False)
    ))
```

    Number of valid barcodes searched for by each parser:



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>library</th>
      <th>number of valid barcodes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>lib2</td>
      <td>99370</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>93059</td>
    </tr>
  </tbody>
</table>


## Parse barcodes
We now parse the barcodes.
Since this will take a while, we utilize multiple CPUs via the Python [multiprocessing](https://docs.python.org/3.6/library/multiprocessing.html) module.
First, determine how many CPUs to use.
We use the minimum of the user-specified number hardcoded below and the number actually available.
(If you are running *interactively* on the Hutch cluster, you may need to reduce the number below in order to avoid an error as there is an enforced CPU limit on the home `rhino` nodes):


```python
ncpus = min(config['max_cpus'], multiprocessing.cpu_count())
print(f"Using {ncpus} CPUs")
```

    Using 8 CPUs


Parse the barcodes in parallel via a [multiprocessing.Pool](https://docs.python.org/3.6/library/multiprocessing.html#multiprocessing.pool.Pool) using all the available CPUs to get a list of the data frames with barcode counts / fates for each sample:


```python
def process_func(parser, r1files, library, sample):
    """Convenience function to be starmapped to multiprocessing pool."""
    return parser.parse(r1files, add_cols={'library': library, 'sample': sample})

# parallel computation of list of data frames
with multiprocessing.pool.Pool(processes=ncpus) as pool:
    bclist = pool.starmap(
                process_func,
                [(parsers[run.library], run.R1, run.library, run.sample)
                  for run in barcode_runs.itertuples()],
                )
```

Now concatenate the list into data frames of barcode counts and barcode fates:


```python
counts = pd.concat([samplecounts for samplecounts, _ in bclist],
                   sort=False,
                   ignore_index=True)

print('First few lines of counts data frame:')
display(HTML(counts.head().to_html(index=False)))

fates = pd.concat([samplefates for _, samplefates in bclist],
                  sort=False,
                  ignore_index=True)

print('First few lines of fates data frame:')
display(HTML(fates.head().to_html(index=False)))
```

    First few lines of counts data frame:



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>barcode</th>
      <th>count</th>
      <th>library</th>
      <th>sample</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>AGCATACCCTTAACAA</td>
      <td>26343</td>
      <td>lib1</td>
      <td>SortSeq_bin1</td>
    </tr>
    <tr>
      <td>TGACGCCTTATCCTCC</td>
      <td>20015</td>
      <td>lib1</td>
      <td>SortSeq_bin1</td>
    </tr>
    <tr>
      <td>TGCGATGGTACGTCAA</td>
      <td>15678</td>
      <td>lib1</td>
      <td>SortSeq_bin1</td>
    </tr>
    <tr>
      <td>AACTACACGGATAGGT</td>
      <td>14906</td>
      <td>lib1</td>
      <td>SortSeq_bin1</td>
    </tr>
    <tr>
      <td>CATAATGAATGTGCAA</td>
      <td>14667</td>
      <td>lib1</td>
      <td>SortSeq_bin1</td>
    </tr>
  </tbody>
</table>


    First few lines of fates data frame:



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>fate</th>
      <th>count</th>
      <th>library</th>
      <th>sample</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>valid barcode</td>
      <td>5878929</td>
      <td>lib1</td>
      <td>SortSeq_bin1</td>
    </tr>
    <tr>
      <td>invalid barcode</td>
      <td>4760919</td>
      <td>lib1</td>
      <td>SortSeq_bin1</td>
    </tr>
    <tr>
      <td>low quality barcode</td>
      <td>833191</td>
      <td>lib1</td>
      <td>SortSeq_bin1</td>
    </tr>
    <tr>
      <td>failed chastity filter</td>
      <td>523889</td>
      <td>lib1</td>
      <td>SortSeq_bin1</td>
    </tr>
    <tr>
      <td>unparseable barcode</td>
      <td>144412</td>
      <td>lib1</td>
      <td>SortSeq_bin1</td>
    </tr>
  </tbody>
</table>


## Examine fates of parsed barcodes
First, we'll analyze the "fates" of the parsed barcodes.
These fates represent what happened to each Illumina read we parsed:
 - Did the barcode read fail the Illumina chastity filter?
 - Was the barcode *unparseable* (i.e., the read didn't appear to be a valid barcode based on flanking regions)?
 - Was the barcode sequence too *low quality* based on the Illumina quality scores?
 - Was the barcode parseable but *invalid* (i.e., not in our list of variant-associated barcodes in the codon variant table)?
 - Was the barcode *valid*, and so will be added to variant counts.
 
First, we just write a CSV file with all the barcode fates:


```python
fatesfile = os.path.join(config['counts_dir'], 'barcode_fates.csv')
print(f"Writing barcode fates to {fatesfile}")
fates.to_csv(fatesfile, index=False)
```

    Writing barcode fates to results/counts/barcode_fates.csv


Next, we tabulate the barcode fates in wide format:


```python
display(HTML(fates
             .pivot_table(columns='fate',
                          values='count',
                          index=['library', 'sample'])
             .to_html()
             ))
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fate</th>
      <th>failed chastity filter</th>
      <th>invalid barcode</th>
      <th>low quality barcode</th>
      <th>unparseable barcode</th>
      <th>valid barcode</th>
    </tr>
    <tr>
      <th>library</th>
      <th>sample</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="92" valign="top">lib1</th>
      <th>PSR_01_bin1</th>
      <td>2760</td>
      <td>76609</td>
      <td>8875</td>
      <td>6707</td>
      <td>261892</td>
    </tr>
    <tr>
      <th>PSR_01_bin2</th>
      <td>33281</td>
      <td>881110</td>
      <td>105293</td>
      <td>53648</td>
      <td>3071392</td>
    </tr>
    <tr>
      <th>PSR_01_bin3</th>
      <td>92962</td>
      <td>2527624</td>
      <td>304416</td>
      <td>138579</td>
      <td>8713391</td>
    </tr>
    <tr>
      <th>PSR_01_bin4</th>
      <td>31565</td>
      <td>876540</td>
      <td>105307</td>
      <td>43313</td>
      <td>2878818</td>
    </tr>
    <tr>
      <th>PSR_02_bin1</th>
      <td>2260</td>
      <td>60269</td>
      <td>7086</td>
      <td>6288</td>
      <td>213345</td>
    </tr>
    <tr>
      <th>PSR_02_bin2</th>
      <td>54573</td>
      <td>1384490</td>
      <td>182158</td>
      <td>79424</td>
      <td>4816497</td>
    </tr>
    <tr>
      <th>PSR_02_bin3</th>
      <td>53449</td>
      <td>1521122</td>
      <td>179420</td>
      <td>83758</td>
      <td>5160624</td>
    </tr>
    <tr>
      <th>PSR_02_bin4</th>
      <td>20733</td>
      <td>573716</td>
      <td>65237</td>
      <td>35128</td>
      <td>1834236</td>
    </tr>
    <tr>
      <th>PSR_03_bin1</th>
      <td>9640</td>
      <td>265342</td>
      <td>33949</td>
      <td>17255</td>
      <td>925723</td>
    </tr>
    <tr>
      <th>PSR_03_bin2</th>
      <td>105581</td>
      <td>2853238</td>
      <td>337705</td>
      <td>155594</td>
      <td>9923513</td>
    </tr>
    <tr>
      <th>PSR_03_bin3</th>
      <td>37799</td>
      <td>1031692</td>
      <td>125139</td>
      <td>60195</td>
      <td>3437962</td>
    </tr>
    <tr>
      <th>PSR_03_bin4</th>
      <td>8856</td>
      <td>240557</td>
      <td>28232</td>
      <td>17746</td>
      <td>784493</td>
    </tr>
    <tr>
      <th>PSR_04_bin1</th>
      <td>59868</td>
      <td>1541305</td>
      <td>186597</td>
      <td>87915</td>
      <td>5366232</td>
    </tr>
    <tr>
      <th>PSR_04_bin2</th>
      <td>85427</td>
      <td>2272152</td>
      <td>275607</td>
      <td>124985</td>
      <td>7832291</td>
    </tr>
    <tr>
      <th>PSR_04_bin3</th>
      <td>16986</td>
      <td>468779</td>
      <td>54832</td>
      <td>24042</td>
      <td>1546298</td>
    </tr>
    <tr>
      <th>PSR_04_bin4</th>
      <td>2689</td>
      <td>63613</td>
      <td>7695</td>
      <td>4497</td>
      <td>204246</td>
    </tr>
    <tr>
      <th>PSR_05_bin1</th>
      <td>96819</td>
      <td>2602487</td>
      <td>315146</td>
      <td>136049</td>
      <td>8845215</td>
    </tr>
    <tr>
      <th>PSR_05_bin2</th>
      <td>4712</td>
      <td>111266</td>
      <td>13496</td>
      <td>9240</td>
      <td>374768</td>
    </tr>
    <tr>
      <th>PSR_05_bin3</th>
      <td>945</td>
      <td>25681</td>
      <td>2129</td>
      <td>3216</td>
      <td>59291</td>
    </tr>
    <tr>
      <th>PSR_05_bin4</th>
      <td>170</td>
      <td>216</td>
      <td>11</td>
      <td>2451</td>
      <td>307</td>
    </tr>
    <tr>
      <th>SortSeq_bin1</th>
      <td>523889</td>
      <td>4760919</td>
      <td>833191</td>
      <td>144412</td>
      <td>5878929</td>
    </tr>
    <tr>
      <th>SortSeq_bin2</th>
      <td>727426</td>
      <td>3538326</td>
      <td>1173192</td>
      <td>251148</td>
      <td>11430301</td>
    </tr>
    <tr>
      <th>SortSeq_bin3</th>
      <td>746488</td>
      <td>3531818</td>
      <td>1201586</td>
      <td>256451</td>
      <td>11931433</td>
    </tr>
    <tr>
      <th>SortSeq_bin4</th>
      <td>699173</td>
      <td>3195131</td>
      <td>1123721</td>
      <td>237730</td>
      <td>11077972</td>
    </tr>
    <tr>
      <th>TiteSeq_01_bin1</th>
      <td>723</td>
      <td>23498</td>
      <td>2512</td>
      <td>5588</td>
      <td>69967</td>
    </tr>
    <tr>
      <th>TiteSeq_01_bin2</th>
      <td>2219</td>
      <td>66911</td>
      <td>7438</td>
      <td>5391</td>
      <td>211761</td>
    </tr>
    <tr>
      <th>TiteSeq_01_bin3</th>
      <td>27109</td>
      <td>736192</td>
      <td>89164</td>
      <td>35800</td>
      <td>2428144</td>
    </tr>
    <tr>
      <th>TiteSeq_01_bin4</th>
      <td>109822</td>
      <td>2886347</td>
      <td>371062</td>
      <td>153582</td>
      <td>9840677</td>
    </tr>
    <tr>
      <th>TiteSeq_02_bin1</th>
      <td>529</td>
      <td>15276</td>
      <td>1740</td>
      <td>2432</td>
      <td>47643</td>
    </tr>
    <tr>
      <th>TiteSeq_02_bin2</th>
      <td>5857</td>
      <td>157481</td>
      <td>19898</td>
      <td>12004</td>
      <td>527511</td>
    </tr>
    <tr>
      <th>TiteSeq_02_bin3</th>
      <td>29469</td>
      <td>836629</td>
      <td>98832</td>
      <td>53673</td>
      <td>2724171</td>
    </tr>
    <tr>
      <th>TiteSeq_02_bin4</th>
      <td>118649</td>
      <td>3200851</td>
      <td>390447</td>
      <td>198223</td>
      <td>10962044</td>
    </tr>
    <tr>
      <th>TiteSeq_03_bin1</th>
      <td>3000</td>
      <td>85774</td>
      <td>10330</td>
      <td>7465</td>
      <td>281796</td>
    </tr>
    <tr>
      <th>TiteSeq_03_bin2</th>
      <td>15854</td>
      <td>439081</td>
      <td>51347</td>
      <td>24883</td>
      <td>1432261</td>
    </tr>
    <tr>
      <th>TiteSeq_03_bin3</th>
      <td>39990</td>
      <td>1079409</td>
      <td>133347</td>
      <td>66266</td>
      <td>3548895</td>
    </tr>
    <tr>
      <th>TiteSeq_03_bin4</th>
      <td>87944</td>
      <td>2313093</td>
      <td>290636</td>
      <td>145619</td>
      <td>7918255</td>
    </tr>
    <tr>
      <th>TiteSeq_04_bin1</th>
      <td>4473</td>
      <td>119291</td>
      <td>13553</td>
      <td>9937</td>
      <td>391594</td>
    </tr>
    <tr>
      <th>TiteSeq_04_bin2</th>
      <td>12230</td>
      <td>336998</td>
      <td>39710</td>
      <td>17381</td>
      <td>1104003</td>
    </tr>
    <tr>
      <th>TiteSeq_04_bin3</th>
      <td>79614</td>
      <td>2132495</td>
      <td>262375</td>
      <td>117650</td>
      <td>7168931</td>
    </tr>
    <tr>
      <th>TiteSeq_04_bin4</th>
      <td>33025</td>
      <td>756991</td>
      <td>106613</td>
      <td>42706</td>
      <td>2610429</td>
    </tr>
    <tr>
      <th>TiteSeq_05_bin1</th>
      <td>6038</td>
      <td>166913</td>
      <td>19351</td>
      <td>8995</td>
      <td>551036</td>
    </tr>
    <tr>
      <th>TiteSeq_05_bin2</th>
      <td>14001</td>
      <td>388159</td>
      <td>44619</td>
      <td>26975</td>
      <td>1282207</td>
    </tr>
    <tr>
      <th>TiteSeq_05_bin3</th>
      <td>43395</td>
      <td>1115507</td>
      <td>137192</td>
      <td>66894</td>
      <td>3825622</td>
    </tr>
    <tr>
      <th>TiteSeq_05_bin4</th>
      <td>2961</td>
      <td>82456</td>
      <td>8955</td>
      <td>10805</td>
      <td>254728</td>
    </tr>
    <tr>
      <th>TiteSeq_06_bin1</th>
      <td>16330</td>
      <td>478179</td>
      <td>56630</td>
      <td>24360</td>
      <td>1588020</td>
    </tr>
    <tr>
      <th>TiteSeq_06_bin2</th>
      <td>115068</td>
      <td>2983159</td>
      <td>367642</td>
      <td>162896</td>
      <td>10241861</td>
    </tr>
    <tr>
      <th>TiteSeq_06_bin3</th>
      <td>3482</td>
      <td>102260</td>
      <td>11895</td>
      <td>6229</td>
      <td>341369</td>
    </tr>
    <tr>
      <th>TiteSeq_06_bin4</th>
      <td>281</td>
      <td>3381</td>
      <td>415</td>
      <td>4784</td>
      <td>11254</td>
    </tr>
    <tr>
      <th>TiteSeq_07_bin1</th>
      <td>105301</td>
      <td>2712156</td>
      <td>342585</td>
      <td>147733</td>
      <td>9198668</td>
    </tr>
    <tr>
      <th>TiteSeq_07_bin2</th>
      <td>12495</td>
      <td>340544</td>
      <td>41258</td>
      <td>27318</td>
      <td>1167631</td>
    </tr>
    <tr>
      <th>TiteSeq_07_bin3</th>
      <td>32</td>
      <td>752</td>
      <td>62</td>
      <td>362</td>
      <td>1381</td>
    </tr>
    <tr>
      <th>TiteSeq_07_bin4</th>
      <td>22</td>
      <td>66</td>
      <td>7</td>
      <td>405</td>
      <td>170</td>
    </tr>
    <tr>
      <th>TiteSeq_08_bin1</th>
      <td>21561</td>
      <td>558221</td>
      <td>70630</td>
      <td>29815</td>
      <td>1894494</td>
    </tr>
    <tr>
      <th>TiteSeq_08_bin2</th>
      <td>2670</td>
      <td>59900</td>
      <td>7131</td>
      <td>8911</td>
      <td>205514</td>
    </tr>
    <tr>
      <th>TiteSeq_08_bin3</th>
      <td>59</td>
      <td>186</td>
      <td>24</td>
      <td>1876</td>
      <td>152</td>
    </tr>
    <tr>
      <th>TiteSeq_08_bin4</th>
      <td>54</td>
      <td>530</td>
      <td>30</td>
      <td>875</td>
      <td>99</td>
    </tr>
    <tr>
      <th>TiteSeq_09_bin1</th>
      <td>119463</td>
      <td>3152051</td>
      <td>390186</td>
      <td>167576</td>
      <td>10674684</td>
    </tr>
    <tr>
      <th>TiteSeq_09_bin2</th>
      <td>1161</td>
      <td>28927</td>
      <td>3801</td>
      <td>3064</td>
      <td>101073</td>
    </tr>
    <tr>
      <th>TiteSeq_09_bin3</th>
      <td>297</td>
      <td>212</td>
      <td>30</td>
      <td>2671</td>
      <td>245</td>
    </tr>
    <tr>
      <th>TiteSeq_09_bin4</th>
      <td>24</td>
      <td>33</td>
      <td>3</td>
      <td>623</td>
      <td>38</td>
    </tr>
    <tr>
      <th>TuGG_TiteSeq_01_bin1</th>
      <td>0</td>
      <td>34391</td>
      <td>23747</td>
      <td>1961</td>
      <td>116108</td>
    </tr>
    <tr>
      <th>TuGG_TiteSeq_01_bin2</th>
      <td>0</td>
      <td>1922364</td>
      <td>1364144</td>
      <td>101478</td>
      <td>6630549</td>
    </tr>
    <tr>
      <th>TuGG_TiteSeq_01_bin3</th>
      <td>0</td>
      <td>2196495</td>
      <td>1579084</td>
      <td>116910</td>
      <td>7395863</td>
    </tr>
    <tr>
      <th>TuGG_TiteSeq_01_bin4</th>
      <td>0</td>
      <td>585227</td>
      <td>413973</td>
      <td>26731</td>
      <td>1890834</td>
    </tr>
    <tr>
      <th>TuGG_TiteSeq_02_bin1</th>
      <td>0</td>
      <td>1614297</td>
      <td>1137941</td>
      <td>87986</td>
      <td>5526837</td>
    </tr>
    <tr>
      <th>TuGG_TiteSeq_02_bin2</th>
      <td>0</td>
      <td>2565247</td>
      <td>1801977</td>
      <td>136491</td>
      <td>8702674</td>
    </tr>
    <tr>
      <th>TuGG_TiteSeq_02_bin3</th>
      <td>0</td>
      <td>349017</td>
      <td>230950</td>
      <td>14592</td>
      <td>1144828</td>
    </tr>
    <tr>
      <th>TuGG_TiteSeq_02_bin4</th>
      <td>0</td>
      <td>115817</td>
      <td>75467</td>
      <td>4357</td>
      <td>360140</td>
    </tr>
    <tr>
      <th>TuGG_TiteSeq_03_bin1</th>
      <td>0</td>
      <td>1889580</td>
      <td>1339872</td>
      <td>100000</td>
      <td>6449176</td>
    </tr>
    <tr>
      <th>TuGG_TiteSeq_03_bin2</th>
      <td>0</td>
      <td>3134038</td>
      <td>2267255</td>
      <td>167143</td>
      <td>10700456</td>
    </tr>
    <tr>
      <th>TuGG_TiteSeq_03_bin3</th>
      <td>0</td>
      <td>555265</td>
      <td>377589</td>
      <td>24604</td>
      <td>1769918</td>
    </tr>
    <tr>
      <th>TuGG_TiteSeq_03_bin4</th>
      <td>0</td>
      <td>76135</td>
      <td>50551</td>
      <td>3004</td>
      <td>241069</td>
    </tr>
    <tr>
      <th>TuGG_TiteSeq_04_bin1</th>
      <td>0</td>
      <td>4128786</td>
      <td>3065871</td>
      <td>218040</td>
      <td>13962133</td>
    </tr>
    <tr>
      <th>TuGG_TiteSeq_04_bin2</th>
      <td>0</td>
      <td>441448</td>
      <td>323055</td>
      <td>22549</td>
      <td>1492240</td>
    </tr>
    <tr>
      <th>TuGG_TiteSeq_04_bin3</th>
      <td>0</td>
      <td>30016</td>
      <td>23203</td>
      <td>1640</td>
      <td>99049</td>
    </tr>
    <tr>
      <th>TuGG_TiteSeq_04_bin4</th>
      <td>0</td>
      <td>10838</td>
      <td>10579</td>
      <td>493</td>
      <td>33093</td>
    </tr>
    <tr>
      <th>TuGG_TiteSeq_05_bin1</th>
      <td>0</td>
      <td>3555675</td>
      <td>2560495</td>
      <td>182995</td>
      <td>11997496</td>
    </tr>
    <tr>
      <th>TuGG_TiteSeq_05_bin2</th>
      <td>0</td>
      <td>114228</td>
      <td>81702</td>
      <td>5314</td>
      <td>385899</td>
    </tr>
    <tr>
      <th>TuGG_TiteSeq_05_bin3</th>
      <td>0</td>
      <td>1740</td>
      <td>1489</td>
      <td>83</td>
      <td>4804</td>
    </tr>
    <tr>
      <th>TuGG_TiteSeq_05_bin4</th>
      <td>0</td>
      <td>1702</td>
      <td>3115</td>
      <td>122</td>
      <td>5965</td>
    </tr>
    <tr>
      <th>TuGG_TiteSeq_06_bin1</th>
      <td>0</td>
      <td>5006492</td>
      <td>3662674</td>
      <td>262476</td>
      <td>16875795</td>
    </tr>
    <tr>
      <th>TuGG_TiteSeq_06_bin2</th>
      <td>0</td>
      <td>107709</td>
      <td>78938</td>
      <td>5455</td>
      <td>371868</td>
    </tr>
    <tr>
      <th>TuGG_TiteSeq_06_bin3</th>
      <td>0</td>
      <td>820</td>
      <td>5824</td>
      <td>56</td>
      <td>2002</td>
    </tr>
    <tr>
      <th>TuGG_TiteSeq_06_bin4</th>
      <td>0</td>
      <td>2139</td>
      <td>794</td>
      <td>31</td>
      <td>2644</td>
    </tr>
    <tr>
      <th>TuGG_TiteSeq_07_bin1</th>
      <td>0</td>
      <td>4294140</td>
      <td>3141335</td>
      <td>223749</td>
      <td>14519467</td>
    </tr>
    <tr>
      <th>TuGG_TiteSeq_07_bin2</th>
      <td>0</td>
      <td>138356</td>
      <td>99278</td>
      <td>7064</td>
      <td>468326</td>
    </tr>
    <tr>
      <th>TuGG_TiteSeq_07_bin3</th>
      <td>0</td>
      <td>145</td>
      <td>687</td>
      <td>10</td>
      <td>460</td>
    </tr>
    <tr>
      <th>TuGG_TiteSeq_07_bin4</th>
      <td>0</td>
      <td>853</td>
      <td>1125</td>
      <td>30</td>
      <td>2244</td>
    </tr>
    <tr>
      <th>TuGG_TiteSeq_09_bin1</th>
      <td>0</td>
      <td>4624113</td>
      <td>3273253</td>
      <td>239292</td>
      <td>15612776</td>
    </tr>
    <tr>
      <th>TuGG_TiteSeq_09_bin2</th>
      <td>0</td>
      <td>138172</td>
      <td>101697</td>
      <td>6871</td>
      <td>461580</td>
    </tr>
    <tr>
      <th>TuGG_TiteSeq_09_bin3</th>
      <td>0</td>
      <td>159</td>
      <td>270</td>
      <td>7</td>
      <td>363</td>
    </tr>
    <tr>
      <th>TuGG_TiteSeq_09_bin4</th>
      <td>0</td>
      <td>2682</td>
      <td>726</td>
      <td>48</td>
      <td>347</td>
    </tr>
    <tr>
      <th rowspan="92" valign="top">lib2</th>
      <th>PSR_01_bin1</th>
      <td>1562</td>
      <td>48082</td>
      <td>5022</td>
      <td>2414</td>
      <td>142422</td>
    </tr>
    <tr>
      <th>PSR_01_bin2</th>
      <td>20497</td>
      <td>602681</td>
      <td>65954</td>
      <td>28390</td>
      <td>1808361</td>
    </tr>
    <tr>
      <th>PSR_01_bin3</th>
      <td>86014</td>
      <td>2634986</td>
      <td>278379</td>
      <td>127365</td>
      <td>7845102</td>
    </tr>
    <tr>
      <th>PSR_01_bin4</th>
      <td>42006</td>
      <td>1312985</td>
      <td>139987</td>
      <td>67516</td>
      <td>3798485</td>
    </tr>
    <tr>
      <th>PSR_02_bin1</th>
      <td>1367</td>
      <td>42002</td>
      <td>4283</td>
      <td>2837</td>
      <td>123214</td>
    </tr>
    <tr>
      <th>PSR_02_bin2</th>
      <td>36368</td>
      <td>1058378</td>
      <td>117892</td>
      <td>55020</td>
      <td>3164506</td>
    </tr>
    <tr>
      <th>PSR_02_bin3</th>
      <td>46831</td>
      <td>1418500</td>
      <td>157480</td>
      <td>68029</td>
      <td>4240036</td>
    </tr>
    <tr>
      <th>PSR_02_bin4</th>
      <td>15274</td>
      <td>471859</td>
      <td>50349</td>
      <td>20475</td>
      <td>1341122</td>
    </tr>
    <tr>
      <th>PSR_03_bin1</th>
      <td>1207</td>
      <td>36838</td>
      <td>3683</td>
      <td>2161</td>
      <td>108908</td>
    </tr>
    <tr>
      <th>PSR_03_bin2</th>
      <td>16676</td>
      <td>482083</td>
      <td>51289</td>
      <td>29575</td>
      <td>1455089</td>
    </tr>
    <tr>
      <th>PSR_03_bin3</th>
      <td>42131</td>
      <td>1353775</td>
      <td>138689</td>
      <td>65531</td>
      <td>4013190</td>
    </tr>
    <tr>
      <th>PSR_03_bin4</th>
      <td>3981</td>
      <td>129208</td>
      <td>12911</td>
      <td>6491</td>
      <td>364997</td>
    </tr>
    <tr>
      <th>PSR_04_bin1</th>
      <td>4946</td>
      <td>159115</td>
      <td>16241</td>
      <td>8762</td>
      <td>480145</td>
    </tr>
    <tr>
      <th>PSR_04_bin2</th>
      <td>26313</td>
      <td>823098</td>
      <td>87697</td>
      <td>41658</td>
      <td>2460338</td>
    </tr>
    <tr>
      <th>PSR_04_bin3</th>
      <td>34705</td>
      <td>1027936</td>
      <td>108936</td>
      <td>52602</td>
      <td>3013980</td>
    </tr>
    <tr>
      <th>PSR_04_bin4</th>
      <td>4733</td>
      <td>143584</td>
      <td>15200</td>
      <td>7153</td>
      <td>409592</td>
    </tr>
    <tr>
      <th>PSR_05_bin1</th>
      <td>161760</td>
      <td>4804606</td>
      <td>525383</td>
      <td>228965</td>
      <td>14207135</td>
    </tr>
    <tr>
      <th>PSR_05_bin2</th>
      <td>24225</td>
      <td>720718</td>
      <td>80451</td>
      <td>35555</td>
      <td>2143354</td>
    </tr>
    <tr>
      <th>PSR_05_bin3</th>
      <td>218</td>
      <td>4214</td>
      <td>417</td>
      <td>1597</td>
      <td>12272</td>
    </tr>
    <tr>
      <th>PSR_05_bin4</th>
      <td>39</td>
      <td>54</td>
      <td>3</td>
      <td>1003</td>
      <td>69</td>
    </tr>
    <tr>
      <th>SortSeq_bin1</th>
      <td>509852</td>
      <td>4627673</td>
      <td>813556</td>
      <td>198130</td>
      <td>5774217</td>
    </tr>
    <tr>
      <th>SortSeq_bin2</th>
      <td>795781</td>
      <td>4205255</td>
      <td>1267242</td>
      <td>273376</td>
      <td>11874774</td>
    </tr>
    <tr>
      <th>SortSeq_bin3</th>
      <td>821118</td>
      <td>4127023</td>
      <td>1307794</td>
      <td>275945</td>
      <td>12092240</td>
    </tr>
    <tr>
      <th>SortSeq_bin4</th>
      <td>759644</td>
      <td>3827605</td>
      <td>1201061</td>
      <td>262920</td>
      <td>11482708</td>
    </tr>
    <tr>
      <th>TiteSeq_01_bin1</th>
      <td>127</td>
      <td>1831</td>
      <td>94</td>
      <td>1012</td>
      <td>1607</td>
    </tr>
    <tr>
      <th>TiteSeq_01_bin2</th>
      <td>1683</td>
      <td>60948</td>
      <td>5501</td>
      <td>2139</td>
      <td>138650</td>
    </tr>
    <tr>
      <th>TiteSeq_01_bin3</th>
      <td>20885</td>
      <td>623761</td>
      <td>65554</td>
      <td>26992</td>
      <td>1727022</td>
    </tr>
    <tr>
      <th>TiteSeq_01_bin4</th>
      <td>61074</td>
      <td>1710486</td>
      <td>197991</td>
      <td>82328</td>
      <td>5071442</td>
    </tr>
    <tr>
      <th>TiteSeq_02_bin1</th>
      <td>1144</td>
      <td>40401</td>
      <td>3776</td>
      <td>2800</td>
      <td>98705</td>
    </tr>
    <tr>
      <th>TiteSeq_02_bin2</th>
      <td>1702</td>
      <td>56116</td>
      <td>5737</td>
      <td>3198</td>
      <td>153481</td>
    </tr>
    <tr>
      <th>TiteSeq_02_bin3</th>
      <td>15466</td>
      <td>510137</td>
      <td>51261</td>
      <td>23383</td>
      <td>1453815</td>
    </tr>
    <tr>
      <th>TiteSeq_02_bin4</th>
      <td>73472</td>
      <td>2044968</td>
      <td>243880</td>
      <td>102647</td>
      <td>6077556</td>
    </tr>
    <tr>
      <th>TiteSeq_03_bin1</th>
      <td>3090</td>
      <td>99733</td>
      <td>9989</td>
      <td>4813</td>
      <td>252931</td>
    </tr>
    <tr>
      <th>TiteSeq_03_bin2</th>
      <td>5054</td>
      <td>178199</td>
      <td>17369</td>
      <td>8192</td>
      <td>472019</td>
    </tr>
    <tr>
      <th>TiteSeq_03_bin3</th>
      <td>31150</td>
      <td>981385</td>
      <td>103120</td>
      <td>50095</td>
      <td>2791190</td>
    </tr>
    <tr>
      <th>TiteSeq_03_bin4</th>
      <td>72237</td>
      <td>2133857</td>
      <td>239146</td>
      <td>111556</td>
      <td>6424338</td>
    </tr>
    <tr>
      <th>TiteSeq_04_bin1</th>
      <td>4149</td>
      <td>131597</td>
      <td>12881</td>
      <td>7011</td>
      <td>350326</td>
    </tr>
    <tr>
      <th>TiteSeq_04_bin2</th>
      <td>9666</td>
      <td>317245</td>
      <td>32646</td>
      <td>15224</td>
      <td>890010</td>
    </tr>
    <tr>
      <th>TiteSeq_04_bin3</th>
      <td>29876</td>
      <td>994115</td>
      <td>104818</td>
      <td>53698</td>
      <td>2885157</td>
    </tr>
    <tr>
      <th>TiteSeq_04_bin4</th>
      <td>23366</td>
      <td>772506</td>
      <td>82782</td>
      <td>40686</td>
      <td>2355943</td>
    </tr>
    <tr>
      <th>TiteSeq_05_bin1</th>
      <td>4485</td>
      <td>149378</td>
      <td>15427</td>
      <td>7103</td>
      <td>410521</td>
    </tr>
    <tr>
      <th>TiteSeq_05_bin2</th>
      <td>15788</td>
      <td>493051</td>
      <td>50774</td>
      <td>31407</td>
      <td>1385127</td>
    </tr>
    <tr>
      <th>TiteSeq_05_bin3</th>
      <td>43332</td>
      <td>1452937</td>
      <td>160151</td>
      <td>75502</td>
      <td>4361933</td>
    </tr>
    <tr>
      <th>TiteSeq_05_bin4</th>
      <td>1896</td>
      <td>53281</td>
      <td>5713</td>
      <td>6059</td>
      <td>164249</td>
    </tr>
    <tr>
      <th>TiteSeq_06_bin1</th>
      <td>19487</td>
      <td>644683</td>
      <td>66729</td>
      <td>30172</td>
      <td>1798422</td>
    </tr>
    <tr>
      <th>TiteSeq_06_bin2</th>
      <td>95814</td>
      <td>2818095</td>
      <td>321691</td>
      <td>144996</td>
      <td>8444622</td>
    </tr>
    <tr>
      <th>TiteSeq_06_bin3</th>
      <td>1435</td>
      <td>44191</td>
      <td>4834</td>
      <td>5635</td>
      <td>138017</td>
    </tr>
    <tr>
      <th>TiteSeq_06_bin4</th>
      <td>156</td>
      <td>97</td>
      <td>11</td>
      <td>3070</td>
      <td>80</td>
    </tr>
    <tr>
      <th>TiteSeq_07_bin1</th>
      <td>65227</td>
      <td>1880842</td>
      <td>212932</td>
      <td>89992</td>
      <td>5545021</td>
    </tr>
    <tr>
      <th>TiteSeq_07_bin2</th>
      <td>7847</td>
      <td>236386</td>
      <td>25683</td>
      <td>12951</td>
      <td>717704</td>
    </tr>
    <tr>
      <th>TiteSeq_07_bin3</th>
      <td>453</td>
      <td>10530</td>
      <td>1149</td>
      <td>1273</td>
      <td>31884</td>
    </tr>
    <tr>
      <th>TiteSeq_07_bin4</th>
      <td>84</td>
      <td>18</td>
      <td>15</td>
      <td>1004</td>
      <td>20</td>
    </tr>
    <tr>
      <th>TiteSeq_08_bin1</th>
      <td>92740</td>
      <td>2822029</td>
      <td>313287</td>
      <td>134308</td>
      <td>8350688</td>
    </tr>
    <tr>
      <th>TiteSeq_08_bin2</th>
      <td>4522</td>
      <td>115452</td>
      <td>11996</td>
      <td>18720</td>
      <td>341937</td>
    </tr>
    <tr>
      <th>TiteSeq_08_bin3</th>
      <td>119</td>
      <td>87</td>
      <td>2</td>
      <td>6303</td>
      <td>84</td>
    </tr>
    <tr>
      <th>TiteSeq_08_bin4</th>
      <td>214</td>
      <td>76</td>
      <td>12</td>
      <td>2888</td>
      <td>204</td>
    </tr>
    <tr>
      <th>TiteSeq_09_bin1</th>
      <td>89791</td>
      <td>2691236</td>
      <td>296258</td>
      <td>133990</td>
      <td>7891843</td>
    </tr>
    <tr>
      <th>TiteSeq_09_bin2</th>
      <td>1902</td>
      <td>59097</td>
      <td>5963</td>
      <td>4419</td>
      <td>172182</td>
    </tr>
    <tr>
      <th>TiteSeq_09_bin3</th>
      <td>33</td>
      <td>18</td>
      <td>7</td>
      <td>554</td>
      <td>12</td>
    </tr>
    <tr>
      <th>TiteSeq_09_bin4</th>
      <td>94</td>
      <td>41</td>
      <td>12</td>
      <td>2863</td>
      <td>28</td>
    </tr>
    <tr>
      <th>TuGG_TiteSeq_01_bin1</th>
      <td>0</td>
      <td>31702</td>
      <td>23044</td>
      <td>1543</td>
      <td>92273</td>
    </tr>
    <tr>
      <th>TuGG_TiteSeq_01_bin2</th>
      <td>0</td>
      <td>3162590</td>
      <td>2048929</td>
      <td>152791</td>
      <td>9480848</td>
    </tr>
    <tr>
      <th>TuGG_TiteSeq_01_bin3</th>
      <td>0</td>
      <td>1769269</td>
      <td>1123222</td>
      <td>82113</td>
      <td>5198219</td>
    </tr>
    <tr>
      <th>TuGG_TiteSeq_01_bin4</th>
      <td>0</td>
      <td>393281</td>
      <td>237384</td>
      <td>16229</td>
      <td>1065719</td>
    </tr>
    <tr>
      <th>TuGG_TiteSeq_02_bin1</th>
      <td>0</td>
      <td>2032170</td>
      <td>1305656</td>
      <td>95452</td>
      <td>6032770</td>
    </tr>
    <tr>
      <th>TuGG_TiteSeq_02_bin2</th>
      <td>0</td>
      <td>3133119</td>
      <td>1999689</td>
      <td>148307</td>
      <td>9247875</td>
    </tr>
    <tr>
      <th>TuGG_TiteSeq_02_bin3</th>
      <td>0</td>
      <td>322132</td>
      <td>196005</td>
      <td>13757</td>
      <td>882369</td>
    </tr>
    <tr>
      <th>TuGG_TiteSeq_02_bin4</th>
      <td>0</td>
      <td>90093</td>
      <td>54537</td>
      <td>3178</td>
      <td>242549</td>
    </tr>
    <tr>
      <th>TuGG_TiteSeq_03_bin1</th>
      <td>0</td>
      <td>1651110</td>
      <td>1099708</td>
      <td>81919</td>
      <td>4915217</td>
    </tr>
    <tr>
      <th>TuGG_TiteSeq_03_bin2</th>
      <td>0</td>
      <td>3413796</td>
      <td>2190110</td>
      <td>161361</td>
      <td>10112490</td>
    </tr>
    <tr>
      <th>TuGG_TiteSeq_03_bin3</th>
      <td>0</td>
      <td>517845</td>
      <td>316503</td>
      <td>22759</td>
      <td>1444008</td>
    </tr>
    <tr>
      <th>TuGG_TiteSeq_03_bin4</th>
      <td>0</td>
      <td>51487</td>
      <td>31545</td>
      <td>2061</td>
      <td>137134</td>
    </tr>
    <tr>
      <th>TuGG_TiteSeq_04_bin1</th>
      <td>0</td>
      <td>6091206</td>
      <td>3896498</td>
      <td>291082</td>
      <td>17920346</td>
    </tr>
    <tr>
      <th>TuGG_TiteSeq_04_bin2</th>
      <td>0</td>
      <td>505427</td>
      <td>325386</td>
      <td>23879</td>
      <td>1478934</td>
    </tr>
    <tr>
      <th>TuGG_TiteSeq_04_bin3</th>
      <td>0</td>
      <td>18101</td>
      <td>11433</td>
      <td>717</td>
      <td>48966</td>
    </tr>
    <tr>
      <th>TuGG_TiteSeq_04_bin4</th>
      <td>0</td>
      <td>12022</td>
      <td>7406</td>
      <td>351</td>
      <td>32557</td>
    </tr>
    <tr>
      <th>TuGG_TiteSeq_05_bin1</th>
      <td>0</td>
      <td>5215279</td>
      <td>3353563</td>
      <td>241708</td>
      <td>15314750</td>
    </tr>
    <tr>
      <th>TuGG_TiteSeq_05_bin2</th>
      <td>0</td>
      <td>133936</td>
      <td>87198</td>
      <td>5853</td>
      <td>389948</td>
    </tr>
    <tr>
      <th>TuGG_TiteSeq_05_bin3</th>
      <td>0</td>
      <td>4027</td>
      <td>2955</td>
      <td>132</td>
      <td>11063</td>
    </tr>
    <tr>
      <th>TuGG_TiteSeq_05_bin4</th>
      <td>0</td>
      <td>1032</td>
      <td>942</td>
      <td>34</td>
      <td>2826</td>
    </tr>
    <tr>
      <th>TuGG_TiteSeq_06_bin1</th>
      <td>0</td>
      <td>5127003</td>
      <td>3332522</td>
      <td>238833</td>
      <td>15016308</td>
    </tr>
    <tr>
      <th>TuGG_TiteSeq_06_bin2</th>
      <td>0</td>
      <td>123626</td>
      <td>80514</td>
      <td>5773</td>
      <td>368137</td>
    </tr>
    <tr>
      <th>TuGG_TiteSeq_06_bin3</th>
      <td>0</td>
      <td>507</td>
      <td>1630</td>
      <td>112</td>
      <td>907</td>
    </tr>
    <tr>
      <th>TuGG_TiteSeq_06_bin4</th>
      <td>0</td>
      <td>7564</td>
      <td>1356</td>
      <td>92</td>
      <td>786</td>
    </tr>
    <tr>
      <th>TuGG_TiteSeq_07_bin1</th>
      <td>0</td>
      <td>5387435</td>
      <td>3457670</td>
      <td>251249</td>
      <td>15872663</td>
    </tr>
    <tr>
      <th>TuGG_TiteSeq_07_bin2</th>
      <td>0</td>
      <td>91426</td>
      <td>59985</td>
      <td>4501</td>
      <td>274519</td>
    </tr>
    <tr>
      <th>TuGG_TiteSeq_07_bin3</th>
      <td>0</td>
      <td>48</td>
      <td>481</td>
      <td>22</td>
      <td>82</td>
    </tr>
    <tr>
      <th>TuGG_TiteSeq_07_bin4</th>
      <td>0</td>
      <td>247</td>
      <td>220</td>
      <td>14</td>
      <td>578</td>
    </tr>
    <tr>
      <th>TuGG_TiteSeq_09_bin1</th>
      <td>0</td>
      <td>5152250</td>
      <td>3377972</td>
      <td>237627</td>
      <td>15119369</td>
    </tr>
    <tr>
      <th>TuGG_TiteSeq_09_bin2</th>
      <td>0</td>
      <td>137924</td>
      <td>89875</td>
      <td>6420</td>
      <td>406898</td>
    </tr>
    <tr>
      <th>TuGG_TiteSeq_09_bin3</th>
      <td>0</td>
      <td>105</td>
      <td>997</td>
      <td>9</td>
      <td>194</td>
    </tr>
    <tr>
      <th>TuGG_TiteSeq_09_bin4</th>
      <td>0</td>
      <td>407</td>
      <td>260</td>
      <td>29</td>
      <td>1008</td>
    </tr>
  </tbody>
</table>


Now we plot the barcode-read fates for each library / sample, showing the bars for valid barcodes in orange and the others in gray.
We see that the largest fraction of barcode reads correspond to valid barcodes, and most of the others are invalid barcodes (probably because the map to variants that aren't present in our variant table since we didn't associate all variants with barcodes). The exception to this is lib2 Titeseq_03_bin3; the PCR for this sample in the original sequencing run failed, so we followed it up with a single MiSeq lane. We did not filter out the PhiX reads from this data before parsing, so these PhiX reads will deflate the fraction of valid barcode reads as expected, but does not indicate any problems.


```python
barcode_fate_plot = (
    ggplot(
        fates
        .assign(sample=lambda x: pd.Categorical(x['sample'],
                                                x['sample'].unique(),
                                                ordered=True),
                fate=lambda x: pd.Categorical(x['fate'],
                                              x['fate'].unique(),
                                              ordered=True),
                is_valid=lambda x: x['fate'] == 'valid barcode'
                ), 
        aes('fate', 'count', fill='is_valid')) +
    geom_bar(stat='identity') +
    facet_grid('sample ~ library') +
    facet_grid('sample ~ library') +
    scale_fill_manual(CBPALETTE, guide=False) +
    theme(figure_size=(1.4 * (1 + fates['library'].nunique()),
                       1.7 * (1.2 + fates['sample'].nunique())),
          axis_text_x=element_text(angle=90),
          panel_grid_major_x=element_blank()
          ) +
    scale_y_continuous(labels=dms_variants.utils.latex_sci_not,
                       name='number of reads')
    )

_ = barcode_fate_plot.draw()
```


    
![png](count_variants_files/count_variants_40_0.png)
    


## Output csv of barcode counts in variant-barcode lookup table


```python
print(f"Writing variant counts to {config['variant_counts_file']}")
counts.to_csv(config['variant_counts_file'], index=False)
```

    Writing variant counts to results/counts/variant_counts.csv


The [CodonVariantTable](https://jbloomlab.github.io/dms_variants/dms_variants.codonvarianttable.html#dms_variants.codonvarianttable.CodonVariantTable) has lots of nice functions that can be used to analyze the counts it contains.
However, we do that in the next notebook so we don't have to re-run this entire (rather computationally intensive) notebook every time we want to analyze a new aspect of the counts.


```python

```
