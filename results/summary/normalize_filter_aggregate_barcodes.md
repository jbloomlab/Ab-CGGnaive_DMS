# Titeseq barcode aggregation

This notebook reads in the barcode-level counts, barcode-variant mapping, and sequencing runs total cells, so that we can prep Titeseq data for fitting by:

1. Merging in the variants substitution annotations to barcodes.
2. Normalize the counts by the total cells in each sequencing run.
3. Aggregate the counts by the variant substitution annotations.
4. Filter out low count barcodes and variants.  

## Set up analysis


```python
import warnings
import yaml
import os

import numpy as np
import pandas as pd
from IPython.display import display, HTML
```

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
```

## Data


```python
barcode_runs = pd.read_csv(config['barcode_runs']).drop(columns=["R1"])
barcode_runs.query("sample.str.startswith('TiteSeq') | sample.str.startswith('SortSeq')", inplace=True)
barcode_runs.set_index(["library", "sample"], inplace=True)
display(HTML(barcode_runs.head().to_html(index=True)))
print(barcode_runs.shape)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>sample_type</th>
      <th>sort_bin</th>
      <th>concentration</th>
      <th>date</th>
      <th>number_cells</th>
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
      <th rowspan="4" valign="top">lib1</th>
      <th>SortSeq_bin1</th>
      <td>SortSeq</td>
      <td>1</td>
      <td>NaN</td>
      <td>210621</td>
      <td>210800</td>
    </tr>
    <tr>
      <th>SortSeq_bin2</th>
      <td>SortSeq</td>
      <td>2</td>
      <td>NaN</td>
      <td>210621</td>
      <td>1984000</td>
    </tr>
    <tr>
      <th>SortSeq_bin3</th>
      <td>SortSeq</td>
      <td>3</td>
      <td>NaN</td>
      <td>210621</td>
      <td>2940000</td>
    </tr>
    <tr>
      <th>SortSeq_bin4</th>
      <td>SortSeq</td>
      <td>4</td>
      <td>NaN</td>
      <td>210621</td>
      <td>3575000</td>
    </tr>
    <tr>
      <th>lib2</th>
      <th>SortSeq_bin1</th>
      <td>SortSeq</td>
      <td>1</td>
      <td>NaN</td>
      <td>210621</td>
      <td>275500</td>
    </tr>
  </tbody>
</table>


    (80, 5)



```python
variant_counts = pd.read_csv(config['variant_counts_file'])
variant_counts.query("sample.str.startswith('TiteSeq') | sample.str.startswith('SortSeq')", inplace=True)
display(HTML(variant_counts.head().to_html(index=True)))
print(variant_counts.shape)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>barcode</th>
      <th>count</th>
      <th>library</th>
      <th>sample</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AGCATACCCTTAACAA</td>
      <td>26343</td>
      <td>lib1</td>
      <td>SortSeq_bin1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TGACGCCTTATCCTCC</td>
      <td>20015</td>
      <td>lib1</td>
      <td>SortSeq_bin1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>TGCGATGGTACGTCAA</td>
      <td>15678</td>
      <td>lib1</td>
      <td>SortSeq_bin1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AACTACACGGATAGGT</td>
      <td>14906</td>
      <td>lib1</td>
      <td>SortSeq_bin1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CATAATGAATGTGCAA</td>
      <td>14667</td>
      <td>lib1</td>
      <td>SortSeq_bin1</td>
    </tr>
  </tbody>
</table>


    (7697160, 4)



```python
codon_variant_table = pd.read_csv(config['codon_variant_table_file'])
display(HTML(codon_variant_table.head().to_html(index=True)))
print(codon_variant_table.shape)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
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
      <th>0</th>
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
      <th>1</th>
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
      <th>2</th>
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
      <th>3</th>
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
      <th>4</th>
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


    (192429, 8)


Define concentrations and bins


```python
concs = np.array(config['concentrations']['CGG']).astype(float)
concs
```




    array([1.e-06, 1.e-07, 1.e-08, 1.e-09, 1.e-10, 1.e-11, 1.e-12, 1.e-13,
           0.e+00])



Combine tables to make a single barcode level frame


```python
df_barcodes = variant_counts.merge(codon_variant_table, on=("barcode", "library"), how="left")
display(HTML(df_barcodes.head().to_html(index=True)))
print(df_barcodes.shape)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>barcode</th>
      <th>count</th>
      <th>library</th>
      <th>sample</th>
      <th>target</th>
      <th>variant_call_support</th>
      <th>codon_substitutions</th>
      <th>aa_substitutions</th>
      <th>n_codon_substitutions</th>
      <th>n_aa_substitutions</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AGCATACCCTTAACAA</td>
      <td>26343</td>
      <td>lib1</td>
      <td>SortSeq_bin1</td>
      <td>CGG_naive</td>
      <td>65</td>
      <td>CGT38ATT GTA148GAA</td>
      <td>R38I V148E</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TGACGCCTTATCCTCC</td>
      <td>20015</td>
      <td>lib1</td>
      <td>SortSeq_bin1</td>
      <td>CGG_naive</td>
      <td>23</td>
      <td>GAT100GTT CAG217TAG</td>
      <td>D100V Q217*</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>TGCGATGGTACGTCAA</td>
      <td>15678</td>
      <td>lib1</td>
      <td>SortSeq_bin1</td>
      <td>CGG_naive</td>
      <td>26</td>
      <td>GAC72TGT</td>
      <td>D72C</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AACTACACGGATAGGT</td>
      <td>14906</td>
      <td>lib1</td>
      <td>SortSeq_bin1</td>
      <td>CGG_naive</td>
      <td>31</td>
      <td>CAA133AAA TTT137TAG</td>
      <td>Q133K F137*</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CATAATGAATGTGCAA</td>
      <td>14667</td>
      <td>lib1</td>
      <td>SortSeq_bin1</td>
      <td>CGG_naive</td>
      <td>35</td>
      <td>TAT94ATT CAG217TGG GAG232TAG</td>
      <td>Y94I Q217W E232*</td>
      <td>3</td>
      <td>3</td>
    </tr>
  </tbody>
</table>


    (7697160, 10)


Add concentrations by parsing the sample name.


```python
# parse the sample name to get the antigen concentration and bin
df_barcodes["antigen_concentration"] = np.nan
df_barcodes.loc[df_barcodes['sample'].str.contains('TiteSeq'), "antigen_concentration"] = concs[
    df_barcodes.query(f"sample.str.contains('TiteSeq')")["sample"].str.extract(r"TiteSeq_(\d+)").astype(int) - 1
]
df_barcodes["bin"] = df_barcodes["sample"].str[-1].astype(int)
```

## Filter synonymous variants with silent mutations


```python

synonymous = df_barcodes.query("n_aa_substitutions == 0 & n_codon_substitutions > 0")
print(f"There are {synonymous.shape[0]} synonymous variants with codon substitutions, dropping them.")
df_barcodes.query("~(n_aa_substitutions == 0 & n_codon_substitutions > 0)", inplace=True)

# drop columns that are not needed
df_barcodes.drop(columns=["codon_substitutions", "n_codon_substitutions", "target", "variant_call_support"], inplace=True)
df_barcodes.rename(columns={"aa_substitutions": "variant",
                            "count": "read_count"},
                   inplace=True)
df_barcodes = df_barcodes.loc[:, ["sample", "library", "variant", "n_aa_substitutions", "barcode", "antigen_concentration", "bin", "read_count"]]
df_barcodes.sort_values(by=list(df_barcodes.columns), inplace=True)
df_barcodes.variant = df_barcodes.variant.fillna("WT")

df_barcodes
```

    There are 27000 synonymous variants with codon substitutions, dropping them.





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sample</th>
      <th>library</th>
      <th>variant</th>
      <th>n_aa_substitutions</th>
      <th>barcode</th>
      <th>antigen_concentration</th>
      <th>bin</th>
      <th>read_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>42492</th>
      <td>SortSeq_bin1</td>
      <td>lib1</td>
      <td>A104C</td>
      <td>1</td>
      <td>AAAAACATCAGTTGGT</td>
      <td>NaN</td>
      <td>1</td>
      <td>17</td>
    </tr>
    <tr>
      <th>29591</th>
      <td>SortSeq_bin1</td>
      <td>lib1</td>
      <td>A104C</td>
      <td>1</td>
      <td>AAAACACTATCTAGGA</td>
      <td>NaN</td>
      <td>1</td>
      <td>36</td>
    </tr>
    <tr>
      <th>59773</th>
      <td>SortSeq_bin1</td>
      <td>lib1</td>
      <td>A104C</td>
      <td>1</td>
      <td>AAAATTCAAAATTATC</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>60801</th>
      <td>SortSeq_bin1</td>
      <td>lib1</td>
      <td>A104C</td>
      <td>1</td>
      <td>AACAAAAGTGTATGTT</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>62532</th>
      <td>SortSeq_bin1</td>
      <td>lib1</td>
      <td>A104C</td>
      <td>1</td>
      <td>AAGTTATGAATACCCT</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>7697113</th>
      <td>TiteSeq_09_bin4</td>
      <td>lib2</td>
      <td>WT</td>
      <td>0</td>
      <td>TTTTTCATGTATATGC</td>
      <td>0.0</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7697148</th>
      <td>TiteSeq_09_bin4</td>
      <td>lib2</td>
      <td>WT</td>
      <td>0</td>
      <td>TTTTTTAAAGTTCATA</td>
      <td>0.0</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7697151</th>
      <td>TiteSeq_09_bin4</td>
      <td>lib2</td>
      <td>WT</td>
      <td>0</td>
      <td>TTTTTTACCTTTACCT</td>
      <td>0.0</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7697153</th>
      <td>TiteSeq_09_bin4</td>
      <td>lib2</td>
      <td>WT</td>
      <td>0</td>
      <td>TTTTTTAGAAGCGAAG</td>
      <td>0.0</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7697157</th>
      <td>TiteSeq_09_bin4</td>
      <td>lib2</td>
      <td>WT</td>
      <td>0</td>
      <td>TTTTTTGCTGTGTATC</td>
      <td>0.0</td>
      <td>4</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>7670160 rows × 8 columns</p>
</div>



Use total cell counts and total read counts in each concentration and bin to estimate the number of cells with each barcode


```python
def normalize_read_count(df):
    library = df.library.iloc[0]
    sample = df["sample"].iloc[0]
    total_reads = df.read_count.sum()
    total_cells = barcode_runs.number_cells[(library, sample)]
    df["estimated_cell_count"] = total_cells * df.read_count / total_reads
    return df

df_barcodes = df_barcodes.groupby(["library", "sample"]).apply(normalize_read_count).reset_index(drop=True)
df_barcodes
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sample</th>
      <th>library</th>
      <th>variant</th>
      <th>n_aa_substitutions</th>
      <th>barcode</th>
      <th>antigen_concentration</th>
      <th>bin</th>
      <th>read_count</th>
      <th>estimated_cell_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>SortSeq_bin1</td>
      <td>lib1</td>
      <td>A104C</td>
      <td>1</td>
      <td>AAAAACATCAGTTGGT</td>
      <td>NaN</td>
      <td>1</td>
      <td>17</td>
      <td>0.610535</td>
    </tr>
    <tr>
      <th>1</th>
      <td>SortSeq_bin1</td>
      <td>lib1</td>
      <td>A104C</td>
      <td>1</td>
      <td>AAAACACTATCTAGGA</td>
      <td>NaN</td>
      <td>1</td>
      <td>36</td>
      <td>1.292898</td>
    </tr>
    <tr>
      <th>2</th>
      <td>SortSeq_bin1</td>
      <td>lib1</td>
      <td>A104C</td>
      <td>1</td>
      <td>AAAATTCAAAATTATC</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SortSeq_bin1</td>
      <td>lib1</td>
      <td>A104C</td>
      <td>1</td>
      <td>AACAAAAGTGTATGTT</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>SortSeq_bin1</td>
      <td>lib1</td>
      <td>A104C</td>
      <td>1</td>
      <td>AAGTTATGAATACCCT</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>7670155</th>
      <td>TiteSeq_09_bin4</td>
      <td>lib2</td>
      <td>WT</td>
      <td>0</td>
      <td>TTTTTCATGTATATGC</td>
      <td>0.0</td>
      <td>4</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>7670156</th>
      <td>TiteSeq_09_bin4</td>
      <td>lib2</td>
      <td>WT</td>
      <td>0</td>
      <td>TTTTTTAAAGTTCATA</td>
      <td>0.0</td>
      <td>4</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>7670157</th>
      <td>TiteSeq_09_bin4</td>
      <td>lib2</td>
      <td>WT</td>
      <td>0</td>
      <td>TTTTTTACCTTTACCT</td>
      <td>0.0</td>
      <td>4</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>7670158</th>
      <td>TiteSeq_09_bin4</td>
      <td>lib2</td>
      <td>WT</td>
      <td>0</td>
      <td>TTTTTTAGAAGCGAAG</td>
      <td>0.0</td>
      <td>4</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>7670159</th>
      <td>TiteSeq_09_bin4</td>
      <td>lib2</td>
      <td>WT</td>
      <td>0</td>
      <td>TTTTTTGCTGTGTATC</td>
      <td>0.0</td>
      <td>4</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>7670160 rows × 9 columns</p>
</div>



## Barcode aggregation

The current $K_D$ estimation procedure does a separate estimate for each barcode, then computes the median of the estimated $\log K_D$ across barcodes for each variant.
We should instead estimate a single $K_D$ parameter for each variants. We will do this by aggregating read counts from all barcodes for a given variant.


```python
df_variants = (
    df_barcodes
    .groupby(
        ["library", "variant", "n_aa_substitutions", "antigen_concentration", "bin", "sample"],
        dropna=False
    )
    .agg(
        {
            "read_count": "sum",
            "estimated_cell_count": "sum",
            "barcode": "count"
        }
    )
    .reset_index()
)
df_variants.sort_values(by=list(df_variants.columns), inplace=True)
df_variants
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>library</th>
      <th>variant</th>
      <th>n_aa_substitutions</th>
      <th>antigen_concentration</th>
      <th>bin</th>
      <th>sample</th>
      <th>read_count</th>
      <th>estimated_cell_count</th>
      <th>barcode</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>lib1</td>
      <td>A104C</td>
      <td>1</td>
      <td>0.000000e+00</td>
      <td>1</td>
      <td>TiteSeq_09_bin1</td>
      <td>2663</td>
      <td>1250.687875</td>
      <td>25</td>
    </tr>
    <tr>
      <th>1</th>
      <td>lib1</td>
      <td>A104C</td>
      <td>1</td>
      <td>0.000000e+00</td>
      <td>2</td>
      <td>TiteSeq_09_bin2</td>
      <td>25</td>
      <td>24.273527</td>
      <td>25</td>
    </tr>
    <tr>
      <th>2</th>
      <td>lib1</td>
      <td>A104C</td>
      <td>1</td>
      <td>0.000000e+00</td>
      <td>3</td>
      <td>TiteSeq_09_bin3</td>
      <td>0</td>
      <td>0.000000</td>
      <td>25</td>
    </tr>
    <tr>
      <th>3</th>
      <td>lib1</td>
      <td>A104C</td>
      <td>1</td>
      <td>0.000000e+00</td>
      <td>4</td>
      <td>TiteSeq_09_bin4</td>
      <td>0</td>
      <td>0.000000</td>
      <td>25</td>
    </tr>
    <tr>
      <th>4</th>
      <td>lib1</td>
      <td>A104C</td>
      <td>1</td>
      <td>1.000000e-13</td>
      <td>1</td>
      <td>TiteSeq_08_bin1</td>
      <td>578</td>
      <td>1494.790836</td>
      <td>25</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>912835</th>
      <td>lib2</td>
      <td>Y94W R145M</td>
      <td>2</td>
      <td>1.000000e-06</td>
      <td>4</td>
      <td>TiteSeq_01_bin4</td>
      <td>16</td>
      <td>12.947315</td>
      <td>1</td>
    </tr>
    <tr>
      <th>912836</th>
      <td>lib2</td>
      <td>Y94W R145M</td>
      <td>2</td>
      <td>NaN</td>
      <td>1</td>
      <td>SortSeq_bin1</td>
      <td>0</td>
      <td>0.000000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>912837</th>
      <td>lib2</td>
      <td>Y94W R145M</td>
      <td>2</td>
      <td>NaN</td>
      <td>2</td>
      <td>SortSeq_bin2</td>
      <td>14</td>
      <td>2.535604</td>
      <td>1</td>
    </tr>
    <tr>
      <th>912838</th>
      <td>lib2</td>
      <td>Y94W R145M</td>
      <td>2</td>
      <td>NaN</td>
      <td>3</td>
      <td>SortSeq_bin3</td>
      <td>57</td>
      <td>10.063316</td>
      <td>1</td>
    </tr>
    <tr>
      <th>912839</th>
      <td>lib2</td>
      <td>Y94W R145M</td>
      <td>2</td>
      <td>NaN</td>
      <td>4</td>
      <td>SortSeq_bin4</td>
      <td>112</td>
      <td>37.721971</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>912840 rows × 9 columns</p>
</div>



## Filter missing concentrations

We filter missing concentrations for each barcode in the barcode data, and for each variant in the variant data. Missing means the sum of reads is zero.


```python
def conc_filter_fn(df):
    return all(df.groupby("antigen_concentration", dropna=False).read_count.sum() > 0)
```


```python
# make a 'sample_type' column, such that we can group sample type variants accross all bins when filtering for concentrations
df_variants["sample_type"] = df_variants["sample"].str.extract(r"^(TiteSeq|SortSeq)")
df_barcodes["sample_type"] = df_barcodes["sample"].str.extract(r"^(TiteSeq|SortSeq)")
```


```python
df_barcodes = (
    df_barcodes
    .groupby(
        ["sample_type", "library", "variant", "n_aa_substitutions", "barcode"]
    )
    .filter(conc_filter_fn)
    .drop(columns=["sample_type"])
)
df_variants = (
    df_variants
    .groupby(
        ["sample_type", "library", "variant", "n_aa_substitutions"]
    )
    .filter(conc_filter_fn)
    .drop(columns=["sample_type"])
)
```


```python
# plot the distribution of 'barcodes' values across the various each of the sample groups
df_variants.groupby(["sample", "library"]).barcode.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>sample</th>
      <th>library</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">SortSeq_bin1</th>
      <th>lib1</th>
      <td>10938.0</td>
      <td>8.470104</td>
      <td>105.405655</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>15.0</td>
      <td>10986.0</td>
    </tr>
    <tr>
      <th>lib2</th>
      <td>11686.0</td>
      <td>8.464060</td>
      <td>114.128714</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>15.0</td>
      <td>12299.0</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">SortSeq_bin2</th>
      <th>lib1</th>
      <td>10938.0</td>
      <td>8.470104</td>
      <td>105.405655</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>15.0</td>
      <td>10986.0</td>
    </tr>
    <tr>
      <th>lib2</th>
      <td>11686.0</td>
      <td>8.464060</td>
      <td>114.128714</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>15.0</td>
      <td>12299.0</td>
    </tr>
    <tr>
      <th>SortSeq_bin3</th>
      <th>lib1</th>
      <td>10938.0</td>
      <td>8.470104</td>
      <td>105.405655</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>15.0</td>
      <td>10986.0</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>TiteSeq_09_bin2</th>
      <th>lib2</th>
      <td>10200.0</td>
      <td>9.548922</td>
      <td>122.122509</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>17.0</td>
      <td>12299.0</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">TiteSeq_09_bin3</th>
      <th>lib1</th>
      <td>9437.0</td>
      <td>9.654551</td>
      <td>113.434854</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>16.0</td>
      <td>10986.0</td>
    </tr>
    <tr>
      <th>lib2</th>
      <td>10200.0</td>
      <td>9.548922</td>
      <td>122.122509</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>17.0</td>
      <td>12299.0</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">TiteSeq_09_bin4</th>
      <th>lib1</th>
      <td>9437.0</td>
      <td>9.654551</td>
      <td>113.434854</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>16.0</td>
      <td>10986.0</td>
    </tr>
    <tr>
      <th>lib2</th>
      <td>10200.0</td>
      <td>9.548922</td>
      <td>122.122509</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>17.0</td>
      <td>12299.0</td>
    </tr>
  </tbody>
</table>
<p>80 rows × 8 columns</p>
</div>




```python
df_barcodes.to_csv(config['prepped_barcode_counts_file'], index=False)
df_variants.to_csv(config['prepped_variant_counts_file'], index=False)
```
