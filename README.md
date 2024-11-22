# Deep mutational scanning of the CGGnaive scFv

Analysis of deep mutational scanning of barcoded codon variants of CGGnaive antibody.

Study and analysis by Tyler Starr, [Jesse Bloom](https://research.fhcrc.org/bloom/en.html), and co-authors.

Code refactored to integrate improved Kd modeling by Jared Galloway, and Will DeWitt.

## Summary of workflow and results

For a summary of the workflow and links to key results files, [click here](results/summary/summary.md).
Reading this summary is the best way to understand the analysis.

## Running the analysis on a single machine (recommended)

Note the sequencing data is not included in this repository and thus the processing of the ccs and the variants counts step are not able to be run from the data. We track the output files for these steps, and thus you can run the pipeline to completion downstream of these steps.

We let `snakemake` conda functionality handle the environment setup, so all you need is an environment with `snakemake` and `git-lfs` installed. We reccomend following [these instructions first](https://snakemake.readthedocs.io/en/stable/getting_started/installation.html#installation-via-conda-mamba), then running the following commands:

```
conda activate snakemake
conda install git-lfs
git clone https://github.com/jbloomlab/Ab-CGGnaive_DMS.git
cd Ab-CGGnaive_DMS
git lfs install
```

Then you can run the pipeline with the following command:

```
snakemake -j8 --use-conda
```


### Configure `.git` to not track Jupyter notebook metadata
To simplify git tracking of Jupyter notebooks, we have added the filter described [here](https://stackoverflow.com/questions/28908319/how-to-clear-an-ipython-notebooks-output-in-all-cells-from-the-linux-terminal/58004619#58004619) to strip notebook metadata to [.gitattributes](.gitattributes) and [.gitconfig](.gitconfig).
The **first time** you check out this repo, run the following command to use this configuration (see [here](https://stackoverflow.com/a/18330114)):
```
   git config --local include.path ../.gitconfig
```
Then don't worry about it anymore.

## Configuring the analysis
The configuration for the analysis is specifed in [config.yaml](config.yaml).
This file defines key variables for the analysis, and should be relatively self-explanatory.
You should modify the analysis by changing this configuration file; do **not** hard-code crucial experiment-specific variables within the Jupyter notebooks or `Snakefile`.

The input files pointed to by [config.yaml](config.yaml) are in the [./data/](data) subdirectory.
See the [./data/README.md](./data/README.md) file for details.


## Cluster configuration (untested since [#13](https://github.com/jbloomlab/Ab-CGGnaive_DMS/pull/13))

To run using the cluster configuration for the Fred Hutch server, simply run the bash script [run_Hutch_cluster.bash](run_Hutch_cluster.bash), which executes [Snakefile](Snakefile) in a way that takes advantage of the Hutch server resources.
This bash script also automates the environment building steps above, so really all you have to do is run this script.
You likely want to submit [run_Hutch_cluster.bash](run_Hutch_cluster.bash) itself to the cluster (since it takes a while to run) with:

    sbatch -t 7-0 run_Hutch_cluster.bash

There is a cluster configuration file [cluster.yaml](cluster.yaml) that configures [Snakefile](Snakefile) for the Fred Hutch cluster, as recommended by the [Snakemake documentation](https://snakemake.readthedocs.io/en/stable/snakefiles/configuration.html).
The [run_Hutch_cluster.bash](run_Hutch_cluster.bash) script uses this configuration to run [Snakefile](Snakefile).
If you are using a different cluster than the Fred Hutch one, you may need to modify the cluster configuration file.

## Notebooks that perform the analysis
The Jupyter notebooks and R markdown dscripts that perform most of the analysis are in this top-level directory with the extension `*.ipynb` or `*.Rmd`.
These notebooks read the key configuration values from [config.yaml](config.yaml).

There is also a [./scripts/](scripts) subdirectory with related scripts.

The notebooks need to be run in the order described in [the workflow and results summary](results/summary/summary.md).
This will occur automatically if you run them via [Snakefile](Snakefile) as described above.

## Results
Results are placed in the [./results/](results) subdirectory.
Many of the files created in this subdirectory are not tracked in the `git` repo as they are very large.
However, key results files are tracked as well as a summary that shows the code and results.
Click [here](./results/summary/summary.md) to see that summary.

The large results files are tracked via [git-lfs](https://git-lfs.github.com/).
This requires `git-lfs` to be installed, which it is in the `conda` environment specified by [environment.yml](environment.yml).
The following commands were then run:

    git lfs install

You may need to run this if you are tracking these files and haven't installed `git-lfs` in your user account.
Then the large results files were added for tracking with:
```
git lfs track results/variants/codon_variant_table.csv
git lfs track results/counts/variant_counts.csv
git lfs track results/binding_Kd/bc_binding.csv
git lfs track results/final_variant_scores/final_variant_scores.csv
```

