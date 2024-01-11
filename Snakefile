"""``snakemake`` file that runs entire analysis."""

# Imports ---------------------------------------------------------------------
import glob
import itertools
import os.path
import os
import textwrap
import urllib.request
import pandas as pd

# Configuration  --------------------------------------------------------------
configfile: 'config.yaml'

# run "quick" rules locally:
# localrules: make_dag,
#             make_summary

# Functions -------------------------------------------------------------------
def nb_markdown(nb):
    """Return path to Markdown results of notebook `nb`."""
    return os.path.join(config['summary_dir'],
                        os.path.basename(os.path.splitext(nb)[0]) + '.md')

# Global variables extracted from config --------------------------------------
pacbio_runs = (pd.read_csv(config['pacbio_runs'], dtype = str)
               .assign(pacbioRun=lambda x: x['library'] + '_' + x['run'])
               )
assert len(pacbio_runs['pacbioRun'].unique()) == len(pacbio_runs['pacbioRun'])

# Information on samples and barcode runs -------------------------------------
barcode_runs = pd.read_csv(config['barcode_runs'])

# Rules -----------------------------------------------------------------------

# making this summary is the target rule (in place of `all`) since it
# is first rule listed.
rule make_summary:
    """Create Markdown summary of analysis."""
    input:
        dag='dag.png',
        process_ccs=nb_markdown('process_ccs.ipynb'),
        barcode_variant_table=config['codon_variant_table_file'],
        variant_counts_file=config['variant_counts_file'],
        count_variants=nb_markdown('count_variants.ipynb'),
        prepped_barcode_counts_file=config['prepped_barcode_counts_file'],
        prepped_variant_counts_file=config['prepped_variant_counts_file'],
        normalize_filter_aggregate_barcodes=nb_markdown('normalize_filter_aggregate_barcodes.ipynb'),
        Titeseq_modeling=nb_markdown('Titeseq-modeling.ipynb'),
        variant_Kds_file=config['Titeseq_Kds_file'],
        calculate_expression='results/summary/compute_expression_meanF.md',
        variant_expression_file=config['expression_sortseq_file'],
        collapse_scores='results/summary/collapse_scores.md',
        mut_phenos_file=config['final_variant_scores_mut_file'],
        structural_mapping='results/summary/structural_mapping.md',
        
    output:
        summary = os.path.join(config['summary_dir'], 'summary.md')
    # log:
        # os.path.join(config['summary_dir'], 'summary.log')
    run:
        def path(f):
            """Get path relative to `summary_dir`."""
            return os.path.relpath(f, config['summary_dir'])
        with open(output.summary, 'w') as f:
            f.write(textwrap.dedent(f"""
            # Summary

            Analysis run by [Snakefile]({path(workflow.snakefile)})
            using [this config file]({path(workflow.configfiles[0])}).
            See the [README in the top directory]({path('README.md')})
            for details.

            Here is the DAG of the computational workflow:
            ![{path(input.dag)}]({path(input.dag)})

            Here is the Markdown output of each analysis step in the
            workflow:
            
            1. [Process PacBio CCSs]({path(input.process_ccs)}). Creates a [barcode-variant lookup table]({path(input.barcode_variant_table)}).
            
            2. [Count variants by barcode]({path(input.count_variants)}). Creates a [variant counts file]({path(input.variant_counts_file)}) giving counts of each barcoded variant in each condition.

            3. [Normalize, filter, aggregate barcodes]({path(input.normalize_filter_aggregate_barcodes)}) produces [prepped barcode counts]({path(input.prepped_barcode_counts_file)}) and [prepped variant counts]({path(input.prepped_variant_counts_file)}). These are the barcode and variant counts after merging substitution annotations, normalizing counts, filtering variants, and aggregating (for variant counts) barcode counts.

            4. [Tite-seq modeling]({path(input.Titeseq_modeling)}). This notebook fits a model to the Tite-seq data to estimate the binding affinity (Kd) of each variant to the CGG antibody. The results are recorded in the [variant Kds file]({path(input.variant_Kds_file)}).
            
            5. [Analyze Sort-seq]({path(input.calculate_expression)}) to calculate per-variant RBD expression, recorded in [this file]({path(input.variant_expression_file)}).

            6. [Collapse scores]({path(input.collapse_scores)}) merges and analyzes the phenotype data. The results are recorded in the final variant scores mut file [here]({path(input.mut_phenos_file)}).
                           
            7. [Map DMS phenotypes to the CGG-bound antibody structure]({path(input.structural_mapping)}).

            """
            ).strip())


rule structural_mapping:
    input:
        config['final_variant_scores_mut_file'],
        config['CGGnaive_site_info'],
        config['pdb']
    output:
        md='results/summary/structural_mapping.md'
    conda:
        'envs/R.yml'
    params:
        nb='structural_mapping.Rmd',
        md='structural_mapping.md'
    log:
        'results/logs/structural_mapping.log'
    shell:
        """
        R -e \"rmarkdown::render(input=\'{params.nb}\')\" &> {log};
        mv {params.md} {output.md}
        """

rule collapse_scores:
    input:
        config['Titeseq_Kds_file'],
        config['expression_sortseq_file'],
        config['CGGnaive_site_info']
    output:
        config['final_variant_scores_mut_file'],
        md='results/summary/collapse_scores.md',
        md_files=directory('results/summary/collapse_scores_files')
    conda:
        'envs/R.yml'
    params:
        nb='collapse_scores.Rmd',
        md='collapse_scores.md',
        md_files='collapse_scores_files'
    log:
        'results/logs/collapse_scores.log'
    shell:
        """
        R -e \"rmarkdown::render(input=\'{params.nb}\')\" &> {log};
        mv {params.md} {output.md};
        mv {params.md_files} {output.md_files}
        """

rule Titeseq_modeling:
    input:
        config['prepped_variant_counts_file'],
        config['barcode_runs'],
        config['CGGnaive_site_info'],
        facs_specimen_data = [f for f in glob.glob(config['facs_file_pattern'])]
    output:
        config['Titeseq_Kds_file'],
        nb_markdown=nb_markdown('Titeseq-modeling.ipynb'),
        md_files=directory('results/summary/Titeseq-modeling_files')
    conda:
        'envs/Titeseq_modeling.yml'
    params:
        nb='Titeseq-modeling.ipynb',
    log:
        'results/logs/Titeseq_modeling.log'
    shell:
        """
        python scripts/run_nb.py {params.nb} {output.nb_markdown} &> {log}
        """


rule calculate_expression:
    input:
        config['codon_variant_table_file'],
        config['prepped_variant_counts_file']
    output:
        config['expression_sortseq_file'],
        md='results/summary/compute_expression_meanF.md',
        md_files=directory('results/summary/compute_expression_meanF_files')
    conda:
        'envs/R.yml'
    params:
        nb='compute_expression_meanF.Rmd',
        md='compute_expression_meanF.md',
        md_files='compute_expression_meanF_files'
    log:
        'results/logs/calculate_expression.log'
    shell:
        """
        R -e \"rmarkdown::render(input=\'{params.nb}\')\" &> {log};
        mv {params.md} {output.md};
        mv {params.md_files} {output.md_files}
        """


rule normalize_filter_aggregate_barcodes:
    """
    Merge annotations, normalize counts, filter variants, 
    and aggregate barcode counts for both TiteSeq, and SortSeq data.
    """
    input:
        config['codon_variant_table_file'],
        config['variant_counts_file'],
        config['barcode_runs']
    output:
        config['prepped_barcode_counts_file'],
        config['prepped_variant_counts_file'],
        nb_markdown=nb_markdown('normalize_filter_aggregate_barcodes.ipynb')
    conda:
        'envs/normalize_filter_aggregate_barcodes.yml'        
    params:
        nb='normalize-filter-aggregate-barcodes.ipynb'
    log:
        'results/logs/normalize_filter_aggregate_barcodes.log'
    shell:
        """
        python scripts/run_nb.py {params.nb} {output.nb_markdown} &> {log}
        """

if config['seqdata_source'] == 'HutchServer' and config['run_from_ngs']:
    # TODO : conda env
    rule count_variants:
        """Count codon variants from Illumina barcode runs."""
        input:
            config['codon_variant_table_file'],
            config['barcode_runs']
        output:
            config['variant_counts_file'],
            nb_markdown=nb_markdown('count_variants.ipynb')
        params:
            nb='count_variants.ipynb'
        log:
            'results/logs/count_variants.log'
        shell:
            """
            python scripts/run_nb.py {params.nb} {output.nb_markdown} &> {log}
            """
    # TODO conda env
    rule process_ccs:
        """Process the PacBio CCSs and build variant table."""
        input:
            expand(os.path.join(config['ccs_dir'], "{pacbioRun}_ccs.fastq.gz"), pacbioRun=pacbio_runs['pacbioRun']),
        output:
            config['processed_ccs_file'],
            config['codon_variant_table_file'],
            nb_markdown=nb_markdown('process_ccs.ipynb')
        params:
            nb='process_ccs.ipynb'
        log:
            'results/logs/process_ccs.log'
        shell:
            """
            python scripts/run_nb.py {params.nb} {output.nb_markdown} &> {log}
            """


    rule build_ccs:
        """Run PacBio ``ccs`` program to build CCSs from subreads."""
        input:
            subreads=lambda wildcards: (pacbio_runs
                                        .set_index('pacbioRun')
                                        .at[wildcards.pacbioRun, 'subreads']
                                        )
        output:
            ccs_report=os.path.join(config['ccs_dir'], "{pacbioRun}_report.txt"),
            ccs_fastq=os.path.join(config['ccs_dir'], "{pacbioRun}_ccs.fastq.gz")
        params:
            min_ccs_length=config['min_ccs_length'],
            max_ccs_length=config['max_ccs_length'],
            min_ccs_passes=config['min_ccs_passes'],
            min_ccs_accuracy=config['min_ccs_accuracy']
        threads: config['max_cpus']
        shell:
            """
            {
                ccs \
                    --min-length {params.min_ccs_length} \
                    --max-length {params.max_ccs_length} \
                    --min-passes {params.min_ccs_passes} \
                    --min-rq {params.min_ccs_accuracy} \
                    --report-file {output.ccs_report} \
                    --num-threads {threads} \
                    {input.subreads} \
                {output.ccs_fastq}
            } &> {output.ccs_report}.log
            """

elif config['seqdata_source'] == 'SRA':
    raise RuntimeError('getting sequence data from SRA not yet implemented')

else:
    pass    
