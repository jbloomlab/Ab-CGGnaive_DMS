"""``snakemake`` file that runs entire analysis."""

# Imports ---------------------------------------------------------------------
import glob
import itertools
import os.path
import os
import textwrap
import urllib.request

# TODO make logging a thing for the entire workflow
# TODO remove the pipeline need to run the first two rules, process_ccs and count_variants
# TODO add the dag argument to the snakemake call
# TODO create a container based up this environment, maybe?
# TODO 

# import Bio.SeqIO

# import dms_variants.codonvarianttable
# import dms_variants.illuminabarcodeparser

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
        dag='dag.svg',
        process_ccs=nb_markdown('process_ccs.ipynb'),
        barcode_variant_table=config['codon_variant_table_file'],
        variant_counts_file=config['variant_counts_file'],
        count_variants=nb_markdown('count_variants.ipynb'),
        prepped_barcode_counts_file=config['prepped_barcode_counts_file'],
        prepped_variant_counts_file=config['prepped_variant_counts_file'],
        prep_Titeseq_barcodes=nb_markdown('prep_Titeseq_barcodes.ipynb'),
        Titeseq_modeling=nb_markdown('Titeseq-modeling.ipynb'),
        variant_Kds_file=config['Titeseq_Kds_file'],
        calculate_expression='results/summary/compute_expression_meanF.md',
        variant_expression_file=config['expression_sortseq_file'],
        collapse_scores='results/summary/collapse_scores.md',
        mut_phenos_file=config['final_variant_scores_mut_file'],
        structural_mapping='results/summary/structural_mapping.md',
        
    output:
        summary = os.path.join(config['summary_dir'], 'summary.md')
    log:
        os.path.join(config['summary_dir'], 'summary.log')
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
            
            1.  [Process PacBio CCSs]({path(input.process_ccs)}). Creates a [barcode-variant lookup table]({path(input.barcode_variant_table)}).
            
            2.  [Count variants by barcode]({path(input.count_variants)}). Creates a [variant counts file]({path(input.variant_counts_file)}) giving counts of each barcoded variant in each condition.

            3.  [Prep Titseq Barcodes]({path(input.prep_Titeseq_barcodes)}) produces [prepped barcode counts]({path(input.prepped_barcode_counts_file)}) and [prepped variant counts]({path(input.prepped_variant_counts_file)}). These are the barcode and variant counts after merging substitution annotations, normalizing counts, filtering variants, and aggregating (for variant counts) barcode counts.

            4.  [Tite-seq modeling]({path(input.Titeseq_modeling)}). This notebook fits a model to the Tite-seq data to estimate the binding affinity of each variant to the CGG antibody. The results are recorded in [this file]({path(input.final_variant_scores_mut_file)}).

            5.  [Analyze Sort-seq]({path(input.calculate_expression)}) to calculate per-barcode RBD expression, recorded in [this file]({path(input.variant_expression_file)}).
                           
            6. [Map DMS phenotypes to the CGG-bound antibody structure]({path(input.structural_mapping)}).

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
    shell:
        """
        R -e \"rmarkdown::render(input=\'{params.nb}\')\";
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
    shell:
        """
        R -e \"rmarkdown::render(input=\'{params.nb}\')\";
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
        md_files=directory('results/summary/compute_binding_Kd_files')
    conda:
        'envs/Titeseq_modeling.yml'
    params:
        nb='Titeseq-modeling.ipynb',
    shell:
        """
        export OUTPUT_DIR={output.md_files}
        python scripts/run_nb.py {params.nb} {output.nb_markdown}
        """

# TODO remove after you finish updating the Titeseq_modeling rule
# rule fit_titrations:
#     input:
#         config['codon_variant_table_file'],
#         config['variant_counts_file']
#     output:
#         config['Titeseq_Kds_file'],
#         md='results/summary/compute_binding_Kd.md',
#         md_files=directory('results/summary/compute_binding_Kd_files')
#     # envmodules:
#         # 'R/3.6.2-foss-2019b'
#     conda:
#         'envs/R.yml'
#     params:
#         nb='compute_binding_Kd.Rmd',
#         md='compute_binding_Kd.md',
#         md_files='compute_binding_Kd_files'
#     shell:
#         """
#         R -e \"rmarkdown::render(input=\'{params.nb}\')\";
#         mv {params.md} {output.md};
#         mv {params.md_files} {output.md_files}
#         """

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
    shell:
        """
        R -e \"rmarkdown::render(input=\'{params.nb}\')\";
        mv {params.md} {output.md};
        mv {params.md_files} {output.md_files}
        """

rule prep_Titeseq_barcodes:
    """
    Merge annotations, normalize counts, filter variants, 
    and aggregate barcode counts.
    """
    input:
        config['codon_variant_table_file'],
        config['variant_counts_file'],
        config['barcode_runs']
    output:
        config['prepped_barcode_counts_file'],
        config['prepped_variant_counts_file'],
        nb_markdown=nb_markdown('prep_Titeseq_barcodes.ipynb')
    conda:
        'envs/prep_Titeseq_barcodes.yml'        
    params:
        nb='prep-Titeseq-barcodes.ipynb'
    shell:
        "python scripts/run_nb.py {params.nb} {output.nb_markdown}"

if config['seqdata_source'] == 'HutchServer' and config['run_from_ngs']:

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
        shell:
            "python scripts/run_nb.py {params.nb} {output.nb_markdown}"

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
        shell:
            "python scripts/run_nb.py {params.nb} {output.nb_markdown}"


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
            ccs \
                --min-length {params.min_ccs_length} \
                --max-length {params.max_ccs_length} \
                --min-passes {params.min_ccs_passes} \
                --min-rq {params.min_ccs_accuracy} \
                --report-file {output.ccs_report} \
                --num-threads {threads} \
                {input.subreads} \
                {output.ccs_fastq}
            """

elif config['seqdata_source'] == 'SRA':
    raise RuntimeError('getting sequence data from SRA not yet implemented')

else:
    # raise ValueError(f"invalid `seqdata_source` {config['seqdata_source']}")
    pass    
