#!/bin/bash

snakemake \
    -j 999 \
    --cluster-config cluster.yaml \
    --cluster "sbatch -A matsen_e -c {cluster.cpus} --mem={cluster.mem} -t {cluster.time} -J {cluster.name} --constraint={cluster.constraint}" \
    --latency-wait 30 \
    --use-envmodules \
    --use-conda \
    --conda-prefix ./env \
    --conda-frontend conda \
    -R make_summary
