Map DMS to the chIgY-Fab sturcture
================
Tyler Starr
4/29/2022

- <a href="#data-input" id="toc-data-input">Data input</a>
- <a
  href="#compute-site-wise-metrics-from-dms-data-for-mapping-to-structure"
  id="toc-compute-site-wise-metrics-from-dms-data-for-mapping-to-structure">Compute
  site-wise metrics from DMS data for mapping to structure</a>
- <a href="#map-metrics-to-the-structure"
  id="toc-map-metrics-to-the-structure">Map metrics to the structure</a>

This notebook analyzes the structure of the CGGnaive Fab bound to the
chicken IgY dimer. It generates a list of inter-residue distances and
maps sitewise DMS properties to the B-factor column for visualization in
PyMol.

``` r
require("knitr")
knitr::opts_chunk$set(echo = T)
knitr::opts_chunk$set(dev.args = list(png = list(type = "cairo")))
options(repos = c(CRAN = "https://cran.r-project.org"))

#list of packages to install/load
packages = c("yaml","data.table","tidyverse","gridExtra","bio3d","ggrepel")
#install any packages not already installed
installed_packages <- packages %in% rownames(installed.packages())
if(any(installed_packages == F)){
  install.packages(packages[!installed_packages])
}
#load packages
invisible(lapply(packages, library, character.only=T))

#read in config file
config <- read_yaml("config.yaml")

#make output directory
if(!file.exists(config$structural_mapping_dir)){
  dir.create(file.path(config$structural_mapping_dir))
}
```

Session info for reproducing environment:

``` r
sessionInfo()
```

    ## R version 3.6.3 (2020-02-29)
    ## Platform: x86_64-conda-linux-gnu (64-bit)
    ## Running under: Ubuntu 18.04.6 LTS
    ## 
    ## Matrix products: default
    ## BLAS/LAPACK: /fh/fast/matsen_e/jgallowa/Ab-CGGnaive_DMS/.snakemake/conda/14854db9156898a213246f7d6480a8f3_/lib/libopenblasp-r0.3.28.so
    ## 
    ## locale:
    ##  [1] LC_CTYPE=en_US.UTF-8       LC_NUMERIC=C              
    ##  [3] LC_TIME=en_US.UTF-8        LC_COLLATE=en_US.UTF-8    
    ##  [5] LC_MONETARY=en_US.UTF-8    LC_MESSAGES=en_US.UTF-8   
    ##  [7] LC_PAPER=en_US.UTF-8       LC_NAME=C                 
    ##  [9] LC_ADDRESS=C               LC_TELEPHONE=C            
    ## [11] LC_MEASUREMENT=en_US.UTF-8 LC_IDENTIFICATION=C       
    ## 
    ## attached base packages:
    ## [1] stats     graphics  grDevices utils     datasets  methods   base     
    ## 
    ## other attached packages:
    ##  [1] ggrepel_0.9.6     bio3d_2.4-5       gridExtra_2.3     forcats_0.5.1    
    ##  [5] stringr_1.4.0     dplyr_1.0.6       purrr_0.3.4       readr_1.4.0      
    ##  [9] tidyr_1.1.3       tibble_3.1.2      ggplot2_3.3.3     tidyverse_1.3.1  
    ## [13] data.table_1.14.0 yaml_2.2.1        knitr_1.33       
    ## 
    ## loaded via a namespace (and not attached):
    ##  [1] tidyselect_1.1.1  xfun_0.23         haven_2.4.1       colorspace_2.0-1 
    ##  [5] vctrs_0.3.8       generics_0.1.0    htmltools_0.5.1.1 utf8_1.2.1       
    ##  [9] rlang_0.4.11      pillar_1.6.1      glue_1.4.2        withr_3.0.2      
    ## [13] DBI_1.1.1         dbplyr_2.1.1      modelr_0.1.8      readxl_1.3.1     
    ## [17] lifecycle_1.0.0   munsell_0.5.0     gtable_0.3.0      cellranger_1.1.0 
    ## [21] rvest_1.0.0       evaluate_0.14     ps_1.6.0          parallel_3.6.3   
    ## [25] fansi_0.4.2       broom_0.7.6       Rcpp_1.0.13-1     scales_1.1.1     
    ## [29] backports_1.2.1   jsonlite_1.7.2    fs_1.5.0          hms_1.1.0        
    ## [33] digest_0.6.27     stringi_1.6.2     grid_3.6.3        cli_2.5.0        
    ## [37] tools_3.6.3       magrittr_2.0.1    crayon_1.4.1      pkgconfig_2.0.3  
    ## [41] ellipsis_0.3.2    xml2_1.3.2        reprex_2.0.0      lubridate_1.7.10 
    ## [45] assertthat_0.2.1  rmarkdown_2.8     httr_1.4.2        rstudioapi_0.13  
    ## [49] R6_2.5.0          compiler_3.6.3

## Data input

Read in tables of mutant measurements.

``` r
dt <- data.table(read.csv(file=config$final_variant_scores_mut_file,stringsAsFactors=F))
```

Read in the structure pdb

``` r
sites <- read.csv(file=config$CGGnaive_site_info,stringsAsFactors=F)
```

Read in the table giving annotations and information on CGGnaive sites

``` r
pdb <- read.pdb(file=config$pdb)
```

## Compute site-wise metrics from DMS data for mapping to structure

Want to map the mean effect of mutations on each of the three properties
to structure. Also plot the *max* effect of any mutation at a site on
binding as a metric of evolvability. Only include mutations with n_bc \>
3 for these properties.

``` r
sites$mean_CGG_bind <- NA
sites$max_CGG_bind <- NA
sites$quartile75_CGG_bind <- NA
sites$mean_expression <- NA
# sites$mean_polyspecificity <- NA
  
for(i in 1:nrow(sites)){
  if(sites[i,"chain"] != "link"){
    sites$mean_CGG_bind[i] <- mean(dt[position_IMGT==sites[i,"site"] & chain==sites[i,"chain"] & wildtype != mutant & n_bc_bind_CGG>3,delta_bind_CGG],na.rm=T)
    sites$max_CGG_bind[i] <- max(dt[position_IMGT==sites[i,"site"] & chain==sites[i,"chain"] & wildtype != mutant & n_bc_bind_CGG>3,delta_bind_CGG],na.rm=T)
    sites$quartile75_CGG_bind[i] <- quantile(dt[position_IMGT==sites[i,"site"] & chain==sites[i,"chain"] & wildtype != mutant & n_bc_bind_CGG>3,delta_bind_CGG],0.75,na.rm=T)
    sites$mean_expression[i] <- mean(dt[position_IMGT==sites[i,"site"] & chain==sites[i,"chain"] & wildtype != mutant & n_bc_expr>3,delta_expr],na.rm=T)
    # sites$mean_polyspecificity[i] <- mean(dt[position_IMGT==sites[i,"site"] & chain==sites[i,"chain"] & wildtype != mutant & n_bc_psr>3,delta_psr],na.rm=T)
  }
}
```

## Map metrics to the structure

``` r
b_mean_CGG <- rep(0, length(pdb$atom$b))
b_max_CGG <- rep(0, length(pdb$atom$b))
b_75th_CGG <- rep(0, length(pdb$atom$b))
b_mean_expr <- rep(0, length(pdb$atom$b))
# b_mean_psr <- rep(0, length(pdb$atom$b))
for(i in 1:nrow(pdb$atom)){
  res <- pdb$atom$resno[i]
  if(pdb$atom$chain[i] %in% c("B","H") & res %in% sites[sites$chain=="H","site"]){
    b_mean_CGG[i] <- sites[sites$site==res & sites$chain=="H","mean_CGG_bind"]
    b_max_CGG[i] <- sites[sites$site==res & sites$chain=="H","max_CGG_bind"]
    b_75th_CGG[i] <- sites[sites$site==res & sites$chain=="H","quartile75_CGG_bind"]
    b_mean_expr[i] <- sites[sites$site==res & sites$chain=="H","mean_expression"]
    # b_mean_psr[i] <- sites[sites$site==res & sites$chain=="H","mean_polyspecificity"]
  }else if(pdb$atom$chain[i] %in% c("C","L") & res %in% sites[sites$chain=="L","site"]){
    b_mean_CGG[i] <- sites[sites$site==res & sites$chain=="L","mean_CGG_bind"]
    b_max_CGG[i] <- sites[sites$site==res & sites$chain=="L","max_CGG_bind"]
    b_75th_CGG[i] <- sites[sites$site==res & sites$chain=="L","quartile75_CGG_bind"]
    b_mean_expr[i] <- sites[sites$site==res & sites$chain=="L","mean_expression"]
    # b_mean_psr[i] <- sites[sites$site==res & sites$chain=="L","mean_polyspecificity"]
  }
}
write.pdb(pdb=pdb, file=paste(config$structural_mapping_dir,"/CGG_mean.pdb",sep=""), b=b_mean_CGG)
write.pdb(pdb=pdb, file=paste(config$structural_mapping_dir,"/CGG_max.pdb",sep=""), b=b_max_CGG)
write.pdb(pdb=pdb, file=paste(config$structural_mapping_dir,"/CGG_75th.pdb",sep=""), b=b_75th_CGG)
write.pdb(pdb=pdb, file=paste(config$structural_mapping_dir,"/expression_mean.pdb",sep=""), b=b_mean_expr)
# write.pdb(pdb=pdb, file=paste(config$structural_mapping_dir,"/psr_mean.pdb",sep=""), b=b_mean_psr)
```

Generate a list of contact residues (5A or closer contacts)

Below are heavy chain residues that are 5A or closer to CGG ligand in
each of the two protomers:

``` r
binding.site(pdb,
             a.inds=atom.select(pdb,chain=c("H","B")),
             b.inds=atom.select(pdb,chain=c("A","D")),
             cutoff=5,hydrogens=F)$resnames
```

    ##  [1] "SER 36 (H)"  "TYR 38 (H)"  "TYR 55 (H)"  "SER 57 (H)"  "TYR 58 (H)" 
    ##  [6] "TYR 66 (H)"  "ARG 106 (H)" "ASP 107 (H)" "ASP 116 (H)" "SER 36 (B)" 
    ## [11] "GLY 37 (B)"  "TYR 38 (B)"  "TYR 55 (B)"  "SER 57 (B)"  "TYR 58 (B)" 
    ## [16] "SER 59 (B)"  "TYR 66 (B)"  "ARG 106 (B)" "ASP 107 (B)" "PHE 115 (B)"
    ## [21] "ASP 116 (B)"

Below are light chain residues that are 5A or closer to CGG ligand in
each of the two protomers:

``` r
binding.site(pdb,
             a.inds=atom.select(pdb,chain=c("L","C")),
             b.inds=atom.select(pdb,chain=c("A","D")),
             cutoff=5,hydrogens=F)$resnames
```

    ##  [1] "ASN 38 (L)"  "TYR 55 (L)"  "SER 56 (L)"  "ALA 57 (L)"  "TYR 66 (L)" 
    ##  [6] "TYR 68 (L)"  "SER 69 (L)"  "TYR 107 (L)" "TYR 108 (L)" "SER 109 (L)"
    ## [11] "TYR 114 (L)" "PRO 115 (L)" "ASN 38 (C)"  "TYR 55 (C)"  "SER 56 (C)" 
    ## [16] "TYR 66 (C)"  "ARG 67 (C)"  "TYR 68 (C)"  "SER 69 (C)"  "TYR 107 (C)"
    ## [21] "TYR 108 (C)" "SER 109 (C)" "TYR 114 (C)" "PRO 115 (C)"

Below are heavy chain residues that are 5A or closer to light chain in
each of the two protomers

``` r
binding.site(pdb,
             a.inds=atom.select(pdb,chain=c("H","B")),
             b.inds=atom.select(pdb,chain=c("L","C")),
             cutoff=5,hydrogens=F)$resnames
```

    ##  [1] "VAL 2 (H)"   "ASN 40 (H)"  "ILE 42 (H)"  "LYS 44 (H)"  "ASN 48 (H)" 
    ##  [6] "LEU 50 (H)"  "GLU 51 (H)"  "TYR 52 (H)"  "TYR 55 (H)"  "TYR 66 (H)" 
    ## [11] "ASN 68 (H)"  "PRO 69 (H)"  "TYR 103 (H)" "ASP 107 (H)" "PHE 115 (H)"
    ## [16] "ASP 116 (H)" "VAL 117 (H)" "TRP 118 (H)" "GLY 119 (H)" "ALA 120 (H)"
    ## [21] "ASN 40 (B)"  "ILE 42 (B)"  "LYS 44 (B)"  "ASN 48 (B)"  "LEU 50 (B)" 
    ## [26] "GLU 51 (B)"  "TYR 52 (B)"  "TYR 55 (B)"  "TYR 66 (B)"  "ASN 68 (B)" 
    ## [31] "TYR 103 (B)" "ASP 107 (B)" "PHE 115 (B)" "ASP 116 (B)" "TRP 118 (B)"
    ## [36] "GLY 119 (B)" "ALA 120 (B)"

Below are light chain residues that are 5A or closer to heavy chain in
each of the two protomers

``` r
binding.site(pdb,
             a.inds=atom.select(pdb,chain=c("L","C")),
             b.inds=atom.select(pdb,chain=c("H","B")),
             cutoff=5,hydrogens=F)$resnames
```

    ##  [1] "PHE 42 (L)"  "GLN 44 (L)"  "SER 49 (L)"  "PRO 50 (L)"  "LYS 51 (L)" 
    ##  [6] "SER 52 (L)"  "TYR 55 (L)"  "TYR 68 (L)"  "GLU 101 (L)" "PHE 103 (L)"
    ## [11] "HIS 105 (L)" "GLN 106 (L)" "TYR 107 (L)" "TYR 114 (L)" "PRO 115 (L)"
    ## [16] "LEU 116 (L)" "PHE 118 (L)" "SER 120 (L)" "GLY 121 (L)" "PHE 42 (C)" 
    ## [21] "GLN 44 (C)"  "GLN 48 (C)"  "SER 49 (C)"  "PRO 50 (C)"  "SER 52 (C)" 
    ## [26] "TYR 55 (C)"  "GLU 101 (C)" "PHE 103 (C)" "HIS 105 (C)" "GLN 106 (C)"
    ## [31] "TYR 107 (C)" "TYR 114 (C)" "PRO 115 (C)" "LEU 116 (C)" "PHE 118 (C)"
    ## [36] "SER 120 (C)" "GLY 121 (C)"
