Map DMS to the chIgY-Fab sturcture
================
Tyler Starr
4/29/2022

-   [Data input](#data-input)
-   [Compute site-wise metrics from DMS data for mapping to
    structure](#compute-site-wise-metrics-from-dms-data-for-mapping-to-structure)
-   [Map metrics to the structure](#map-metrics-to-the-structure)

This notebook analyzes the structure of the CGGnaive Fab bound to the
chicken IgY dimer. It generates a list of inter-residue distances and
maps sitewise DMS properties to the B-factor column for visualization in
PyMol.

``` r
require("knitr")
knitr::opts_chunk$set(echo = T)
knitr::opts_chunk$set(dev.args = list(png = list(type = "cairo")))

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

    ## R version 3.6.2 (2019-12-12)
    ## Platform: x86_64-pc-linux-gnu (64-bit)
    ## Running under: Ubuntu 18.04.4 LTS
    ## 
    ## Matrix products: default
    ## BLAS/LAPACK: /app/software/OpenBLAS/0.3.7-GCC-8.3.0/lib/libopenblas_haswellp-r0.3.7.so
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
    ##  [1] ggrepel_0.8.1     bio3d_2.4-0       gridExtra_2.3     forcats_0.4.0    
    ##  [5] stringr_1.4.0     dplyr_0.8.3       purrr_0.3.3       readr_1.3.1      
    ##  [9] tidyr_1.0.0       tibble_3.0.2      ggplot2_3.3.0     tidyverse_1.3.0  
    ## [13] data.table_1.12.8 yaml_2.2.0        knitr_1.26       
    ## 
    ## loaded via a namespace (and not attached):
    ##  [1] tidyselect_1.1.0 xfun_0.11        haven_2.2.0      colorspace_1.4-1
    ##  [5] vctrs_0.3.1      generics_0.0.2   htmltools_0.4.0  rlang_0.4.7     
    ##  [9] pillar_1.4.5     glue_1.3.1       withr_2.1.2      DBI_1.1.0       
    ## [13] dbplyr_1.4.2     modelr_0.1.5     readxl_1.3.1     lifecycle_0.2.0 
    ## [17] munsell_0.5.0    gtable_0.3.0     cellranger_1.1.0 rvest_0.3.5     
    ## [21] evaluate_0.14    parallel_3.6.2   fansi_0.4.0      broom_0.7.0     
    ## [25] Rcpp_1.0.3       scales_1.1.0     backports_1.1.5  jsonlite_1.6    
    ## [29] fs_1.3.1         hms_0.5.2        digest_0.6.23    stringi_1.4.3   
    ## [33] grid_3.6.2       cli_2.0.0        tools_3.6.2      magrittr_1.5    
    ## [37] crayon_1.3.4     pkgconfig_2.0.3  ellipsis_0.3.0   xml2_1.2.2      
    ## [41] reprex_0.3.0     lubridate_1.7.4  assertthat_0.2.1 rmarkdown_2.0   
    ## [45] httr_1.4.1       rstudioapi_0.10  R6_2.4.1         compiler_3.6.2

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
sites$mean_polyspecificity <- NA
  
for(i in 1:nrow(sites)){
  if(sites[i,"chain"] != "link"){
    sites$mean_CGG_bind[i] <- mean(dt[position_IMGT==sites[i,"site"] & chain==sites[i,"chain"] & wildtype != mutant & n_bc_bind_CGG>3,delta_bind_CGG],na.rm=T)
    sites$max_CGG_bind[i] <- max(dt[position_IMGT==sites[i,"site"] & chain==sites[i,"chain"] & wildtype != mutant & n_bc_bind_CGG>3,delta_bind_CGG],na.rm=T)
    sites$quartile75_CGG_bind[i] <- quantile(dt[position_IMGT==sites[i,"site"] & chain==sites[i,"chain"] & wildtype != mutant & n_bc_bind_CGG>3,delta_bind_CGG],0.75,na.rm=T)
    sites$mean_expression[i] <- mean(dt[position_IMGT==sites[i,"site"] & chain==sites[i,"chain"] & wildtype != mutant & n_bc_expr>3,delta_expr],na.rm=T)
    sites$mean_polyspecificity[i] <- mean(dt[position_IMGT==sites[i,"site"] & chain==sites[i,"chain"] & wildtype != mutant & n_bc_psr>3,delta_psr],na.rm=T)
  }
}
```

## Map metrics to the structure

``` r
b_mean_CGG <- rep(0, length(pdb$atom$b))
b_max_CGG <- rep(0, length(pdb$atom$b))
b_75th_CGG <- rep(0, length(pdb$atom$b))
b_mean_expr <- rep(0, length(pdb$atom$b))
b_mean_psr <- rep(0, length(pdb$atom$b))
for(i in 1:nrow(pdb$atom)){
  res <- pdb$atom$resno[i]
  if(pdb$atom$chain[i] %in% c("B","H") & res %in% sites[sites$chain=="H","site"]){
    b_mean_CGG[i] <- sites[sites$site==res & sites$chain=="H","mean_CGG_bind"]
    b_max_CGG[i] <- sites[sites$site==res & sites$chain=="H","max_CGG_bind"]
    b_75th_CGG[i] <- sites[sites$site==res & sites$chain=="H","quartile75_CGG_bind"]
    b_mean_expr[i] <- sites[sites$site==res & sites$chain=="H","mean_expression"]
    b_mean_psr[i] <- sites[sites$site==res & sites$chain=="H","mean_polyspecificity"]
  }else if(pdb$atom$chain[i] %in% c("C","L") & res %in% sites[sites$chain=="L","site"]){
    b_mean_CGG[i] <- sites[sites$site==res & sites$chain=="L","mean_CGG_bind"]
    b_max_CGG[i] <- sites[sites$site==res & sites$chain=="L","max_CGG_bind"]
    b_75th_CGG[i] <- sites[sites$site==res & sites$chain=="L","quartile75_CGG_bind"]
    b_mean_expr[i] <- sites[sites$site==res & sites$chain=="L","mean_expression"]
    b_mean_psr[i] <- sites[sites$site==res & sites$chain=="L","mean_polyspecificity"]
  }
}
write.pdb(pdb=pdb, file=paste(config$structural_mapping_dir,"/CGG_mean.pdb",sep=""), b=b_mean_CGG)
write.pdb(pdb=pdb, file=paste(config$structural_mapping_dir,"/CGG_max.pdb",sep=""), b=b_max_CGG)
write.pdb(pdb=pdb, file=paste(config$structural_mapping_dir,"/CGG_75th.pdb",sep=""), b=b_75th_CGG)
write.pdb(pdb=pdb, file=paste(config$structural_mapping_dir,"/expression_mean.pdb",sep=""), b=b_mean_expr)
write.pdb(pdb=pdb, file=paste(config$structural_mapping_dir,"/psr_mean.pdb",sep=""), b=b_mean_psr)
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
