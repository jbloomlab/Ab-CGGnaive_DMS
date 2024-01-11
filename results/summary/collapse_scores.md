Collapse barcodes to final per-scFv/mutant phenotype scores
================
Tyler Starr
08/06/2021

- <a href="#setup" id="toc-setup">Setup</a>
- <a href="#barcode-distribution" id="toc-barcode-distribution">Barcode
  distribution</a>
- <a href="#calculate-per-mutant-score-across-libraries"
  id="toc-calculate-per-mutant-score-across-libraries">Calculate
  per-mutant score across libraries</a>
- <a href="#heatmaps" id="toc-heatmaps">Heatmaps!</a>

This notebook reads in the per-barcode titration Kds and expr
measurements from the `compute_binding_Kd` and `compute_expr_meanF`
scripts. It synthesizes these two sets of results and calculates the
final ‘mean’ phenotypes for each variant, and generates some coverage
and QC analyses.

``` r
require("knitr")
knitr::opts_chunk$set(echo = T)
knitr::opts_chunk$set(dev.args = list(png = list(type = "cairo")))
options(repos = c(CRAN = "https://cran.r-project.org"))

#list of packages to install/load
packages = c("yaml","data.table","tidyverse","gridExtra","seqinr")
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
if(!file.exists(config$final_variant_scores_dir)){
  dir.create(file.path(config$final_variant_scores_dir))
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
    ##  [1] seqinr_4.2-36     gridExtra_2.3     forcats_0.5.1     stringr_1.4.0    
    ##  [5] dplyr_1.0.6       purrr_0.3.4       readr_1.4.0       tidyr_1.1.3      
    ##  [9] tibble_3.1.2      ggplot2_3.3.3     tidyverse_1.3.1   data.table_1.14.0
    ## [13] yaml_2.2.1        knitr_1.33       
    ## 
    ## loaded via a namespace (and not attached):
    ##  [1] tidyselect_1.1.1  xfun_0.23         haven_2.4.1       colorspace_2.0-1 
    ##  [5] vctrs_0.3.8       generics_0.1.0    htmltools_0.5.1.1 utf8_1.2.1       
    ##  [9] rlang_0.4.11      pillar_1.6.1      glue_1.4.2        withr_3.0.2      
    ## [13] DBI_1.1.1         dbplyr_2.1.1      modelr_0.1.8      readxl_1.3.1     
    ## [17] lifecycle_1.0.0   munsell_0.5.0     gtable_0.3.0      cellranger_1.1.0 
    ## [21] rvest_1.0.0       evaluate_0.14     ps_1.6.0          fansi_0.4.2      
    ## [25] broom_0.7.6       Rcpp_1.0.13-1     scales_1.1.1      backports_1.2.1  
    ## [29] jsonlite_1.7.2    fs_1.5.0          hms_1.1.0         digest_0.6.27    
    ## [33] stringi_1.6.2     ade4_1.7-22       grid_3.6.3        cli_2.5.0        
    ## [37] tools_3.6.3       magrittr_2.0.1    crayon_1.4.1      pkgconfig_2.0.3  
    ## [41] MASS_7.3-54       ellipsis_0.3.2    xml2_1.3.2        reprex_2.0.0     
    ## [45] lubridate_1.7.10  assertthat_0.2.1  rmarkdown_2.8     httr_1.4.2       
    ## [49] rstudioapi_0.13   R6_2.5.0          compiler_3.6.3

## Setup

Read in tables of per-barcode expr and binding Kd measurements and
combine.

``` r
dt_bind <- data.table(read.csv(config$Titeseq_Kds_file),stringsAsFactors=F)
dt_expr <- data.table(read.csv(config$expression_sortseq_file),stringsAsFactors=F)
setnames(dt_expr,"expression","expr")

dt <- merge(dt_bind, dt_expr)
```

## Barcode distribution

Some QC plots. First, look at distribution of number barcodes for
binding, expr, polyspecificity measurements for single mutant
detemrinations. These are ‘left-justified’ histograms, so the leftmost
bar represents the number of genotypes for which no barcodes were
collapsed to final measurement in a pool. (Currently, includes mutations
within the linker)

``` r
par(mfrow=c(2,2))
hist(dt[library=="lib1" & variant_class=="1 nonsynonymous",n_bc_bind_CGG],main="lib1, bind_CGG",right=F,breaks=max(dt[library=="lib1" & variant_class=="1 nonsynonymous",n_bc_bind_CGG],na.rm=T),xlab="")
hist(dt[library=="lib2" & variant_class=="1 nonsynonymous",n_bc_bind_CGG],main="lib2, bind_CGG",right=F,breaks=max(dt[library=="lib2" & variant_class=="1 nonsynonymous",n_bc_bind_CGG],na.rm=T),xlab="")
hist(dt[library=="lib1" & variant_class=="1 nonsynonymous",n_bc_expr],main="lib1, expr",right=F,breaks=max(dt[library=="lib1" & variant_class=="1 nonsynonymous",n_bc_expr],na.rm=T),xlab="number barcodes collapsed")
hist(dt[library=="lib2" & variant_class=="1 nonsynonymous",n_bc_expr],main="lib2, expr",right=F,breaks=max(dt[library=="lib2" & variant_class=="1 nonsynonymous",n_bc_expr],na.rm=T),xlab="number barcodes collapsed")
```

<img src="collapse_scores_files/figure-gfm/hist_n_bc_per_mutant-1.png" style="display: block; margin: auto;" />

``` r
invisible(dev.print(pdf, paste(config$final_variant_scores_dir,"/histogram_n_bc_per_geno_sep-libs.pdf",sep=""),useDingbats=F))
```

Format into a ‘mutation lookup table’, where we focus just on the single
mutants (and wildtype), breakup the string of mutations, and fill in the
table to also include any missing mutants.

``` r
dt_mutant <- dt[variant_class %in% "1 nonsynonymous",]
#split mutation string
#define function to apply
split_mut <- function(x){
  split <- strsplit(x,split="")[[1]]
  return(list(split[1],as.numeric(paste(split[2:(length(split)-1)],collapse="")),split[length(split)]))
}
dt_mutant[,c("wildtype","position","mutant"):=split_mut(as.character(aa_substitutions)),by=aa_substitutions]

dt_mutant <- dt_mutant[,.(library,target,wildtype,position,mutant,
                         bind_CGG,n_bc_bind_CGG,sum_conc_bin_norm_count_bind_CGG,
                         expr,n_bc_expr,expr_count)]

aas <- c("A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y")
#fill out missing values in table with a hideous loop, so the table is complete for all mutaitons (including those that are missing). If you are somebody who is reading this code, I apologize.
for(lib in c("lib1","lib2")){
  for(bg in as.character(unique(dt_mutant$target))){
    for(pos in 1:max(dt_mutant$position)){
      for(aa in aas){
        if(!(aa %in% as.character(dt_mutant[library==lib & target==bg & position==pos,mutant]))){
          dt_mutant <- rbind(dt_mutant,list(lib, bg, dt_mutant[library==lib & target==bg & position==pos,wildtype][1],pos,aa),fill=T)
        }
      }
    }
  }
}
setkey(dt_mutant,library,target,position,mutant)

#fill in wildtype values -- should vectorize in data table but being so stupid so just going to write for loop
for(lib in c("lib1","lib2")){
  dt_mutant[library==lib & wildtype==mutant, c("bind_CGG", "n_bc_bind_CGG","sum_conc_bin_norm_count_bind_CGG",
                                               "expr","n_bc_expr","expr_count"
                                              ):=
              dt[library==lib & variant_class=="wildtype",.(bind_CGG,n_bc_bind_CGG,sum_conc_bin_norm_count_bind_CGG,
                                                            expr,n_bc_expr,expr_count
                                                            )]]
}

# TODO should we be overwriting the delta_bind_CGG that was computed in Titseq_modeling?
#add delta bind and expr measures
for(lib in c("lib1","lib2")){
  ref_bind_CGG <- dt[library==lib & variant_class=="wildtype", bind_CGG]
  ref_expr <- dt[library==lib & variant_class=="wildtype", expr]
  dt_mutant[library==lib,delta_bind_CGG := bind_CGG - ref_bind_CGG]
  dt_mutant[library==lib,delta_expr := expr - ref_expr]
}
```

We have duplicates for each measurement. Let’s look at correlations!
Later on, can look at how correlation degrades when subsetting on lower
and lower n_bcs, and use that to determine if I need to filter for a
minimum number of collapsed bcs

``` r
par(mfrow=c(1,2))
x <- dt_mutant[library=="lib1" & wildtype!=mutant & !(position %in% 113:127),bind_CGG]; y <- dt_mutant[library=="lib2" & wildtype!=mutant  & !(position %in% 113:127),bind_CGG]; plot(x,y,pch=16,col="#00000020",xlab="replicate 1",ylab="replicate 2",main="CGG binding affinity");model <- lm(y~x);abline(model,lty=2,col="red");legend("topleft",legend=paste("R2: ",round(summary(model)$r.squared,3),sep=""),bty="n")

x <- dt_mutant[library=="lib1" & wildtype!=mutant & !(position %in% 113:127),expr]; y <- dt_mutant[library=="lib2" & wildtype!=mutant & !(position %in% 113:127),expr]; plot(x,y,pch=16,col="#00000020",xlab="replicate 1",ylab="replicate 2",main="expr");model <- lm(y~x);abline(model,lty=2,col="red");legend("topleft",legend=paste("R2: ",round(summary(model)$r.squared,3),sep=""),bty="n")
```

<img src="collapse_scores_files/figure-gfm/plot_correlations-1.png" style="display: block; margin: auto;" />

``` r
invisible(dev.print(pdf, paste(config$final_variant_scores_dir,"/replicate_correlations.pdf",sep=""),useDingbats=F))
```

## Calculate per-mutant score across libraries

Collapse down to mean from both replicates, and total n barcodes between
the two replicates. Also record the number of the replicates the variant
was quantified within. Note, we are currently keeping a value even if
it’s determined from a single bc fit in a single pool. Later on, we’ll
want to require some combination of minimum number of bcs within or
between libraries for retention.

``` r
dt_final <- copy(dt_mutant)

dt_final[ ,bind_tot_CGG:=mean(bind_CGG,na.rm=T),by=c("target","position","mutant")]
dt_final[ ,delta_bind_tot_CGG:=mean(delta_bind_CGG,na.rm=T),by=c("target","position","mutant")]
dt_final[ ,n_bc_bind_tot_CGG:=sum(n_bc_bind_CGG,na.rm=T),by=c("target","position","mutant")]
dt_final[ ,n_libs_bind_tot_CGG:=sum(!is.na(bind_CGG)),by=c("target","position","mutant")]

dt_final[ ,expr_tot:=mean(expr,na.rm=T),by=c("target","position","mutant")]
dt_final[ ,delta_expr_tot:=mean(delta_expr,na.rm=T),by=c("target","position","mutant")]
dt_final[ ,n_bc_expr_tot:=sum(n_bc_expr,na.rm=T),by=c("target","position","mutant")]
dt_final[ ,n_libs_expr_tot:=sum(!is.na(expr)),by=c("target","position","mutant")]


#switch to antibody indexing of postitions, also add annotation column for CDR/FWR annotations
CGG_sites <- read.csv(file=config$CGGnaive_site_info, stringsAsFactors = F)
for(i in 1:nrow(CGG_sites)){
  dt_final[position==CGG_sites[i,"site_scFv"],position_IMGT:=CGG_sites[i,"site"]]
  dt_final[position==CGG_sites[i,"site_scFv"],chain:=CGG_sites[i,"chain"]]
  dt_final[position==CGG_sites[i,"site_scFv"],codon:=CGG_sites[i,"KI_codon"]]
  dt_final[position==CGG_sites[i,"site_scFv"],annotation:=CGG_sites[i,"annotation"]]
}

#add single mutation string
dt_final[,mutation:=paste(wildtype,position_IMGT,"(",chain,")",mutant,sep=""),by=c("wildtype","position","mutant")]

dt_final <- unique(dt_final[,.(target,wildtype,position,position_IMGT,chain,annotation,mutant,mutation,codon,
                               bind_tot_CGG,delta_bind_tot_CGG,n_bc_bind_tot_CGG,n_libs_bind_tot_CGG,
                               expr_tot,delta_expr_tot,n_bc_expr_tot,n_libs_expr_tot
                               )])

#rename some of the columns
setnames(dt_final,"bind_tot_CGG","bind_CGG")
setnames(dt_final,"delta_bind_tot_CGG","delta_bind_CGG")
setnames(dt_final,"n_bc_bind_tot_CGG","n_bc_bind_CGG")
setnames(dt_final,"n_libs_bind_tot_CGG","n_libs_bind_CGG")
setnames(dt_final,"expr_tot","expr")
setnames(dt_final,"delta_expr_tot","delta_expr")
setnames(dt_final,"n_bc_expr_tot","n_bc_expr")
setnames(dt_final,"n_libs_expr_tot","n_libs_expr")
```

Censor any measurements that are from \<3 bc or only sampled in a single
replicate

``` r
min_bc <- config$min_variant_bc_replicates
min_lib <- config$min_variant_lib_replicates

dt_final[n_bc_bind_CGG < min_bc | n_libs_bind_CGG < min_lib, c("bind_CGG","delta_bind_CGG","n_bc_bind_CGG","n_libs_bind_CGG") := list(NA,NA,NA,NA)]
dt_final[n_bc_expr < min_bc | n_libs_expr < min_lib, c("expr","delta_expr","n_bc_expr","n_libs_expr") := list(NA,NA,NA,NA)]
```

Coverage stats on n_barcodes for different measurements in the final
pooled measurements.

``` r
par(mfrow=c(1,2))
hist(dt_final[wildtype!=mutant & !(chain=="link"), n_bc_bind_CGG],col="gray50",main=paste("mutant bind_CGG score,\nmedian ",median(dt_final[wildtype!=mutant & !(chain=="link"), n_bc_bind_CGG],na.rm=T),sep=""),right=F,breaks=max(dt_final[wildtype!=mutant & !(chain=="link"), n_bc_bind_CGG],na.rm=T),xlab="")
hist(dt_final[wildtype!=mutant & !(chain=="link"), n_bc_expr],col="gray50",main=paste("mutant expr score,\nmedian ",median(dt_final[wildtype!=mutant & !(chain=="link"), n_bc_expr],na.rm=T),sep=""),right=F,breaks=max(dt_final[wildtype!=mutant & !(chain=="link"), n_bc_expr],na.rm=T),xlab="")
```

<img src="collapse_scores_files/figure-gfm/n_barcode_plots-1.png" style="display: block; margin: auto;" />

``` r
invisible(dev.print(pdf, paste(config$final_variant_scores_dir,"/histogram_n_bc_per_geno_pooled-libs.pdf",sep="")))
```

Relationships in mutation effects between the four properties? And make
some specific plots for the ones I want to show as primary figs.

``` r
pairs(dt_final[wildtype!=mutant & !(chain=="link"), .(delta_bind_CGG,delta_expr)],main="",pch=16,col="#00000010")
```

<img src="collapse_scores_files/figure-gfm/scatters_mut_effects_on_phenos-1.png" style="display: block; margin: auto;" />

``` r
invisible(dev.print(pdf, paste(config$final_variant_scores_dir,"/scatterplots_mut_effects_on_3_phenos.pdf",sep="")))
```

``` r
par(mfrow=c(1,1))

plot(dt_final[wildtype!=mutant & !(chain=="link"),delta_expr],dt_final[wildtype!=mutant & !(chain=="link"),delta_bind_CGG],pch=16,col="#00000015",xlab="mutant effect on scFv expr",ylab="mutant effect on CGG-binding affinity")
```

<img src="collapse_scores_files/figure-gfm/scatters_individual-1.png" style="display: block; margin: auto;" />

``` r
invisible(dev.print(pdf, paste(config$final_variant_scores_dir,"/scatterplots_mut_effects_on_specific_phenos.pdf",sep="")))
```

Annotate with whether a mutation is single-nt-mut accessible from the
CGG naive knockin allele.

``` r
#define a function that takes a character of three nucleotides (a codon), and outputs all amino acids that can be accessed via single-nt mutation of that codon
get.codon.muts <- function(codon){
  nt <- c("A","C","G","T")
  codon_split <- strsplit(codon,split="")[[1]]
  codon_muts <- seqinr::translate(codon_split)
  for(i in nt[nt!=codon_split[1]]){
    codon_muts <- c(codon_muts,seqinr::translate(c(i,codon_split[2:3])))
  }
  for(i in nt[nt!=codon_split[2]]){
    codon_muts <- c(codon_muts,seqinr::translate(c(codon_split[1],i,codon_split[3])))
  }
  for(i in nt[nt!=codon_split[3]]){
    codon_muts <- c(codon_muts,seqinr::translate(c(codon_split[1:2],i)))
  }
  return(codon_muts)
}

dt_final[!is.na(codon),single_nt := mutant %in% get.codon.muts(codon),by=mutation]
```

## Heatmaps!

Order factor variables for plotting, translate to heavy chain and light
chain numbering, etc.

``` r
#order mutant as a factor for grouping by rough biochemical grouping
dt_final$mutant <- factor(dt_final$mutant, levels=c("C","P","G","V","M","L","I","A","F","W","Y","T","S","N","Q","E","D","H","K","R"))
#add character vector indicating wildtype to use as plotting symbols for wt
dt_final[,wildtype_indicator := ""]
dt_final[as.character(mutant)==as.character(wildtype),wildtype_indicator := "x"]

#add character vector indicating multi mut indicator to use as plotting symbol
dt_final[,multimut_indicator := ""]
dt_final[single_nt==FALSE, multimut_indicator := "/"]

#make temp long-form data frame
temp <- data.table::melt(dt_final[, .(target,position,position_IMGT,chain,mutant,
                                      bind_CGG,delta_bind_CGG,expr,delta_expr,wildtype_indicator,multimut_indicator)],
                         id.vars=c("target","position","position_IMGT","chain","mutant","wildtype_indicator","multimut_indicator"),
                         measure.vars=c("bind_CGG","delta_bind_CGG","expr","delta_expr"),
                         variable.name="measurement",
                         value.name="value")
temp[,position_IMGT:=paste(chain,position_IMGT,sep="")]

temp$position_IMGT <- factor(temp$position_IMGT,levels=unique(temp$position_IMGT))

#for method to duplicate aa labels on right side of plot https://github.com/tidyverse/ggplot2/issues/3171
guide_axis_label_trans <- function(label_trans = identity, ...) {
  axis_guide <- guide_axis(...)
  axis_guide$label_trans <- rlang::as_function(label_trans)
  class(axis_guide) <- c("guide_axis_trans", class(axis_guide))
  axis_guide
}

guide_train.guide_axis_trans <- function(x, ...) {
  trained <- NextMethod()
  trained$key$.label <- x$label_trans(trained$key$.label)
  trained
}
```

Make heatmaps showing raw affinity and delta-affinity of muts relative
to wildtype for CGG binding

``` r
p1 <- ggplot(temp[measurement=="bind_CGG" & chain != "link",],aes(position_IMGT,mutant))+geom_tile(aes(fill=value),color="black",lwd=0.1)+
  scale_fill_gradientn(colours=c("#FFFFFF","#003366"),limits=c(6,11.5),na.value="yellow")+
  labs(x="",y="")+theme_classic(base_size=9)+
  coord_equal()+theme(axis.text.x=element_text(angle=90,hjust=1,vjust=0.6,face="bold",size=10),axis.text.y=element_text(face="bold",size=10))+
  guides(y.sec=guide_axis_label_trans())+
  geom_text(aes(label=wildtype_indicator),size=2,color="gray10")

p1
```

<img src="collapse_scores_files/figure-gfm/heatmap_DMS_log10Ka_CGG-1.png" style="display: block; margin: auto;" />

``` r
invisible(dev.print(pdf, paste(config$final_variant_scores_dir,"/heatmap_SSM_log10Ka_CGG.pdf",sep="")))
```

Second, illustrating delta_log10Ka grouped by SSM position.

``` r
p1 <- ggplot(temp[measurement=="delta_bind_CGG" & chain != "link",],aes(position_IMGT,mutant))+geom_tile(aes(fill=value),color="black",lwd=0.1)+
  scale_fill_gradientn(colours=c("#A94E35","#A94E35","#F48365","#FFFFFF","#7378B9","#383C6C"),limits=c(-5,1),values=c(0/6,1/6,3/6,5/6,5.5/6,6/6),na.value="gray70")+
  labs(x="",y="")+theme_classic(base_size=9)+
  coord_equal()+theme(axis.text.x=element_text(angle=90,hjust=1,vjust=0.6,face="bold",size=10),axis.text.y=element_text(face="bold",size=10))+
  guides(y.sec=guide_axis_label_trans())+
  geom_text(aes(label=wildtype_indicator),size=2,color="gray10")

p1
```

<img src="collapse_scores_files/figure-gfm/heatmap_DMS_delta-log10Ka_CGG-by-target-1.png" style="display: block; margin: auto;" />

``` r
invisible(dev.print(pdf, paste(config$final_variant_scores_dir,"/heatmap_SSM_delta-log10Ka_CGG.pdf",sep="")))
```

Same as above, but hatch out mutations that are not accessible via
single nt mutation from the KI naive BCR

``` r
p1 <- ggplot(temp[measurement=="delta_bind_CGG" & chain != "link",],aes(position_IMGT,mutant))+geom_tile(aes(fill=value),color="black",lwd=0.1)+
  scale_fill_gradientn(colours=c("#A94E35","#A94E35","#F48365","#FFFFFF","#7378B9","#383C6C"),limits=c(-5,1),values=c(0/6,1/6,3/6,5/6,5.5/6,6/6),na.value="gray70")+
  labs(x="",y="")+theme_classic(base_size=9)+
  coord_equal()+theme(axis.text.x=element_text(angle=90,hjust=1,vjust=0.6,face="bold",size=10),axis.text.y=element_text(face="bold",size=10))+
  guides(y.sec=guide_axis_label_trans())+
  geom_text(aes(label=wildtype_indicator),size=2,color="gray10")+
  geom_text(aes(label=multimut_indicator),size=2.5,color="gray10")

p1
```

<img src="collapse_scores_files/figure-gfm/heatmap_DMS_delta-log10Ka_CGG-by-target_singlent-1.png" style="display: block; margin: auto;" />

``` r
invisible(dev.print(pdf, paste(config$final_variant_scores_dir,"/heatmap_SSM_delta-log10Ka_CGG_singlent.pdf",sep="")))
```

Make heatmaps faceted by target, showing raw expr and delta-expr of muts
relative to respective wildtype

``` r
p1 <- ggplot(temp[measurement=="expr" & chain != "link",],aes(position_IMGT,mutant))+geom_tile(aes(fill=value),color="black",lwd=0.1)+
  scale_fill_gradientn(colours=c("#FFFFFF","#003366"),limits=c(7,11),na.value="yellow")+
  labs(x="",y="")+theme_classic(base_size=9)+
  coord_equal()+theme(axis.text.x=element_text(angle=90,hjust=1,vjust=0.6,face="bold",size=10),axis.text.y=element_text(face="bold",size=10))+
  guides(y.sec=guide_axis_label_trans())+
  geom_text(aes(label=wildtype_indicator),size=2,color="gray10")

p1
```

<img src="collapse_scores_files/figure-gfm/heatmap_DMS_expr-by-target-1.png" style="display: block; margin: auto;" />

``` r
invisible(dev.print(pdf, paste(config$final_variant_scores_dir,"/heatmap_SSM_expr-by-target.pdf",sep="")))
```

Second, illustrating delta_expr grouped by SSM position.

``` r
p1 <- ggplot(temp[measurement=="delta_expr" & chain != "link",],aes(position_IMGT,mutant))+geom_tile(aes(fill=value),color="black",lwd=0.1)+
  scale_fill_gradientn(colours=c("#A94E35","#F48365","#FFFFFF","#7378B9","#383C6C"),limits=c(-4,1),values=c(0/5,2/5,4/5,4.5/5,5/5),na.value="yellow")+
  labs(x="",y="")+theme_classic(base_size=9)+
  coord_equal()+theme(axis.text.x=element_text(angle=90,hjust=1,vjust=0.6,face="bold",size=10),axis.text.y=element_text(face="bold",size=10))+
  guides(y.sec=guide_axis_label_trans())+
  geom_text(aes(label=wildtype_indicator),size=2,color="gray10")

p1
```

<img src="collapse_scores_files/figure-gfm/heatmap_DMS_delta-expr-by-target-1.png" style="display: block; margin: auto;" />

``` r
invisible(dev.print(pdf, paste(config$final_variant_scores_dir,"/heatmap_SSM_delta-expr-by-target.pdf",sep="")))
```

And expr, hatchign out \>single-nt-accessible

``` r
p1 <- ggplot(temp[measurement=="delta_expr" & chain != "link",],aes(position_IMGT,mutant))+geom_tile(aes(fill=value),color="black",lwd=0.1)+
  scale_fill_gradientn(colours=c("#A94E35","#F48365","#FFFFFF","#7378B9","#383C6C"),limits=c(-4,1),values=c(0/5,2/5,4/5,4.5/5,5/5),na.value="yellow")+
  labs(x="",y="")+theme_classic(base_size=9)+
  coord_equal()+theme(axis.text.x=element_text(angle=90,hjust=1,vjust=0.6,face="bold",size=10),axis.text.y=element_text(face="bold",size=10))+
  guides(y.sec=guide_axis_label_trans())+
  geom_text(aes(label=wildtype_indicator),size=2,color="gray10")+
  geom_text(aes(label=multimut_indicator),size=2.5,color="gray10")

p1
```

<img src="collapse_scores_files/figure-gfm/heatmap_DMS_delta-expr-by-target_singlent-1.png" style="display: block; margin: auto;" />

``` r
invisible(dev.print(pdf, paste(config$final_variant_scores_dir,"/heatmap_SSM_delta-expr-by-target_singlent.pdf",sep="")))
```

That’s the data! Other analyses in additional notebooks

Save output files.

``` r
dt_final[,.(target,wildtype,position,position_IMGT,chain,annotation,mutant,mutation,codon,single_nt,
            bind_CGG,delta_bind_CGG,n_bc_bind_CGG,n_libs_bind_CGG,
            expr,delta_expr,n_bc_expr,n_libs_expr
            )] %>%
  mutate_if(is.numeric, round, digits=5) %>%
  write.csv(file=config$final_variant_scores_mut_file, row.names=F,quote=F)
```
