# Author: Yuanrong Liu
# Title: RNA-Seq Normalization
# Time: 03/01
library(DESeq2)
library(edgeR)

setwd("~/")
# variables are in an environment
load("C:./.RData")

# DATA
info #experiment information 
total #genome information
# RNA counts matrix
counts_matrix <- counts(dds)
info$mode_of_action <- relevel(info$mode_of_action, ref = "Control")

# generate a GRangesList for the FPKM function in deseq2 
genome_data_split <- function(string_data, return_int = T, sep = ";"){
  temp <- strsplit(as.character(string_data), sep)[[1]]
  if(return_int)
    return(as.integer(temp))
  else
    return(temp)
}

get_GRangesList <- function(total){
  gene_pos <- GRangesList()
  for (id in 1:nrow(total)) {
    chr_num <- genome_data_split(total$V2[id],return_int = F)
    start_pos <- genome_data_split(total$V3[id])
    end_pos <- genome_data_split(total$V4[id])
    strand_strand <- genome_data_split(total$V5[id],return_int = F)
    if(length(start_pos) != length(end_pos) | length(start_pos) != length(chr_num) | length(start_pos) != length(strand_strand))
      stop("numbers of objects are not match;")
    temp <- GRanges(seqnames = Rle(unique(chr_num),as.integer(table(chr_num))),
                    ranges = IRanges(start_pos, end_pos),
                    strand = strand_strand)
    gene_pos <- c(gene_pos, GRangesList(temp))
    print(id)
  }
  return(gene_pos)
}

#FPKM deseq2
dds <- DESeqDataSetFromMatrix(
  countData = counts_matrix,
  colData = info,
  design= ~ mode_of_action
)
genes <- intersect(rownames(cnts), total$V1)
ids <- which(total$V1 %in% genes)
total_info <- total[ids,]
#res_GRangesList <- get_GRangesList(total_info)
res_GRangesList <- readRDS("genes_GRangesList.rds")
rowRanges(dds) <- res_GRangesList
DESEQ_fpkm <- fpkm(dds)

# TMM edgeR
dglist <- DGEList(counts_matrix, genes = rownames(counts_matrix))
dglist_tmm <- calcNormFactors(dglist,method = "TMM")
lib.size <- dglist_tmm$samples$lib.size
scale.factor <- dglist_tmm$samples$norm.factors

edgeR_tmm_manual <-  t(t(dglist_tmm$counts)/(scale.factor*lib.size))
edgeR_tmm <- cpm(dglist_tmm)

#RPKM edgeR
edgeR_rpkm <- rpkm(dglist,gene.length = as.numeric(total_info$V6))

# CPM edgeR
edgeR_cpm <- cpm(dglist)

#TPM 
library(scater)
example_rna <- SingleCellExperiment(
  assays = list(counts = counts_matrix), colData = info)
scater_tpm <- calculateTPM(example_rna, effective_length = as.numeric(total_info$V6))