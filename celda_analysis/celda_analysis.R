library(celda)
library(SummarizedExperiment)
library(MAST)
library(Matrix)
pbmc_res_list <- readRDS("/usr3/graduate/yliu6680/2018summer/0821_4k/diff_l/pbmc4k_k15-24_l50-90.rds")

data <- readMM("/usr3/graduate/yliu6680/2018summer/models/4k/matrix.mtx")
barcodes <- read.table("/usr3/graduate/yliu6680/2018summer/models/4k/barcodes.tsv",sep = "\t")
genes <- read.table("/usr3/graduate/yliu6680/2018summer/models/4k/genes.tsv",sep = "\t")

pbmc4k <- as.matrix(data)
rownames(pbmc4k) <- paste(genes$V1, "_", genes$V2, sep = "")
colnames(pbmc4k) <- barcodes$V1

pbmc4k_select <- pbmc4k[rowSums(pbmc4k>3) > 3,]
class(pbmc4k_select) = "integer"
dim(pbmc4k_select)

### prepare for the DrGenes
marker_genes <- c("ENSG00000163736_PPBP", "ENSG00000198851_CD3E",
                  "ENSG00000168685_IL7R", "ENSG00000153563_CD8A",
                  "ENSG00000203747_FCGR3A", "ENSG00000166927_MS4A7",
                  "ENSG00000090382_LYZ", "ENSG00000170458_CD14",
                  "ENSG00000156738_MS4A1", "ENSG00000185507_IRF7",
                  "ENSG00000101439_CST3", "ENSG00000179639_FCER1A",
                  "ENSG00000196230_TUBB", "ENSG00000132646_PCNA",
                  "ENSG00000105374_NKG7", "ENSG00000115523_GNLY"
                  )

gene_counts <- pbmc4k_select[marker_genes,]

### get the marker genes for each type of cells
allGenes <- rownames(pbmc4k_select)
markerGenes <- c("PPBP$","CD3E$","IL7R$","CD8A$","FCGR3A$","MS4A7$",
                 "LYZ$","CD14$","MS4A1$","IRF7$","CST3$", "FCER1A$",
                 "PCNA$","TUBB$","NKG7$","GNLY$"
                 )
markerGenesFullnames <- c()
for (i in 1:length(markerGenes)){
  markerGenesFullnames <- append(markerGenesFullnames, allGenes[grep(markerGenes[i],allGenes)])
}

k <- 18
l <- 80

bestModel1 <- selectBestModel(celda.list = pbmc_res_list,K = k, L = l)

### relative heatmap
png("model1RelativeHeatmap.png",width = 3000,height = 3000,res = 360)
relativeProbabilityHeatmap(counts = pbmc4k_select, celda.mod = bestModel1)
dev.off()

### prepare the tsne plots
set.seed(123)
model1FactorizeMatrix <- factorizeMatrix(counts = pbmc4k_select, celda.mod = bestModel1)
model1NormPbmc <- normalizeCounts(model1FactorizeMatrix$counts$cell.states)
model1Tsne <- celdaTsne(counts = pbmc4k_select, celda.mod = bestModel1,distance = "hellinger")

### plot the tsne plots
png("model1DrCluster.png",width = 3000,height = 3000,res = 360)
plotDrCluster(dim1 = model1Tsne[,1],dim2 = model1Tsne[,2],cluster = as.factor(bestModel1$z),size = 0.3)
dev.off()

png("model1DeState.png",width = 3000,height = 3000,res = 360)
plotDrState(dim1 = model1Tsne[,1], dim2 = model1Tsne[,2], 
            matrix = model1FactorizeMatrix$proportions$cell.states,
            rescale = TRUE)
dev.off()

### prepare for the DrGenes
marker_genes <- c("ENSG00000163736_PPBP", "ENSG00000198851_CD3E",
                  "ENSG00000168685_IL7R", "ENSG00000153563_CD8A",
                  "ENSG00000203747_FCGR3A", "ENSG00000166927_MS4A7",
                  "ENSG00000090382_LYZ", "ENSG00000170458_CD14",
                  "ENSG00000156738_MS4A1", "ENSG00000185507_IRF7",
                  "ENSG00000101439_CST3", "ENSG00000179639_FCER1A",
                  "ENSG00000196230_TUBB", "ENSG00000132646_PCNA",
                  "ENSG00000105374_NKG7", "ENSG00000115523_GNLY"
)

gene_counts <- pbmc4k_select[marker_genes,]

### get the marker genes for each type of cells
allGenes <- rownames(pbmc4k_select)
markerGenes <- c("PPBP$","CD3E$","IL7R$","CD8A$","FCGR3A$","MS4A7$",
                 "LYZ$","CD14$","MS4A1$","IRF7$","CST3$", "FCER1A$",
                 "PCNA$","TUBB$","NKG7$","GNLY$"
)
markerGenesFullnames <- c()
for (i in 1:length(markerGenes)){
  markerGenesFullnames <- append(markerGenesFullnames, allGenes[grep(markerGenes[i],allGenes)])
}

png("model1DrGene.png",width = 3000,height = 3000,res = 360)
plotDrGene(dim1 = model1Tsne[,1], dim2 = model1Tsne[,2], 
           counts = gene_counts, 
           rescale = TRUE)
dev.off()

### render heatmap
model1FactorizeMatrix <- factorizeMatrix(counts = pbmc4k_select, celda.mod = bestModel1)
model1TopGenes <- topRank(model1FactorizeMatrix$proportions$gene.states, n = 25)

model1TopGenesIx <- unique(unlist(model1TopGenes$index))
model1NormPbmc <- normalizeCounts(pbmc4k_select)

### plot heatmap with out gini filter
png("model1RenderCeldaHeatmap.png")
renderCeldaHeatmap(counts = model1NormPbmc[model1TopGenesIx,], z = bestModel1$z, 
                   y = bestModel1$y[model1TopGenesIx], normalize = NULL,
                   color_scheme = "divergent")
dev.off()

### compute the gini coefficient
png("model1Gini.png")
model1gini <- GiniPlot(counts = pbmc4k_select, celda.mod = bestModel1)
model1gini
dev.off()

### plot the heatmap with the gini filtered
model1FilteredStates <- model1gini$data$Transcriptional_States[model1gini$data$Gini_Coefficient > 0.3]
model1TopGenesFilteredIx <- unique(unlist(model1TopGenes$index[as.numeric(levels(model1FilteredStates))][as.numeric(model1FilteredStates)]))

png("model1RenderCeldaHeatmapGini.png")
renderCeldaHeatmap(model1NormPbmc[model1TopGenesFilteredIx,], z = bestModel1$z, 
                   y = bestModel1$y[model1TopGenesFilteredIx], 
                   normalize = NULL, color_scheme = "divergent")
dev.off()

model1CellType <- unlist(lookupGeneModule(counts = pbmc4k_select, model = bestModel1, gene = markerGenesFullnames))
write.csv(model1CellType, "model1CellType.csv")



