
library(Seurat)
library(RColorBrewer)
library(gprofiler2)
library(dplyr)
library(ggplot2)

options(stringsAsFactors=FALSE)
setwd("S:\\rkok\\papers\\2024 Rutger - Cell type prediction\\Data\\Serra2019 scRNAseq")
getwd()

##load data
data <- read.csv("GSE115956_raw_UMI_cnt.csv",header=T) #the data was not log2 transformed

#checking the data
data[1:5,1:5]

##replace the gene od into gene name
gene.symbols <- gconvert(query=as.vector(data$Ensembl),organism="mmusculus",target="ENTREZGENE",filter_na = F)
gene_name <- c()
for (i in 1:nrow(data)){
  temp <- gene.symbols$name[gene.symbols$input == data$Ensembl[i]]
  if (length(temp) == 1){
    gene_name <- c(gene_name, temp)
  } else { 
    if (length(temp) > 1){
      gene_name <- c(gene_name, temp[1])
    } else { gene_name <- c(gene_name, NA)}
  }
}

data$Ensembl <- toupper(gene_name)

##remove any duplicates in the gene names with the lower number of counts
sum(duplicated(data$Ensembl))
data$Mean_row <- rowMeans(data[,2:(ncol(data)-1)],na.rm=T)
data <- data %>%
  group_by(Ensembl) %>%
  filter(Mean_row == max(Mean_row))
data <- as.data.frame(ungroup(data))

#remove duplicated if neccessary
if (sum(duplicated(data$Ensembl)) > 0){
  data <- data[-duplicated(data$Ensembl),]
}

##asign the rownames and remove the first and the last column
data <- data[!is.na(data$Ensembl),]
rownames(data) <- data$Ensembl
data <- data[,-c(1,ncol(data))]

#get the time out
colnames(data)[1:5]
day_code <- sapply(strsplit(colnames(data)[-1], ".C"), function(x) x[1])
table(day_code)


#create the seurat file
prot.seurat <- CreateSeuratObject(counts=data[,-1], project = "ng1", min.cells = 3, min.features = 200)
prot.seurat$day_code <- day_code

##cleaning the data
prot.seurat[["percent.mt"]] <- PercentageFeatureSet(prot.seurat, pattern = "^MT-")
VlnPlot(prot.seurat, features = c("nFeature_RNA", "nCount_RNA", "percent.mt"), ncol = 3)


prot.seurat <- subset(prot.seurat, subset = nFeature_RNA > 200 & nFeature_RNA < 6000 & percent.mt < 15)

#normalize the data http://127.0.0.1:40031/graphics/8c30cff7-c954-4d0a-9630-a40667471217.png
prot.seurat <- NormalizeData(prot.seurat, normalization.method = "LogNormalize", scale.factor = 10000)

#check the data
rownames(prot.seurat)[1:10]

##processing the data
prot.seurat <- FindVariableFeatures(prot.seurat, selection.method = "vst", nfeatures = 2000)
prot.seurat <- ScaleData(prot.seurat, features = rownames(prot.seurat))
prot.seurat <- RunPCA(prot.seurat, features = VariableFeatures(object = prot.seurat))
DimPlot(prot.seurat, reduction = "pca") + NoLegend()

##doing teh clustering
ElbowPlot(prot.seurat)
prot.seurat <- FindNeighbors(prot.seurat, dims = 1:10)
prot.seurat <- FindClusters(prot.seurat, resolution = 0.8)

##plotting the UMAP
prot.seurat <- RunUMAP(prot.seurat, dims = 1:10)
DimPlot(prot.seurat, reduction = "umap")

##plot umap with the day_code on top
DimPlot(prot.seurat, group.by="day_code", reduction="umap")
DimPlot(prot.seurat, reduction = "umap", label = TRUE,label.size = 6, pt.size = 1) + NoLegend()

#plotting the genes to define which cell type is which
genes_to_plot <- c("LGR5","ASCL2","AXIN2","CD44","KRT20","KLF4","HES1","ATOH1","DLL1","DLL4", "REG4","MYC")
FeaturePlot(prot.seurat, features = genes_to_plot, reduction="umap")

##Paneth cell markers
FeaturePlot(prot.seurat, features = c("REG4","ATOH1","DLL1","LYZ1","DEFA5"), reduction="umap")

##ent cell markers
FeaturePlot(prot.seurat, features = c("KRT20","ALPI","SLC5A1","VIL1", "SIS","ALDOB", "APOC3"), reduction="umap")

##goblet cell markers
FeaturePlot(prot.seurat, features = c("MUC2","TFF3","GOB5", "AGR2"), reduction="umap")

##enteroendorine cell markers
FeaturePlot(prot.seurat, features = c("CHGA", "CHGB"), reduction="umap")

##TA cell markers
FeaturePlot(prot.seurat, features = c("MKI67", "PCNA","TOP2A","CDK1","CENPF","BUB1","MYC", "UBE2C"), reduction="umap")

##TA cell markers (from the paper)
FeaturePlot(prot.seurat, features = c("NME2", "ERH","PPIL1","LSM2","NXT1", "POP1", "WDR4", 
                                      "NUP107", "NOC2L","USP1"), reduction="umap")

##Stem cell markers
FeaturePlot(prot.seurat, features = c("LGR5", "OLFM4","ASCL2","AXIN2","SOX9"), reduction="umap")

##plotting the epethial markers
FeaturePlot(prot.seurat, features = c("EPCAM", "CDH1","KRT8"), reduction="umap")

##plotting the Tuff markers
FeaturePlot(prot.seurat, features = c("DCLK1", "PTGS1","TRPM5","RGS13"), reduction="umap")

##naming the clusters
new.cluster.ids <- c("TA_d3","Early ENT_d5","Stem_d3","ENT 2_d5","Stem_d5","TA_d5",
                     "ENT 1_d5", "TA 1_d3","Panneth_Gob_d3/5","Early ENT_d5","TA 2_d3")
names(new.cluster.ids) <- levels(prot.seurat)
levels(prot.seurat)
prot.seurat <- RenameIdents(prot.seurat, new.cluster.ids)
DimPlot(prot.seurat, reduction = "umap", label = TRUE,label.size = 5, pt.size = 1) + NoLegend()


###running the TA score and STEM score and make violin plots to compare how they are different
stem_genes <- c("LGR5", "OLFM4","ASCL2","AXIN2")
ta_genes <- c("NME2", "ERH","PPIL1","LSM2","NXT1", "POP1", "WDR4", 
              "NUP107", "NOC2L","USP1", "SYCE2","DAZAP1","SMCHD1","NUP155","ZFP808", "PRPF19","RTEL1", "ENKD1", "KNTC1")
proli_genes <- c("MKI67", "PCNA", "TOP2A", "CDK1", "CCNB1", "CCNB2", "CCNA2", "CCNE1", "CCNE2", "CDC20", "CDC45", "CDC6", 
                 "CDT1", "AURKA", "AURKB", "PLK1", "BUB1", "BUB1B", "CENPF", "CENPE", "MCM2", "MCM3", 
                 "MCM4", "MCM5", "MCM6", "MCM7", "TYMS", "RRM1", "RRM2", "E2F1", "MYBL2", "MYC", "TFDP1", 
                 "UBE2C", "ESPL1", "KIF11", "KIF20A")
enterocyte_genes <- c("ALPI", "SIS","ALDOB", "APOC3")
gene_list <- list(STEM_scores = stem_genes, TA_scores = ta_genes,
                  Proli_scores = proli_genes, EC_scores = enterocyte_genes)


prot.seurat <- AddModuleScore(prot.seurat, 
                              features = gene_list, 
                              name = "ModuleScore",
                              ctrl = 100)  # Use 100 control genes for normalization

##rename the scores
colnames(prot.seurat@meta.data)[colnames(prot.seurat@meta.data) == "ModuleScore1"] <- "Stemness_Scores"
colnames(prot.seurat@meta.data)[colnames(prot.seurat@meta.data) == "ModuleScore2"] <- "TA_Scores"
colnames(prot.seurat@meta.data)[colnames(prot.seurat@meta.data) == "ModuleScore3"] <- "Proliferation_Scores"
colnames(prot.seurat@meta.data)[colnames(prot.seurat@meta.data) == "ModuleScore4"] <- "EC_Scores"


prot.seurat@active.ident <- factor(Idents(prot.seurat), 
                                   levels = c("TA_d3","Early ENT_d5","Stem_d3","ENT 2_d5","Stem_d5","TA_d5",
                                              "ENT 1_d5", "TA 1_d3","Panneth_Gob_d3/5","TA 2_d3"))  # Your order

VlnPlot(prot.seurat, features = "Stemness_Scores",
        idents = c("Stem_d3", "Stem_d5", "TA 1_d3", "TA 2_d3", "TA_d5"))

VlnPlot(prot.seurat, features = "TA_Scores",
        idents = c("Stem_d3", "Stem_d5", "TA 1_d3", "TA 2_d3", "TA_d5"))


VlnPlot(prot.seurat, features = "Proliferation_Scores",
        idents = c("Stem_d3", "Stem_d5", "TA 1_d3", "TA 2_d3", "TA_d5"))



###export the score for each cell types
counts <- GetAssayData(prot.seurat, layer = "counts")
export_dt_all <- data.frame(cell_ID = colnames(prot.seurat),
                            cell_type = Idents(prot.seurat),
                            TA_score = prot.seurat@meta.data$TA_Scores,
                            Stem_score = prot.seurat@meta.data$Stemness_Scores,
                            EC_score = prot.seurat@meta.data$EC_Scores,
                            Proliferation_score = prot.seurat@meta.data$Proliferation_Scores
                            )

sum(!is.na(export_dt_all$cell_type))

write.csv(export_dt_all, "TA_Stem score_all cells.csv")

###dont for now!
















table(Idents(prot.seurat), useNA = "always")  # Check if any NA values exist
nrow(Embeddings(prot.seurat, "umap"))  # Number of cells in UMAP
nrow(prot.seurat@meta.data)  # Total number of cells




unique((prot.seurat@active.ident))

length((prot.seurat$seurat_clusters))

(prot.seurat$seurat_clusters)[1:10]
(prot.seurat@active.ident)[1:10]













#identify DEGs
pbmc.markers <- FindAllMarkers(prot.seurat, only.pos = F)
pbmc.markers %>%
  group_by(cluster) %>%
  dplyr::filter(avg_log2FC > 1) %>%
  slice_head(n = 20) %>%
  ungroup() -> top10
DoHeatmap(prot.seurat, features = top10$gene) + NoLegend()

#only plot the genes of paneth cells
pbmc.markers %>%
  group_by(cluster) %>%
  dplyr::filter(avg_log2FC > 0.5 & cluster == "Paneth_d3/5") %>%
  ungroup() -> top10
DoHeatmap(prot.seurat, features = top10$gene) + NoLegend() +
  theme(text = element_text(size = 10))

##check which metabolic markers are found in them
#choose the genes you want to plot
#get the metabolic genes
met_gene <- read.csv("cell_type_marker_NG.csv",header=T)
met_gene <- paste(met_gene$geneSymbolmore1[met_gene$cellName %in% 
                                             c("HM_GLYCOLYSIS","HM_HYPOXIA", "HM_FATTY_ACID_METABOLISM","HM_OXIDATIVE_PHOSPHORYLATION" )], collapse=",")
met_gene <- unique(strsplit(met_gene, split=",")[[1]])

#remove the metabolic genes in these markers
top10$gene[top10$gene %in% met_gene]





#get the data of paneth out and do the find all markers again
seurat_subset <- subset(prot.seurat, idents = "Paneth_d3/5")  # Replace "1" with your cluster ID
##ploting the only paneth cells and group by the day 3 and day 5
DoHeatmap(seurat_subset, features = top10$gene, group.by="day_code") + NoLegend()+
  theme(text = element_text(size = 8.5))
