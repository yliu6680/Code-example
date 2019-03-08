library(celda)
library(Matrix)
library(SummarizedExperiment)
library(MAST)

pbmc8k_select <- readRDS("pbmc8k_select.rds")
dim(pbmc8k_select)

k_to_test <- seq(15,24,by = 1)
l_to_test <- seq(50,70,by = 10)

k_to_test
l_to_test

t1 <- c(date())
t1
pbmc8k_res1 <- celda(pbmc8k_select, K = k_to_test, L = l_to_test, 
                     cores = 8, model = "celda_CG", nchains = 8, max.iter = 200)
t2 <- c(t1,date())

saveRDS(pbmc8k_res1, "8k_model_l15-24_k50-70.rds")

t2

write.csv(t2,"run_time.csv")
