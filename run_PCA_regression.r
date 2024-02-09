library(ape)
library(tidyverse)
library(phylolm)

args = commandArgs(trailingOnly=TRUE)

num_PC = as.numeric(args[1])
num_genes = as.numeric(args[2])

print(num_genes)

fungi_tree = read.tree("input_data/fungi_tree.tre")
fungi_data = read_table("input_data/gene_expression_tpm_matrix_updated_Standard.LogNorm.tsv")

species_in_data = colnames(fungi_data)[2:ncol(fungi_data)]

num_species = fungi_data %>%
    select(-Protein) %>%
    as.data.frame() %>%
    apply(1,function(x){sum(!is.na(x))})

fungi_data_species = fungi_data[num_species>15,]

fungi_data_species = fungi_data_species %>% sample_n(num_genes)

res_no_PC = tibble()
res_PC = tibble()
for (i in 1:(nrow(fungi_data_species)-1)) {
    message(i)
    # if (i > 200) break
    for (j in (i+1):nrow(fungi_data_species)) {
        # message(j)
        # if (j > 200) break
        gene_x = fungi_data_species$Protein[i]
        gene_y = fungi_data_species$Protein[j]
        comparison = paste(gene_x,"vs",gene_y,sep="_")
        X = as.numeric(fungi_data_species[i,2:ncol(fungi_data)])
        Y = as.numeric(fungi_data_species[j,2:ncol(fungi_data)])
        
        good_species_index = which(!is.na(X)&!is.na(Y))
        bad_species = species_in_data[-good_species_index]
        cur_tree = drop.tip(fungi_tree,bad_species)
        
        projections = eigen(vcv(cur_tree))$vectors
        colnames(projections)=paste0("PC",1:length(cur_tree$tip.label))
        projections=bind_cols(tibble(Species=cur_tree$tip.label), as_tibble(projections))
        projections = projections[,1:num_PC]
        
        
        X = X[good_species_index]
        Y = Y[good_species_index]
        cur_species = species_in_data[good_species_index]
        all_data = inner_join(tibble(Species = cur_species, X = X, Y = Y),as_tibble(projections),by="Species") %>% column_to_rownames("Species")
        
        suppressWarnings(cur_lm <- phylolm(Y ~ X,all_data,phy=cur_tree))
        cur_p = summary(cur_lm)$coefficients["X","p.value"]
        cur_res = as_tibble(summary(cur_lm)$coefficients,rownames="Variable") %>% add_column(comparison=comparison)
        res_no_PC = bind_rows(res_no_PC,cur_res)
    
        suppressWarnings(cur_lm <- phylolm(Y ~ .,all_data,phy=cur_tree)) 
        cur_res = as_tibble(summary(cur_lm)$coefficients,rownames="Variable") %>% add_column(comparison=comparison)
        res_PC = bind_rows(res_PC,cur_res)
        

    }
    
}

res_no_PC = add_column(res_no_PC,PC=num_PC)
res_PC = add_column(res_PC,PC=num_PC)

write_tsv(res_no_PC,paste0("output_data/res_no_PC_",num_PC,".tsv"))
write_tsv(res_PC,paste0("output_data/res_PC_",num_PC,".tsv"))
