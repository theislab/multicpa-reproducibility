import sys
sys.path.append("../")

import torch
import pickle
import numpy as np
import pandas as pd
from itertools import compress
import warnings
import os
import scanpy as sc
from sklearn.metrics import r2_score
from MultiCPA.helper import model_importer


os.chdir('./../')
print(os.getcwd())
input('ok?')

DATAM = {'papalexi': {'raw_dir_name': 'Papalexi2021', 'adata_name': 'Papalexi21_prep.h5ad', 'seml': 'mpert12',
                    'nt_condition': "THP-1_NT_1.0"},
        'wessel': {'raw_dir_name': 'Wessel2022', 'adata_name': 'Wessel22_prep.h5ad', 'seml': 'mpert15',
                  'nt_condition': "THP-1_NT_1"},
        'wessel_ood': {'raw_dir_name': 'Wessel2022', 'adata_name': 'Wessel22_prep.h5ad', 'seml': 'mpert16',
                  'nt_condition': "THP-1_NT_1"},
        'wessel_ood_noprot': {'raw_dir_name': 'Wessel2022', 'adata_name': 'Wessel22_noprot_prep.h5ad', 'seml': 'mpert17',
                  'nt_condition': "THP-1_NT_1"},
        'papalexi_all': {'raw_dir_name': 'Papalexi2021', 'adata_name': 'Papalexi21_prep.h5ad', 'seml': 'mpert18',
                    'nt_condition': "THP-1_NT_1.0"},
       }
dataset = 'wessel_ood'
DATA = DATAM[dataset]

print('data')

raw_data_dir = f"/storage/groups/ml01/workspace/kemal.inecik/{DATA['raw_dir_name']}"
adata = sc.read(os.path.join(raw_data_dir, DATA['adata_name']))

adata.X = adata.layers['counts'].copy()
adata.obsm["protein_expression"] = adata.obsm["protein_expression_raw"].copy()
del adata.layers['counts']
del adata.obsm["protein_expression_raw"]

adata.obsm['protein_expression'] = pd.DataFrame(adata.obsm['protein_expression'].astype(int), index=adata.obs.index, columns=adata.uns['protein_names'])
adata.X = adata.X.A.astype(int)
de_genes = adata.uns["rank_genes_groups_cov"].copy()
del adata.uns
del adata.var
adata.obs = adata.obs[['cov_drug_dose_name', 'control', 'split']]
adata.obs.columns = ['batch', 'control', 'split']
adata.uns["rank_genes_groups_cov"] = de_genes

with open(f"pretrained_models/TotalVI_{DATA['raw_dir_name']}_model.pt", 'rb') as f1:
    model = pickle.load(f1)
with open(f"pretrained_models/TotalVI_{DATA['raw_dir_name']}_adata.pt", 'rb') as f2:
    adata_train = pickle.load(f2)

mulpert_models = pd.read_pickle(f"/storage/groups/ml01/workspace/kemal.inecik/seml/{DATA['seml']}/{DATA['seml']}_pickled.pk")
spt = 'test'
focus_metric_1 = f"{spt}_mean_score_genes_last"
focus_metric_2 = f"{spt}_mean_score_proteins_last"

mulpert_models = mulpert_models[(mulpert_models['status'] == 1)]
mulpert_models = mulpert_models[(mulpert_models[focus_metric_1] > 0) & (mulpert_models[focus_metric_2] > 0)]
mulpert_models["overall_best"] = np.sqrt(mulpert_models[focus_metric_1]**2 + mulpert_models[focus_metric_2]**2)

top_top = 5
mulpert_models = mulpert_models.sort_values(by=["overall_best"], ascending=False)[:top_top]
mulpert_models[[f"{'training'}_mean_score_genes_last", f"{'training'}_mean_score_proteins_last", f"{'test'}_mean_score_genes_last", f"{'test'}_mean_score_proteins_last", "overall_best"]]

chosen_model = mulpert_models.index[0]
print(chosen_model)

chosen_model_entry = mulpert_models.loc[chosen_model]
cm_autoencoder, cm_datasets, cm_state, cm_history, cm_hypers = \
    model_importer(chosen_model, mulpert_models, 
                   model_dir=f"/storage/groups/ml01/workspace/kemal.inecik/seml/{DATA['seml']}/_last/", 
                   #model_dir=f"/home/icb/kemal.inecik/work/codes/mulpert/seml/_last/", 
                   dataset_relative_to="/home/icb/kemal.inecik/work/codes/mulpert")
print(cm_hypers['adversary_wd'] == mulpert_models.loc[chosen_model]['config.model.model_args.hparams.adversary_wd'])

print('model')

import logging
logging.getLogger('scvi').setLevel(logging.WARNING)

def delete_percent(percentage_gene, percentage_protein):
    print(f"%{percentage_gene}G%{percentage_protein}P")

    adata_raw = model.adata.X.copy()
    adata_raw_obsm = adata_train.obsm['protein_expression'].copy()

    def random_select_gene(gene_array, perc, seed=0):
        np.random.seed(perc + seed)
        a = np.arange(len(gene_array))
        np.random.shuffle(a)
        how_many = int(np.ceil(len(gene_array) / 100 * perc))
        return gene_array[a[:how_many]]

    def random_select_protein(protein_array, perc, seed=0):
        np.random.seed(perc + seed)
        a = np.arange(len(protein_array))
        np.random.shuffle(a)
        how_many = int(np.ceil(len(protein_array) / 100 * perc))
        return protein_array[a[:how_many]]
    
    nt_condition = DATA['nt_condition']
    set_test = set(model.test_indices)

    true_genes = pd.DataFrame(adata_train.X, index=adata_train.obs.index, columns=adata_train.var.index)
    true_proteins = adata_train.obsm['protein_expression'].copy()
    scale_factors = np.array(np.sum(true_genes, axis=1)).ravel()

    test_bools = np.array([i in set_test for i in range(len(adata_train))])
    NT_cells = (adata_train.obs['batch'] == nt_condition) & test_bools
    scale_factors = scale_factors[NT_cells]

    r2genes, r2proteins, r2genes_de = [], [], []
    perts_tot = []

    perts = sorted(np.unique(adata_train.obs["batch"]))
    for ind, perturbation in enumerate(perts):
        #print(f"{ind + 1}/{len(perts)}", end=' ')

        if perturbation != nt_condition:
            ii = adata_train.obs[(adata_train.obs['batch'] == perturbation) & test_bools].index
            if len(ii) > 20:
                perts_tot.append(perturbation)

                #modify adata_train
                #genes_to_process = adata_train.uns["rank_genes_groups_cov"][perturbation]
                genes_to_process = adata_train.var.index
                genes_to_remove = random_select_gene(genes_to_process, percentage_gene)
                model.adata[:, genes_to_remove] = np.zeros((len(adata_train), len(genes_to_remove)))
                
                proteins_to_process = model.adata.obsm["protein_expression"].columns
                proteins_to_remove = random_select_protein(proteins_to_process, percentage_protein)
                model.adata.obsm['protein_expression'].loc[:, proteins_to_remove] = np.zeros_like(model.adata.obsm['protein_expression'].loc[:, proteins_to_remove])

                idx_de = adata_train.var.index.isin(adata_train.uns["rank_genes_groups_cov"][perturbation])

                # get perturbed batch from NT cells
                gene_means, protein_means = model.get_normalized_expression(
                    indices=NT_cells, 
                    transform_batch=perturbation,
                    include_protein_background=True,
                    sample_protein_mixing=False,
                    return_mean=True,
                    scale_protein=False,
                )
                gene_means = (gene_means.T * scale_factors).T

                # get true perturbed cells
                ii_p = true_proteins.loc[ii]
                ii_g = true_genes.loc[ii]

                # Get mean value for each gene for a given perturbation
                predicted_genes_mean = gene_means.to_numpy().mean(0)
                predicted_proteins_mean = protein_means.to_numpy().mean(0)
                true_genes_mean = ii_g.to_numpy().mean(0)
                true_proteins_mean = ii_p.to_numpy().mean(0)

                # Calculate r^2 score for both
                r2genes.append(r2_score(predicted_genes_mean, true_genes_mean))
                r2proteins.append(r2_score(predicted_proteins_mean, true_proteins_mean))
                r2genes_de.append(r2_score(predicted_genes_mean[idx_de], true_genes_mean[idx_de]))

                model.adata.X = adata_raw.copy()
                adata_train.obsm['protein_expression'] = adata_raw_obsm.copy()
    del adata_raw, adata_raw_obsm
        
    # For Mulpert
    dataset = cm_datasets["test_treated"]
    genes_control = cm_datasets["test_control"].raw_genes.clone()
    proteins_control = cm_datasets["test_control"].raw_proteins.clone()

    num, dim_genes = genes_control.size(0), genes_control.size(1)
    dim_proteins = proteins_control.size(1)

    genes_control_original = cm_datasets["test_control"].raw_genes.clone()
    proteins_control_original = cm_datasets["test_control"].raw_proteins.clone()

    def random_select_gene(len_gene_array, perc, seed=0):
        np.random.seed(perc + seed)
        a = np.arange(len_gene_array)
        np.random.shuffle(a)
        how_many = int(np.ceil(len_gene_array / 100 * perc))
        return a[:how_many]

    def random_select_protein(len_protein_array, perc, seed=0):
        np.random.seed(perc + seed)
        a = np.arange(len_protein_array)
        np.random.shuffle(a)
        how_many = int(np.ceil(len_protein_array / 100 * perc))
        return a[:how_many]

    mean_score_proteins = []
    mean_score_genes = []
    mean_score_genes_de = []
    perts_mul = []
    for ind, pert_category in enumerate(sorted(np.unique(dataset.pert_categories))):
        de_idx = np.where(dataset.var_names.isin(np.array(dataset.de_genes[pert_category])))[0]
        idx = np.where(dataset.pert_categories == pert_category)[0]
        if len(idx) > 20:

            perts_mul.append(pert_category)
            #print(f"{ind + 1}/{len(np.unique(dataset.pert_categories))}", end=' ')
            emb_drugs = dataset.drugs[idx][0].view(1, -1).repeat(num, 1).clone()
            emb_cts = dataset.cell_types[idx][0].view(1, -1).repeat(num, 1).clone()

            # modify gene_control here
            genes_to_remove = random_select_gene(len(adata.var.index), percentage_gene)
            genes_control_ = genes_control.clone().numpy()
            genes_control_[:, genes_to_remove] = np.zeros((len(genes_control_), len(genes_to_remove)))
            genes_control = torch.Tensor(genes_control_)

            proteins_to_remove = random_select_protein(len(adata.obsm["protein_expression"].columns), percentage_protein)
            proteins_control_ = proteins_control.clone().numpy()
            proteins_control_[:, proteins_to_remove] = np.zeros((len(proteins_control_), len(proteins_to_remove)))
            proteins_control = torch.Tensor(proteins_control_)

            #print(int(genes_control.clone().numpy().sum()))

            gene_predictions, protein_predictions = cm_autoencoder.predict(genes_control, emb_drugs, emb_cts, proteins_control)
            gene_predictions = gene_predictions.detach().cpu()
            protein_predictions = protein_predictions.detach().cpu()

            mean_predict_genes = gene_predictions[:, :dim_genes]
            mean_predict_proteins = protein_predictions[:, :dim_proteins]

            y_true_genes = dataset.raw_genes[idx, :].numpy()
            yt_m_genes = y_true_genes.mean(axis=0)
            yp_m_genes = np.array(mean_predict_genes.mean(0))
            mean_score_genes.append(r2_score(yt_m_genes, yp_m_genes))
            mean_score_genes_de.append(r2_score(yt_m_genes[de_idx], yp_m_genes[de_idx]))

            y_true_proteins = dataset.raw_proteins[idx, :].numpy()
            yt_m_proteins = y_true_proteins.mean(axis=0)
            yp_m_proteins = mean_predict_proteins.mean(0)
            mean_score_proteins.append(r2_score(yt_m_proteins, yp_m_proteins))

            genes_control = genes_control_original.clone()
            proteins_control = proteins_control_original.clone()
    del genes_control_original, proteins_control_original, genes_control, proteins_control     

    perts_mul_filter = [i in perts_tot for i in perts_mul]
    perts_tot_filter = [i in perts_mul for i in perts_tot]
    perts_mul = list(compress(perts_mul, perts_mul_filter))
    perts_tot = list(compress(perts_tot, perts_tot_filter)) 

    r2genes = list(compress(r2genes, perts_tot_filter)) 
    r2genes_de = list(compress(r2genes_de, perts_tot_filter)) 
    r2proteins = list(compress(r2proteins, perts_tot_filter)) 
    mean_score_genes = list(compress(mean_score_genes, perts_mul_filter))
    mean_score_genes_de = list(compress(mean_score_genes_de, perts_mul_filter))
    mean_score_proteins = list(compress(mean_score_proteins, perts_mul_filter))

    df_tot = pd.DataFrame([perts_tot, r2genes, r2genes_de, r2proteins]).T
    df_mul = pd.DataFrame([perts_mul, mean_score_genes, mean_score_genes_de, mean_score_proteins]).T
    df_cols_ = ["Perturbation", "Genes", "DE Genes", "Proteins"]
    df_tot.columns = df_cols_
    df_mul.columns = df_cols_
    df_tot["Model"] = "TotalVI"
    df_mul["Model"] = "MulPert"
    df_all = pd.concat([df_tot, df_mul])
    df_all["Dataset"] = DATA['raw_dir_name']

    df_ = pd.DataFrame()
    for m in ["Genes", "DE Genes", "Proteins"]:
        df_a = df_all[['Perturbation', 'Model', 'Dataset', m]]
        df_a.columns = ['Perturbation', 'Model', 'Dataset', 'Score']
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            df_a['Feature Set'] = m
        df_ = pd.concat([df_, df_a])
    df_all = df_.copy()
    del df_
    df_all.reset_index(inplace=True, drop=True)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        df_all["Genes Removed Percantage"] = percentage_gene
        df_all["Proteins Removed Percantage"] = percentage_protein
    #torch.cuda.empty_cache()
    #torch.clear_autocast_cache()
    
    return df_all
print('ok')

# Run Papalexi First
if DATA['raw_dir_name'] == 'Papalexi2021':
    df_del = pd.DataFrame()
    print('here')
else:
    df_del = pd.read_pickle(f"/home/icb/kemal.inecik/work/mulpert_totalvi_vs_mulpert_remove_percentages_papalexi_df.pk")
    print('here2')

for i_del in list(range(0, 110, 10)):
    for j_del in list(range(0, 110, 10)):
        print((i_del, j_del))
        df_del_p = delete_percent(i_del, j_del)
        df_del = pd.concat([df_del, df_del_p])

df_del.reset_index(inplace=True, drop=True)
df_del.to_pickle(f"/home/icb/kemal.inecik/work/mulpert_totalvi_vs_mulpert_remove_percentages_df.pk")



