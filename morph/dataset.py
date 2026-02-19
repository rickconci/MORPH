import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import scanpy as sc
import pandas as pd
import time

from config import resolve_scdata_paths_df


class SCDataset(Dataset):
    """
    A PyTorch Dataset class for loading and processing single-cell perturbation data.
    
    Attributes:
        - dataset: Name of the dataset (match with the name in ./data/scdata_file_path.csv).
        - adata_path: Path to the single-cell AnnData file.
        - leave_out_test_set: List of perturbation targets to leave out for testing.
        - ptb_targets: List of perturbation targets in the dataset.
        - representation_type: Type of representation for the perturbation targets (e.g., 'Baseline', 'DepMap', etc.).
        - min_counts: Minimum counts for filtering perturbations.
        - gene_embs: Dictionary of gene embeddings (optional)
    """
    def __init__(self,
                base_dir=None,
                dataset_name=None,
                adata_path=None,
                leave_out_test_set=None,
                representation_type=None, representation_type_2=None, representation_type_3=None,
                gene_embs=None, gene_embs_2=None, gene_embs_3=None,
                min_counts=32,
                random_seed=12,
                use_hvg=False,
                n_top_genes=5000,
                fixed_genes=None):
        super(Dataset, self).__init__()
        
        self.seed = random_seed

        # Load perturbation embeddings from CSV only when not provided (standalone: pass gene_embs directly).
        need_csv = (
            (gene_embs is None and representation_type is not None)
            or (gene_embs_2 is None and representation_type_2 is not None)
            or (gene_embs_3 is None and representation_type_3 is not None)
        )
        if need_csv:
            if base_dir is None or dataset_name is None:
                raise ValueError(
                    "When embeddings are not provided, base_dir and dataset_name are required "
                    "to load them from data/perturb_embed_file_path.csv. "
                    "For standalone use, pass gene_embs (e.g. from --embedding_path) and representation_type."
                )
            embedding_file_df = pd.read_csv(os.path.join(base_dir, "data", "perturb_embed_file_path.csv"))
            embedding_file_df = resolve_scdata_paths_df(embedding_file_df)
            if gene_embs is None and representation_type is not None:
                gene_embs = self.load_embedding(embedding_file_df, dataset_name, representation_type)
            if gene_embs_2 is None and representation_type_2 is not None:
                gene_embs_2 = self.load_embedding(embedding_file_df, dataset_name, representation_type_2)
            if gene_embs_3 is None and representation_type_3 is not None:
                gene_embs_3 = self.load_embedding(embedding_file_df, dataset_name, representation_type_3)
        elif gene_embs is None and representation_type is not None:
            raise ValueError("Provide gene_embs (e.g. from --embedding_path) or base_dir+dataset_name to load embeddings.")
        
        # Get the list of perturbation targets leave out for testing
        ptb_leave_out_list = leave_out_test_set
        
        # Load adata
        adata = sc.read_h5ad(adata_path)
        print('Loaded adata from: ', adata_path)

        # Subset to fixed gene set (e.g. pretrained HVG) and preprocess
        if fixed_genes is not None:
            if hasattr(adata.X, 'toarray'):
                adata.X = adata.X.toarray()
            adata.X = np.asarray(adata.X, dtype=np.float32)
            totals = np.array(adata.X.sum(axis=1)).flatten()
            keep = totals > 0
            if (~keep).any():
                adata = adata[keep].copy()
            X = np.zeros((adata.n_obs, len(fixed_genes)), dtype=np.float32)
            for j, g in enumerate(fixed_genes):
                if g in adata.var_names:
                    X[:, j] = np.asarray(adata[:, g].X.flatten())
            adata = sc.AnnData(X, obs=adata.obs.copy(), var=pd.DataFrame(index=list(fixed_genes)))
            sc.pp.normalize_total(adata)
            sc.pp.log1p(adata)
            print('Subset to fixed_genes: %d genes' % len(fixed_genes))

        # Optionally subset to highly variable genes (expects raw counts input)
        elif use_hvg:
            if base_dir is None:
                raise ValueError("base_dir is required when use_hvg=True (for HVG cache).")
            if hasattr(adata.X, 'toarray'):
                adata.X = adata.X.toarray()
            adata.X = np.asarray(adata.X, dtype=np.float32)

            # QC: remove cells with zero total counts (avoids 0/0 in normalize_total)
            totals = np.array(adata.X.sum(axis=1)).flatten()
            keep = totals > 0
            if (~keep).any():
                n_removed = (~keep).sum()
                adata = adata[keep].copy()
                print('QC: removed %d cells with zero total counts (kept %d)' % (n_removed, adata.shape[0]))

            hvg_cache_dir = os.path.join(base_dir, 'data', 'hvg_cache')
            adata_stem = os.path.splitext(os.path.basename(adata_path))[0]
            hvg_cache_path = os.path.join(hvg_cache_dir, f'{adata_stem}_n{n_top_genes}.pkl')

            # Standard scanpy preprocessing: normalize → log1p
            sc.pp.normalize_total(adata)
            sc.pp.log1p(adata)

            if os.path.isfile(hvg_cache_path):
                with open(hvg_cache_path, 'rb') as f:
                    hvg_genes = pickle.load(f)
                genes_in_adata = [g for g in hvg_genes if g in adata.var_names]
                adata = adata[:, genes_in_adata].copy()
                print('Loaded HVG from cache: %s (%d genes)' % (hvg_cache_path, adata.shape[1]))
            else:
                print('Identifying HVGs...')
                time_start = time.time()
                sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, subset=True)
                print('Computed HVG and subset: n_top_genes=%d, %d genes' % (n_top_genes, adata.shape[1]))
                print('Time taken: %.1f seconds' % (time.time() - time_start))
                os.makedirs(hvg_cache_dir, exist_ok=True)
                with open(hvg_cache_path, 'wb') as f:
                    pickle.dump(adata.var_names.tolist(), f, protocol=pickle.HIGHEST_PROTOCOL)
                print('Saved HVG list to: ', hvg_cache_path)
        # Filter out the perturbation targets with counts < min_counts
        gene_counts = adata.obs['gene'].value_counts()
        gene_counts = gene_counts[gene_counts >= min_counts]
        print('Length of raw ptb_leave_out_list: ', len(ptb_leave_out_list))
        ptb_leave_out_list = [ptb for ptb in ptb_leave_out_list if ptb in gene_counts.index]
        print('Length of filtered ptb_leave_out_list (after removing gene with counts < ', min_counts ,'): ', len(ptb_leave_out_list))
        
        # Filter out perturbation targets that do not have embeddings from adata and ptb_leave_out_list
        if gene_embs is not None:
            if gene_embs_2 is not None:
                if gene_embs_3 is not None:
                    ptb_lst_with_embs = list(set(gene_embs.keys()) & set(gene_embs_2.keys()) & set(gene_embs_3.keys()))
                else:
                    ptb_lst_with_embs = list(set(gene_embs.keys()) & set(gene_embs_2.keys()))
            else:
                ptb_lst_with_embs = gene_embs.keys()
            print("Checking for perturbation targets without embeddings in adata...")
            
            # Get the single genes in adata
            single_genes = adata.obs['gene'].unique().tolist()
            single_genes = [gene for gene in single_genes if '+' not in gene]
            single_genes.remove('non-targeting')

            # Get the list of single genes without embeddings
            single_genes_without_embs = [gene for gene in single_genes if gene not in ptb_lst_with_embs]
            if len(single_genes_without_embs) > 0:
                print('Perturbation targets without embeddings: ', single_genes_without_embs, 'There are ', len(single_genes_without_embs), ' perturbations without embeddings')
                print('Original shape of adata: ', adata.shape)

                # Create a function to check if any gene in the perturbation (single or double) is in single_genes_without_embs
                def contains_genes_without_embeds(perturbation):
                    genes = perturbation.split('+')
                    return any(gene in single_genes_without_embs for gene in genes)
                
                # Apply the function to filter adata
                adata = adata[~adata.obs['gene'].apply(contains_genes_without_embeds)]
                print('Removed perturbations without embeddings from adata')
                print('New shape of adata: ', adata.shape)
                
                # Update ptb_leave_out_list by removing perturbations without embeddings
                ptb_leave_out_list = [ptb for ptb in ptb_leave_out_list if not contains_genes_without_embeds(ptb)]
                print('Length of filtered ptb_leave_out_list (after removing perturbations without embeddings): ', len(ptb_leave_out_list))
        
        self.ptb_leave_out_list = ptb_leave_out_list
        
        # Get list of perturbation targets
        ptb_targets = list(set(adata.obs['gene']))
        ptb_targets.remove('non-targeting')
        self.ptb_targets = ptb_targets
        print("There are", len(ptb_targets), "perturbation targets.")
        
        # Map the perturbation targets to the corresponding representation vectors
        ptb_adata = adata[adata.obs['gene']!='non-targeting'].copy()
        self.ptb_samples = ptb_adata.X
        self.ptb_names = ptb_adata.obs['gene'].values
        print("Mapping perturbation targets to the corresponding representation vectors with type:", representation_type)
        self.ptb_vector_dict, self.ptb_vectors_1, self.ptb_vectors_2 = map_ptb_features(ptb_targets, self.ptb_names,
                                                                                        representation_type=representation_type, 
                                                                                        gene_embs = gene_embs)
        
        if gene_embs_2 is not None:
            print("Mapping perturbation targets to the corresponding representation vectors with type:", representation_type_2)
            self.ptb_vector_dict_2, self.ptb_vectors_1_2, self.ptb_vectors_2_2 = map_ptb_features(ptb_targets, self.ptb_names,
                                                                                                  representation_type=representation_type_2, 
                                                                                                  gene_embs = gene_embs_2)
        else:
            self.ptb_vector_dict_2 = None
            self.ptb_vectors_1_2 = None
            self.ptb_vectors_2_2 = None

        if gene_embs_3 is not None:
            print("Mapping perturbation targets to the corresponding representation vectors with type:", representation_type_3)
            self.ptb_vector_dict_3, self.ptb_vectors_1_3, self.ptb_vectors_2_3 = map_ptb_features(ptb_targets, self.ptb_names,
                                                                                                  representation_type=representation_type_3, 
                                                                                                  gene_embs = gene_embs_3)
        else:
            self.ptb_vector_dict_3 = None
            self.ptb_vectors_1_3 = None
            self.ptb_vectors_2_3 = None

        del ptb_adata
        
        # Control samples
        self.ctrl_samples = adata[adata.obs['gene']=='non-targeting'].X.copy()

        # A random subset of control samples, matching in number to the perturbed samples, is selected for balanced comparison.
        np.random.seed(self.seed)
        self.rand_ctrl_samples = self.ctrl_samples[
            np.random.choice(self.ctrl_samples.shape[0], self.ptb_samples.shape[0], replace=True)
            ]
        del adata
    
    def load_embedding(self, embedding_file_df, dataset_name, representation_type):
        """Loads embedding file based on dataset and representation type."""
        if representation_type is None:
            return None
        
        embedding_file_subset = embedding_file_df[embedding_file_df['representation_type'] == representation_type]
        if embedding_file_subset.shape[0] == 1:
            embed_file = embedding_file_subset['file_path'].values[0]
        elif embedding_file_subset.shape[0] > 1:
            embed_file = embedding_file_subset[(embedding_file_subset['dataset_name'] == dataset_name)]['file_path'].values[0]
        else:
            return None
        
        with open(embed_file, 'rb') as file:
            print(f'Loading perturbation target embeddings from {embed_file}')
            return pickle.load(file)

    def __getitem__(self, item):
        # A control sample
        x = torch.from_numpy(self.rand_ctrl_samples[item].flatten()).double()
        
        # A perturbed sample
        y = torch.from_numpy(self.ptb_samples[item].flatten()).double()

        # The vector representation of the perturbation target (target 1)
        c_1 = torch.from_numpy(self.ptb_vectors_1[item]).double()
        
        # The vector representation of the perturbation target (target 2)
        c_2 = torch.from_numpy(self.ptb_vectors_2[item]).double()

        # Get perturbation target name
        ptb_name = self.ptb_names[item]

        if self.ptb_vector_dict_2 is not None:
            c_1_2 = torch.from_numpy(self.ptb_vectors_1_2[item]).double()
            c_2_2 = torch.from_numpy(self.ptb_vectors_2_2[item]).double()
            if self.ptb_vector_dict_3 is not None:
                c_1_3 = torch.from_numpy(self.ptb_vectors_1_3[item]).double()
                c_2_3 = torch.from_numpy(self.ptb_vectors_2_3[item]).double()
                return x, y, c_1, c_2, ptb_name, c_1_2, c_2_2, c_1_3, c_2_3
            else:
                return x, y, c_1, c_2, ptb_name, c_1_2, c_2_2
        else:
            return x, y, c_1, c_2, ptb_name
        
    def __len__(self):
        # return number of perturbed samples in the dataset
        return self.ptb_samples.shape[0]


def map_ptb_features(all_ptb_targets, ptb_ids, representation_type, gene_embs):    
    if representation_type == 'Baseline' and gene_embs is None:
        'This function maps the perturbation targets to binary vectors.'
        # get all single ptb_targets
        split_ptb_targets = [target for item in all_ptb_targets for target in item.split('+')]
        split_ptb_targets = list(set(split_ptb_targets))
        gene_embs = {target: np.eye(len(split_ptb_targets))[i] for i, target in enumerate(split_ptb_targets)}
    
    ptb_features_1 = []
    ptb_features_2 = []
    for id in ptb_ids:
        if "+" in id:
            id_1 = id.split("+")[0]
            id_2 = id.split("+")[1]
            feature_1 = gene_embs.get(id_1)
            feature_2 = gene_embs.get(id_2)
            ptb_features_1.append(feature_1)
            ptb_features_2.append(feature_2)
        else:
            feature = gene_embs.get(id)
            ptb_features_1.append(feature)
            ptb_features_2.append(np.full(feature.shape, np.nan))
    return gene_embs, np.vstack(ptb_features_1), np.vstack(ptb_features_2)