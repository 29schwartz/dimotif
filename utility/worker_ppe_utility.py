import sys
import tqdm
import numpy as np
import csv

sys.path.append('../')
from data_preprocessing.positive_negative_sampling import DataLoader
from chi2analysis.chi2analysis import Chi2Analysis
from sklearn.feature_extraction.text import TfidfVectorizer
from utility.math_utility import get_sym_kl_rows
from utility.file_utility import FileUtility
from clustering.hierarchical import HierarchicalClustering
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

def ppe(vocab_sizes, cluster_id=1):
    dataloader = DataLoader(cluster_id=cluster_id)

    pos_train_file, neg_train_file = dataloader.import_data()

    # load positive and negative sequences
    pos_seqs = FileUtility.load_list(pos_train_file)
    neg_seqs = FileUtility.load_list(neg_train_file)

    print(f"Loaded and processed positive and negative ferredoxin samples for cluster {cluster_id}")

    # prepare labels and sequences
    seqs = [seq.lower() for seq in pos_seqs + neg_seqs]
    labels = [1] * len(pos_seqs) + [0] * len(neg_seqs)

    # -------------------------
    from make_representations.cpe_apply import CPE

    print(f"Beginning segmentation of sequences in cluster {cluster_id}")

    segmented_seqs = []
    for i, vocab in tqdm.tqdm(enumerate(vocab_sizes)):
        f = open('../data_config/swissprot_ppe', 'r')
        CPE_Applier = CPE(f, separator='', merge_size=vocab)
        for idx, seq in enumerate(seqs):
            if i == 0:
                segmented_seqs.append([CPE_Applier.segment(seq)])
            else:
                segmented_seqs[idx] += [CPE_Applier.segment(seq)]
    extended_sequences = [' '.join(l) for l in segmented_seqs]
    possible_segmentations = ['@@@'.join(l) for l in segmented_seqs]

    print(f"Completed segmentation of sequences in cluster {cluster_id}")

    # ----------------------------

    # top 50 motifs
    topn = 50
    cpe_vectorizer = TfidfVectorizer(use_idf=False, analyzer='word',
                                     norm=None, stop_words=[], lowercase=True, binary=False, tokenizer=str.split)

    tf_vec = cpe_vectorizer.fit_transform(extended_sequences)
    vocab = cpe_vectorizer.get_feature_names_out()
    CH = Chi2Analysis(tf_vec, labels, vocab)
    vocab_binary = [(x[0], x[2], cluster_id) for x in CH.extract_features_fdr('../datasets/' + '/motifs.txt',
                                                                              N=topn, alpha=5e-2,
                                                                              direction=False,
                                                                              allow_subseq=False,
                                                                              binarization=True,
                                                                              remove_redundant_markers=False) if
                    x[1] > 0]
    
    #print()
    #print('motif', '\t', 'p-value')
    #print('=====================')
    #for motif, pval, cluster_id in vocab_binary:
    #    print(motif, '\t', pval, '\t', cluster_id)
    # -----------------------------

    idxs = np.array([np.where(vocab == v[0])[0][0] for v in vocab_binary])
    pos_matrix = tf_vec.toarray()[0:len(pos_seqs), idxs]

    # it saves the co-occurance matrix in the output directory and sym_KL.pickle
    DIST = get_sym_kl_rows(pos_matrix.T)
    FileUtility.save_obj('../datasets/PPE_input_datasets/' + 'cluster_' + str(cluster_id) + '/sym_KL', DIST)  # made need to change to include dataset

    # ------------------------------

    print(f"Creating dendogram for k-mers in cluster {cluster_id}")
    HC = HierarchicalClustering(DIST, [x[0] for x in vocab_binary]) #need to edit code to save fig
    motifs = vocab_binary
    #tree = HC.nwk

    return vocab_binary

def multiplex_ppe(cluster_ids, vocab_sizes, max_workers = 2):

    start = time.time()
    args_list = list(range(cluster_ids + 1))

    print(f"number of cores to be used {max_workers}")

    with ProcessPoolExecutor(max_workers = max_workers) as executor:
        futures = [
            executor.submit(ppe, vocab_sizes, arg) for arg in args_list
        ]

        results = []
        for f in as_completed(futures):
            try:
                results.append(f.result())
            except Exception as e:
                print(f"Task failed with error: {e}")

    end = time.time()
    print(f"process completed in: {end - start:2f} seconds")

    with open("./output.csv", mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["kmer", "pval", "cluster"])

        # Write each tuple inside the sublists to the CSV
        for sublist in results:
            writer.writerows(sublist)

    return results