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

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#----------function for making heatmaps
def create_mat_plot(
    mat,
    axis_names,
    title,
    filename,
    xlab,
    ylab,
    cmap="Blues",
    filetype=".png",
    rx=0,
    ry=0,
    font_s=10,
    annot=True,
    show_plot=False,
):
    """Generates and saves a heatmap for 1D or 2D NumPy arrays without system TeX/dvipng dependencies."""
    # Ensure Matplotlib uses internal rendering (no external TeX/dvipng required)
    plt.rcParams["text.usetex"] = False

    # Convert 1D array to a 2D row matrix (1, N) for Seaborn
    mat = np.atleast_2d(mat)

    plt.figure()

    # Handle tick labels based on matrix dimensions
    if len(axis_names) > 0:
        xtick_labels = (
            axis_names if mat.shape[1] == len(axis_names) else "auto"
        )
        ytick_labels = (
            axis_names if mat.shape[0] == len(axis_names) else False
        )

        ax = sns.heatmap(
            mat,
            annot=annot,
            cmap=cmap,
            xticklabels=xtick_labels,
            yticklabels=ytick_labels,
        )
    else:
        ax = sns.heatmap(mat, annot=annot, cmap=cmap)

    # Set labels and title
    plt.title(title, fontsize=font_s)
    plt.xlabel(xlab, fontsize=font_s)
    plt.ylabel(ylab, fontsize=font_s)

    # Adjust tick font size and rotation
    plt.xticks(fontsize=font_s, rotation=rx)
    plt.yticks(fontsize=font_s, rotation=ry)

    # Save figure
    plt.tight_layout()
    if show_plot == True:
        plt.show()
    plt.savefig(f"{filename}.{filetype}", bbox_inches="tight")
    plt.close()

#-------------------kmer finder function
def ppe(vocab_sizes, topn, show_plot, cluster_id):
    dataloader = DataLoader(cluster_id=cluster_id)

    pos_train_file, neg_train_file = dataloader.import_data()

    # load positive and negative sequences
    pos_seqs = FileUtility.load_list(pos_train_file)
    neg_seqs = FileUtility.load_list(neg_train_file)

    print(f"Loaded and processed positive and negative ferredoxin samples for cluster {cluster_id}\n")

    # prepare labels and sequences
    seqs = [seq.lower() for seq in pos_seqs + neg_seqs]
    labels = [1] * len(pos_seqs) + [0] * len(neg_seqs)

    # -------------------------
    from make_representations.cpe_apply import CPE

    print(f"Beginning segmentation of sequences in cluster {cluster_id}\n")

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

    print(f"Completed segmentation of sequences in cluster {cluster_id}\n")

    # ----------------------------

    # top 50 motifs
    topn = topn
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
    FileUtility.save_obj('../output/matrices/' + '/sym_KL' + str(cluster_id), DIST)  # made need to change to include dataset

    # ------------------------------

    print(f"Creating dendogram for k-mers in cluster {cluster_id}\n")
    HC = HierarchicalClustering(DIST, [x[0] for x in vocab_binary]) #need to edit code to save fig
    motifs = vocab_binary
    #tree = HC.nwk

    figure_caption = f'Divergence between co-occurrence patterns of motifs for cluster {cluster_id}' 

    create_mat_plot(mat = DIST[0:topn,0:topn], 
                    axis_names = [x[0] for x in motifs[0:topn]],
                    title = figure_caption,
                    filename = f"../output/heatmaps/cluster_{cluster_id}",
                    xlab = f'Top {topn} motifs',
                    ylab = f'Top {topn} motifs',
                    annot=False,
                    rx=90,
                    show_plot=False)

    return vocab_binary

def multiplex_ppe(cluster_ids, vocab_sizes, topn = 50, show_plot = False, max_workers = 2):

    start = time.time()
    args_list = list(range(cluster_ids + 1))

    print(f"number of cores to be used {max_workers}\n")

    with ProcessPoolExecutor(max_workers = max_workers) as executor:
        futures = [
            executor.submit(ppe, vocab_sizes, topn, show_plot, arg) for arg in args_list
        ]

        results = []
        for f in as_completed(futures):
            try:
                results.append(f.result())
            except Exception as e:
                print(f"Task failed with error: {e}")

    end = time.time()
    print(f"process completed in: {end - start:2f} seconds")

    print("saving results to output folder")
    with open("../output/motifs.csv", mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["kmer", "pval", "cluster"])

        # Write each tuple inside the sublists to the CSV
        for sublist in results:
            writer.writerows(sublist)

    return results