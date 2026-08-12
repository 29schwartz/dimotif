# This function is for creating the test data, this should be a separate function or script need to think about how to organize it.
import pandas as pd
import os
from sklearn.model_selection import train_test_split

class DataSampling:
    def __init__(self,
                 df_filepath = 'data/foldseek_umap_clusters.csv',
                 output_path = '../datasets/PPE_input_datasets/',
                 train_prop: float = 0.8):
        self.df_filepath = df_filepath
        self.output_path = output_path
        self.train_prop = train_prop

    # input max number of clusters here
    def cluster_sampling(self):
        df = pd.read_csv(self.df_filepath)

        cluster_ids = df['cluster'].nunique()

        for id in range(cluster_ids):

            folder_path = os.path.join(self.output_path, "cluster_" + str(id) + "/")

            try:
                # os.makedirs with exist_ok=True will not raise an error if the folder exists
                os.makedirs(folder_path, exist_ok=True)
                print(f"Folder ready at: {os.path.abspath(folder_path)}")
            except OSError as e:
                print(f"Error creating folder '{folder_path}': {e}")

            unique_rows_df = df.drop_duplicates(subset=['protein_ID'])

            # filter based on cluster number
            cluster = unique_rows_df[unique_rows_df['cluster'] == id]
            notcluster = unique_rows_df[unique_rows_df['cluster'] != id]

            # sample cluster subset based on given percentage
            cluster_train_df = cluster.sample(frac=self.train_prop, random_state=42)
            cluster_test_df = cluster.drop(cluster_train_df.index)

            # sample non-cluster subset based on given percentage
            notcluster_train_df = notcluster.sample(frac=self.train_prop, random_state=42)
            notcluster_test_df = notcluster.drop(notcluster_train_df.index)

            # write out cluster as fasta
            # train
            with open(os.path.join(folder_path, "pos_train_cluster_" + str(id)) + '.txt', "w") as f:
                for index, row in cluster_train_df.iterrows():
                    f.write(f">{row['protein_ID']}\n{row['qseq']}\n")
                # test
            with open(os.path.join(folder_path, "pos_test_cluster_" + str(id)) + '.txt', "w") as f:
                for index, row in cluster_test_df.iterrows():
                    f.write(f">{row['protein_ID']}\n{row['qseq']}\n")

            # write out non-cluster as fasta
            with open(os.path.join(folder_path + "neg_train_cluster_" + str(id)) + '.txt', "w") as f:
                for index, row in notcluster_train_df.iterrows():
                    f.write(f">{row['protein_ID']}\n{row['qseq']}\n")
                # test
            with open(os.path.join(folder_path + "neg_test_cluster_" + str(id)) + '.txt', "w") as f:
                for index, row in notcluster_test_df.iterrows():
                    f.write(f">{row['protein_ID']}\n{row['qseq']}\n")
        print('complete')
        return cluster_ids

class DataLoader:
    def __init__(self,
                 output_path = '../datasets/PPE_input_datasets/',
                 cluster_id : int = -1):
        self.output_path = output_path
        self.cluster_id = cluster_id

    def import_data(self):

        if self.cluster_id < 0:
            raise ValueError(f"cluster id not found")

        id = 'cluster_' + str(self.cluster_id)

        # training file
        pos_train_file = os.path.join(self.output_path, id, 'pos_train_' + id + '.txt')
        neg_train_file = os.path.join(self.output_path, id, 'neg_train_' + id + '.txt')

        # testing file
        #pos_test_file = os.path.join(self.output_path, id, 'pos_test_' + id + '.txt')
        #neg_test_file = os.path.join(self.output_path, id, 'neg_train_' + id + '.txt')

        return pos_train_file, neg_train_file #,pos_test_file, neg_test_file