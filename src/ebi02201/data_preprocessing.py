import pandas as pd
from plotting import plot_ner_counts
from utils import process_articles
from tqdm import tqdm

annotations_df = pd.read_csv('../../data/raw/annotation_df.csv')

plot_ner_counts(dataframe=annotations_df, ner_column='ner', 
                title="NER Category Distribution", xlabel="Categories", ylabel="Frequency", 
                palette="coolwarm")

def balance_ner_samples(df):
    min_count = df['ner'].value_counts().min()
    balanced_df = df.groupby('ner').apply(lambda x: x.sample(min_count)).reset_index(drop=True)
    return balanced_df

balanced_annotations_df = balance_ner_samples(annotations_df)

plot_ner_counts(dataframe=balanced_annotations_df, ner_column='ner', 
                title="NER Category Distribution after Trimming", xlabel="Categories", ylabel="Frequency", 
                palette="coolwarm")

unique_pmcids = balanced_annotations_df['pmc_id'].unique()
print(str(len(unique_pmcids)))

# Process in batches
batch_size = 4  # Adjust the batch size
final_data = []

for i in tqdm(range(0, len(unique_pmcids), batch_size), desc="Processing Batches"):
    batch_pmcids = unique_pmcids[i:i + batch_size]
    final_data.extend(process_articles(batch_pmcids, annotations_df))

final_df = pd.DataFrame(final_data, columns=['pmc_id', 'sentence', 'ner'])
final_df.to_csv('../../data/raw/biggest_test.csv', index=False)