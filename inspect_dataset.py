import pandas as pd
import pickle
import gzip

# pd.set_option('display.max_columns', None)  # Show all columns
# pd.set_option('display.width', None)        # Let Pandas use full terminal width
# pd.set_option('display.expand_frame_repr', False)  # Do not wrap to new lines
# file_path = "./results/adult/tabdiff/results_detailed_1/Custom_tabdiff_Synthesizer_adult.data.gz"
# # file_path = "./results/Custom:DSF_GAUSSIAN_COPULA_adult.data.gz"
# with gzip.open(file_path, "rb") as f:
#     df = pickle.load(f)
# print(df)
# print(df['capital-gain'].value_counts())
# print(df['age'].value_counts())
# print(set(df['education']))
# print(set(df['native-country']))


from sdgym.datasets import get_available_datasets
from sdv.datasets.demo import download_demo

data, metadata = download_demo(
    modality='single_table',
    dataset_name='child'
)

data1, metadata1 = download_demo(
    modality='single_table',
    dataset_name='adult'
)

ds = get_available_datasets()


print(data1['capital-gain'].value_counts())
print(data1)
print(metadata1)