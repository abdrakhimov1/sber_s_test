import pandas as pd
from sklearn.model_selection import train_test_split

from models.CustomGridSearch import CustomGridSearch

"""
Reading data
"""
interactions = pd.read_csv('data/interactions.csv')
item_subclass = pd.read_csv('data/item_subclass.csv')
item_price = pd.read_csv('data/item_price.csv')
item_asset = pd.read_csv('data/item_asset.csv')
user_region = pd.read_csv('data/user_region.csv')
user_age = pd.read_csv('data/user_age.csv')

"""
Splitting data to train and test. Adding users, who wasn't included in train or test with zero vectors
"""

train, test = train_test_split(interactions, test_size=0.2, random_state=42, shuffle=True)
mismatch_set = set(item_subclass['row']).difference(set(train['col'].values))
new_dataframe = pd.DataFrame(data={
    'row': [0 for _ in mismatch_set],
    'col': list(mismatch_set),
    'data': [0 for _ in mismatch_set]})
mismatch_set_test = set(item_subclass['row']).difference(set(test['col'].values))
new_dataframe_test = pd.DataFrame(data={
    'row': [0 for _ in mismatch_set_test],
    'col': list(mismatch_set_test),
    'data': [0 for _ in mismatch_set_test]})
train = pd.concat([train, new_dataframe])
test = pd.concat([test, new_dataframe_test])

"""
Params for grid search
"""
metrics = ['cityblock', 'cosine', 'euclidean']
n_neighbours_list = [3]


"""Grid search """
grid = CustomGridSearch(metrics,
                        n_neighbours_list,
                        train,
                        test,
                        item_asset,
                        item_price,
                        item_subclass,
                        user_region,
                        user_age)

grid.grid_search()
print(grid.best())
