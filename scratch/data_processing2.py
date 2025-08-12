import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler, OrdinalEncoder, SimpleImputer
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from transforms_2 import scalar_inputs, ordinal, onehot, onehot_ignore, onehot_plus, case_by_case  

class OneHotWithUnknownMean(BaseEstimator, TransformerMixin):
    def __init__(self, unknown_values, default_unknown_value=None):
        self.unknown_values = unknown_values
        self.default_unknown_value = default_unknown_value
        self.encoders = {}
        self.unknown_mean_ = {}

    def fit(self, X, y=None):
        for col in X.columns:
            unknown_value = self.unknown_values.get(col, self.default_unknown_value)
            mask = X[col] != unknown_value
            encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            encoder.fit(X[mask, [col]])
            encoded = encoder.transform(X[mask, [col]])
            self.encoders[col] = encoder
            self.unknown_mean_[col] = encoded.mean(axis=0) if len(encoded) > 0 else np.zeros(encoded.shape[1])
        return self

    def transform(self, X):
        output = []
        for col, encoder in self.encoders.items():
            unknown_value = self.unknown_values.get(col, self.default_unknown_value)
            if col not in X.columns:
                raise ValueError(f"Column {col} not found in input data.")
            mask = X[col] != unknown_value
            encoded = encoder.transform(X[[col]])
            encoded[~mask.values] = self.unknown_mean_
            output.append(encoded)
        return np.concatenate(output, axis=1)

class OneHotAmplitude(BaseEstimator, TransformerMixin):
    def __init__(self, col_pairs):
        self.col_pairs = col_pairs
        self.encoder = {}
        self.scalers = {}
        self.output_dims_ = []

    def fit(self, X, y=None):
        #TODO: handle case where X is a numpy array
        #if isinstance(X, np.ndarray):
        #    X = pd.DataFrame(X, columns=[self.cat_col, self.amp_col])
        for cat_col, amp_col in self.col_pairs:
            # build the encoder for each category-amp pair
            encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            encoder.fit(X[[cat_col]])
            self.encoder[(cat_col, amp_col)] = encoder

            #build scaler for amplitude
            scaler = StandardScaler()
            scaler.fit(X[[amp_col]])
            self.scalers[(cat_col, amp_col)] = scaler

        return self

    def transform(self, X):
        output = []
        #TODO: handle case where X is a numpy array
        #if isinstance(X, np.ndarray):
        #    X = pd.DataFrame(X, columns=[self.cat_col, self.amp_col])
        for (cat_col, amp_col), encoder in self.encoder.items():
            # store the scaler for the amplitude column
            scaler = self.scalers[(cat_col, amp_col)]
            # scale the amplitude column
            amp = scaler.transform(X[[amp_col]])
            # transform the category column
            onehot = encoder.transform(X[[cat_col]])
            # multiply one-hot encoding by the scaled amplitude
            output.append(onehot * X[[amp_col]].values)
            
        return np.concatenate(output, axis=1)

class OneHotPlus(BaseEstimator, TransformerMixin):
    def __init__(self, directions: dict, options_dict: dict):
        self.directions = directions
        self.options_dict = options_dict
        self.unknown_means_ = {}
        self.encoder = {}

    def fit(self, X, y=None):
        #TODO: handle case where X is a numpy array
        #if isinstance(X, np.ndarray):
        #    X = pd.DataFrame(X, columns=[self.cat_col, self.amp_col])
        for col, directions in self.directions.items():
            #drop ignore value if present
            mask_entries = []
            if 'ignore' in directions.keys():
                self.options_dict[col] = self.options_dict[col].remove(directions['ignore'])
                mask_entries.append(directions['ignore'])
            if 'get_mean' in directions.keys():
                self.options_dict[col] = self.options_dict[col].remove(directions['get_mean'])
                mask_entries.append(directions['get_mean'])
            mask = ~X[col].isin(mask_entries)
            # build the encoder for each category
            encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            encoder.fit(pd.DataFrame(self.options_dict[col], columns=[col]))
            encoded = encoder.transform(X[mask, [col]])
            self.encoder[col] = encoder
            if 'use_amp' in directions.keys():
                amp_col = directions['use_amp']
                if amp_col not in X.columns:
                    raise ValueError(f"Column {amp_col} not found in input data.")
                # multiply the encoded values by the amplitude column
                amp = X[amp_col][mask].values.reshape(-1, 1)
                encoded *= amp
            
            #take the mean of the encoded values for the unknown category
            if 'get_mean' in directions.keys():
                self.unknown_means_[col] = encoded.mean(axis=0) if len(encoded) > 0 else np.zeros(encoded.shape[1])
        return self

    def transform(self, X):
        output = []
        #TODO: handle case where X is a numpy array
        for col, encoder in self.encoder.items():
            if col not in X.columns:
                raise ValueError(f"Column {col} not found in input data.")
            # transform the category column
            encoded = encoder.transform(X[[col]])

            # if directions specify an amplitude column, multiply the one-hot encoding by the amplitude column
            if 'use_amp' in self.directions[col].keys():
                amp_col = self.directions[col]['use_amp']
                if amp_col not in X.columns:
                    raise ValueError(f"Column {amp_col} not found in input data.")
                amp = X[amp_col].values.reshape(-1, 1)
                encoded *= amp

            mean_mask = X[col] == self.directions[col]['get_mean']
            # if the directions specify a get_mean entry, fill it with the mean
            if 'get_mean' in self.directions[col].keys():
                encoded[mean_mask] = self.unknown_means_[col]

            output.append(encoded)
            
        return np.concatenate(output, axis=1)

class DummyEmbedding(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=4):
        self.n_components = n_components
        self.map = {}

    def fit(self, X, y=None):
        unique_cats = pd.Series(X.squeeze()).unique()
        rng = np.random.default_rng(42)
        for cat in unique_cats:
            self.map[cat] = rng.normal(size=self.n_components)
        return self

    def transform(self, X):
        return np.vstack([self.map[val] for val in pd.Series(X.squeeze())])
    
# Pipeline for one-hot with amplitude (first scale the amplitude column)
class String2Float(BaseEstimator, TransformerMixin):
    def __init__(self, na_values=['NA', 'None']):
        self.na_values = set(na_values)
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        for col in X.columns:
            X_col = X[col].where(~X[col].isin(self.na_values), other=np.nan)
            X[col] = pd.to_numeric(X_col, errors='coerce')
        return X


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.columns]


'''
# Example feature lists (edit for your schema)
raw_nums = ['age', 'income']
ord_cats = ['risk_level']                  # ordinal
onehot_cats = ['gender']                   # one-hot
onehot_drop_cats = ['marital_status']      # one-hot with drop
onehot_unknown_mean = ['product']          # special one-hot
region_cat = 'region'
region_amp = 'region_amp'
onehot_amp = [(region_cat, region_amp)]      # one-hot with amplitude
custom_embed_cats = ['device_type']
'''
if __name__ == "__main__":

    convert_clean_pipeline = Pipeline([
        ('to_float', String2Float(na_values=('NA', ''))),
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler())
    ]).set_output(transform="pandas")

    region_amp_pipeline = Pipeline([
        ('select', ColumnSelector(onehot_amp)),
        ('scale_amp', ColumnTransformer([
            ('region', 'passthrough', [region_cat]),
            ('amp_scaled', StandardScaler(), [region_amp]),
        ], remainder='drop')),
        ('to_df', FunctionTransformer(lambda X: pd.DataFrame(X, columns=[region_cat, region_amp]), validate=False)),
        ('onehot_amp', OneHotAmplitude(amp_col=region_amp)),
    ])

    # Full ColumnTransformer for preprocessing
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), raw_nums),
        ('ord', OrdinalEncoder(), ord_cats),
        ('onehot', OneHotEncoder(sparse=False), onehot_cats),
        ('onehot_drop', OneHotEncoder(sparse=False, drop='first'), onehot_drop_cats),
        ('onehot_unknown', OneHotWithUnknownMean(), onehot_unknown_mean),
        ('onehot_amp', region_amp_pipeline, onehot_amp),
        ('embed', DummyEmbedding(n_components=4), custom_embed_cats),
    ], remainder='drop')
