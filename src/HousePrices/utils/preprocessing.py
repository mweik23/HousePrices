import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
#from transforms_2 import scalar_inputs, ordinal, onehot, onehot_ignore, onehot_plus, case_by_case  

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
        self.categories_ = {}
        self.order_ = []

    def set_output(self, *, transform=None):
        self._transform_output = transform
        return self
    
    def fit(self, X, y=None):
        #TODO: handle case where X is a numpy array
        #if isinstance(X, np.ndarray):
        #    X = pd.DataFrame(X, columns=[self.cat_col, self.amp_col])
        for col, direction in self.directions.items():
            self.order_.append(col)
            #drop ignore value if present
            mask_entries = []
            if 'ignore' in direction.keys():
                if direction['ignore'] not in self.options_dict[col]:
                    print(f"Warning: {direction['ignore']} not found in options for {col}. This column was set to be ignored anyway for OneHotEncoder.")
                else:
                    self.options_dict[col].remove(direction['ignore'])
                mask_entries.append(direction['ignore'])
            if 'get_mean' in direction.keys():
                if direction['get_mean'] not in self.options_dict[col]:
                    print(f"Warning: {direction['get_mean']} not found in options for {col}. This column was set to be ignored anyway for OneHotEncoder.")
                else:
                    self.options_dict[col].remove(direction['get_mean'])
                mask_entries.append(direction['get_mean'])
            mask = ~X[col].isin(mask_entries)
            # build the encoder for each category
            try:
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            except TypeError:
                encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            encoder.fit(pd.DataFrame(self.options_dict[col], columns=[col]))
            self.categories_[col] = encoder.get_feature_names_out([col])
            encoded = encoder.transform(X[[col]][mask])
            self.encoder[col] = encoder
            if 'use_amp' in direction.keys():
                amp_col = direction['use_amp']
                if amp_col not in X.columns:
                    raise ValueError(f"Column {amp_col} not found in input data.")
                # multiply the encoded values by the amplitude column
                amp = X[amp_col][mask].values.reshape(-1, 1)
                encoded *= amp
            
            #take the mean of the encoded values for the unknown category
            if 'get_mean' in direction.keys():
                self.unknown_means_[col] = encoded.mean(axis=0) if len(encoded) > 0 else np.zeros(encoded.shape[1])
        self.feature_names_in_ = getattr(X, "columns", None)
        return self

    def transform(self, X):
        output = []
        #TODO: handle case where X is a numpy array
        for col in self.order_:
            if col not in X.columns:
                raise ValueError(f"Column {col} not found in input data.")
            # transform the category column
            encoded = self.encoder[col].transform(X[[col]])

            # if directions specify an amplitude column, multiply the one-hot encoding by the amplitude column
            if 'use_amp' in self.directions[col].keys():
                amp_col = self.directions[col]['use_amp']
                if amp_col not in X.columns:
                    raise ValueError(f"Column {amp_col} not found in input data.")
                amp = X[amp_col].values.reshape(-1, 1)
                encoded *= amp

            # if the directions specify a get_mean entry, fill it with the mean
            if 'get_mean' in self.directions[col].keys():
                mean_mask = X[col] == self.directions[col]['get_mean']
                encoded[mean_mask] = self.unknown_means_[col]

            output.append(encoded)
            
        return np.concatenate(output, axis=1)
    
    def get_feature_names_out(self, input_features=None):
        names = []
        for col in self.order_:
            names.extend([f"{self.directions[col]['use_amp']}__{cat}" if 'use_amp' in self.directions[col].keys()
                          else f"{cat}" for cat in self.categories_[col]])
        return np.asarray(names, dtype=object)

class OrdinalWithMedian(BaseEstimator, TransformerMixin):
    def __init__(self, mappings: dict):
        self.mappings = mappings
        self.medians_ = {}
        self.encoders = {}

    def set_output(self, *, transform=None):
        self._transform_output = transform
        return self

    def fit(self, X, y=None):
        for col in X.columns:
            if col in self.mappings.keys():
                self.encoders[col] = OrdinalEncoder(categories=[self.mappings[col]], handle_unknown='use_encoded_value', unknown_value=-1)
                encoded = self.encoders[col].fit_transform(X[[col]])
                self.medians_[col] = np.median(encoded[encoded != -1], axis=0)
            else:
                raise ValueError(f"No mapping found for column {col}. Please provide a mapping.")
        self.feature_names_in_ = getattr(X, "columns", None)
        return self

    def transform(self, X):
        for col in X.columns:
            if col in self.mappings.keys():
                # Transform the column using the encoder
                encoded = self.encoders[col].transform(X[[col]])
                # Replace -1 (unknown) with the median value
                encoded[encoded == -1] = self.medians_[col]
                X[col] = encoded
            else:
                raise ValueError(f"No mapping found for column {col}. Please provide a mapping.")
        return X
    
    def get_feature_names_out(self, input_features=None):
        feats = input_features if input_features is not None else self.feature_names_in_
        return np.asarray(feats, dtype=object)

class CustomEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, mappings={}):
        self.mappings = mappings
        self.order_ = []

    def set_output(self, *, transform=None):
        self._transform_output = transform
        return self

    def fit(self, X, y=None):
        self.order_ = list(self.mappings.keys())
        return self

    def transform(self, X):
        output = []
        for col in self.order_:
            if col in X.columns and col:
                output.append(np.array(X[col].map(self.mappings[col]['transformations']).to_list()))
            else:
                print(f"Warning: Column {col} not found in input data or is empty. Skipping.")
        return np.concatenate(output, axis=1)
    
    def get_feature_names_out(self, input_features=None):
        names = []
        for col in self.order_:
            names.extend([f"{col}_{cat}" for cat in self.mappings[col]['new_cols']])
        return np.asarray(names, dtype=object)
    
class String2Float(BaseEstimator, TransformerMixin):
    def __init__(self, na_values=['NA', 'None']):
        self.na_values = na_values

    def set_output(self, *, transform=None):
        self._transform_output = transform
        return self
    
    def fit(self, X, y=None):
        if self.na_values is None:
            na_vals = ['NA', 'None']
        else:
            na_vals = list(self.na_values)  # copy to avoid aliasing
        self.na_values_ = set(na_vals)
        self.feature_names_in_ = getattr(X, "columns", None)
        return self
    def transform(self, X):
        X = pd.DataFrame(X, columns=self.feature_names_in_) if not isinstance(X, pd.DataFrame) else X.copy()
        for col in X.columns:
            X_col = X[col].where(~X[col].isin(self.na_values), other=np.nan)
            X[col] = pd.to_numeric(X_col, errors='coerce')
        return X
    def get_feature_names_out(self, input_features=None):
        feats = input_features if input_features is not None else self.feature_names_in_
        return np.asarray(feats, dtype=object)

class EnsureString(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def set_output(self, *, transform=None):
        self._transform_output = transform
        return self
    def fit(self, X, y=None):
        self.feature_names_in_ = getattr(X, "columns", None)
        return self
    def transform(self, X):
        X = pd.DataFrame(X, columns=self.feature_names_in_) if not isinstance(X, pd.DataFrame) else X.copy()
        for col in X.columns:
            if not pd.api.types.is_string_dtype(X[col]):
                X[col] = X[col].astype(str)
        return X
    def get_feature_names_out(self, input_features=None):
        feats = input_features if input_features is not None else self.feature_names_in_
        return np.asarray(feats, dtype=object)  

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
