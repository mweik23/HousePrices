from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from HousePrices.utils.preprocessing import CustomEncoder, String2Float, OneHotPlus, OrdinalWithMedian, EnsureString


def build_preprocessor(pre_config: dict, as_dataframe: bool = False) -> Pipeline:

    #makes sure that all categorical columns are strings
    cat_ensure_string = ColumnTransformer(
        transformers=[
            ('ensure_string', EnsureString(), pre_config['categorical'])
        ],
        remainder='passthrough',  # Keep all other columns unchanged
        verbose_feature_names_out=False,
        n_jobs=-1  # Use all available cores for parallel processing
    ).set_output(transform='pandas')

    # Define transformations for numerical features
    raw_num_pipe = Pipeline([
        ('to_float', String2Float(na_values=('NA', ''))),
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler())
        ]).set_output(transform="pandas")

    # Define transformations for ordinal categorical features
    ordinal_pipe = Pipeline([
        ('ordinal', OrdinalWithMedian(mappings=pre_config['features_dict'])),
        ('scale', StandardScaler())
    ]).set_output(transform="pandas")

    # Define ColumnTransformer for numerical and ordinal features
    num_ordinal = ColumnTransformer(
        transformers=[
            ('raw_num', raw_num_pipe, pre_config['raw_numeric']),
            ('ordinal', ordinal_pipe, pre_config['ordinal'])
        ],
        remainder='passthrough',  # Keep all other columns unchanged
        verbose_feature_names_out=False,
        n_jobs=-1  # Use all available cores for parallel processing
    ).set_output(transform="pandas")

    # Define modified OneHotEncoder for categorical features
    ohe_plus = OneHotPlus(
        directions=pre_config['onehot_plus'],
        features_dict=pre_config['features_dict']
    )

    # Define custom encoder for case-by-case categorical features
    custom = CustomEncoder(
        mappings=pre_config['case_by_case']
    )

    # get list of features for onehot_plus encoder
    ohe_plus_cats = list(pre_config['onehot_plus'].keys())
    ohe_plus_nums = []
    for item in pre_config['onehot_plus'].values():
        if 'use_amp' in item.keys():
            ohe_plus_nums.append(item['use_amp'])

    # Define ColumnTransformer for categorical features
    cat_encodings = ColumnTransformer(
        transformers=[
            ('onehot_plus', ohe_plus, ohe_plus_cats + ohe_plus_nums),
            ('custom', custom, list(pre_config['case_by_case'].keys()))
        ],
        remainder='passthrough',  # Keep all other columns unchanged
        n_jobs=-1  # Use all available cores for parallel processing
    )

    # Combine the preprocessing steps into a single pipeline
    pre = Pipeline(
        steps=[
            ("categorical_ensure_string", cat_ensure_string),
            ("numerical_ordinal", num_ordinal),
            ("categorical_encodings", cat_encodings),
        ]
    )
    if as_dataframe:
        pre.set_output(transform="pandas")
    return pre