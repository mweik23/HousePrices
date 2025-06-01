#different methods of transforming data
scalar_inputs = ['LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',\
                'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'GarageYrBlt', 'GarageCars', 'GarageArea',\
                'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'YrSold', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',\
                    'KitchenAbvGr', 'Fireplaces']
to_scalar = ['LotShape', 'LandSlope', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'CentralAir', 'Functional',\
            'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'KitchenQual', 'FireplaceQu', 'HeatingQC']
one_hot = ['MSSubClass', 'MSZoning', 'Street', 'LotConfig', 'Neighborhood', 'RoofStyle', 'RoofMatl', 'Foundation',\
            'SaleCondition']
one_hot_plus = {'Alley': {'NA': 0}, 'LandContour': {'Lvl': 0}, 'MasVnrType': {'None': 0, 'NA': 'dist', 'weight': 'MasVnrArea'}, 'GarageType':{'NA': 0},\
                'MiscFeature': {'NA': 0, 'weight': 'MiscVal'}, 'Heating': {'weight': 'HeatingQC'}, 'Exterior1st': {'NA': 0}, 'Exterior2nd': {'NA': 0}, 'SaleType': {'NA': 0}}
case_by_case =  {'Utilities': [{'AllPub':[1, 1, 1, 0], 'NoSewr':[0, 1, 1, 0], 'NoSeWa':[0, 0, 1, 0], 'ELO':[0, 0, 0, 0], 'NA':[0, 0, 0, 1]}, ['Sew', 'Wat', 'Gas', 'Other']],
        'Condition1': [{'Artery':[1, 0, 0, 0], 'Feedr':[0.5, 0, 0, 0], 'Norm': [0, 0, 0, 0], 'RRNn': [0, 0.5, 0, 0],
                        'RRAn':[0, 1, 0, 0], 'PosN':[0, 0, 0, .5], 'PosA': [0, 0, 0, 1], 'RRNe':[0, 0, 0.5, 0], 'RRAe': [0, 0, 1, 0]},
                        ['Road', 'NS_rail', 'EW_rail', 'Positive']],
            'Condition2': 'Condition1',
            'BldgType': [{'1Fam':[0, 0], '2fmCon':[1, 0], 'Duplex': [0.5, 0], 'TwnhsE':[0, 0.5], 'TwnhsI': [0, 1], 'Twnhs':[0, 1]},
                        ['2fam', 'Twnhs']],
            'HouseStyle': [{'1Story': [0, 0, 0], '1.5Fin': [0.4, 0, 0], '1.5Unf': [0.2, 0, 0], '2Story': [0.6, 0, 0], '2.5Fin':[1, 0, 0],
                            '2.5Unf':[0.8, 0, 0], 'SFoyer': [0, 1, 0], 'SLvl': [0, 0, 1]}, ['Stories', 'SFoyer', 'SLvl']],
            'BsmtFinType1': [{'NA': [0], 'Unf': [0], 'LwQ':[0.2], 'Rec':[0.4], 'BLQ':[0.6], 'ALQ':[0.8], 'GLQ':[1]},
                            ['BsmtFiQual1']],
            'BsmtFinType2': [{'NA': [0], 'Unf': [0], 'LwQ':[0.2], 'Rec':[0.4], 'BLQ':[0.6], 'ALQ':[0.8], 'GLQ':[1]},
                            ['BsmtFiQual2']],
            'Electrical': [{'SBrkr':[1, 0, 0], 'FuseA':[2/3, 0, 0], 'FuseF':[1/3, 0, 0], 'FuseP':[0, 0, 0], 'Mix':[0, 1, 0], 'NA':[0, 0, 1]},
                        ['BreakWire_rate', 'Mixed', 'NA']],
            'MSZoning': [{'A':[1, 0, 0, 0, 0, 0], 'C (all)': [0, 1, 0, 0, 0, 0], 'FV': [0, 0, 1, 0, 0, 0], 'I': [0, 0, 0, 1, 0, 0],
                          'RH': [0, 0, 0, 0, 0.25, 0], 'RL': [0, 0, 0, 0, 0.75, 0], 'RP': [0, 0, 0, 0, 1, 0], 'RM': [0, 0, 0, 0, 0.5, 0],
                          'NA': [0, 0, 0, 0, 0, 1]}, ['A', 'C (all)', 'FV', 'I', 'R', 'NA']]
                        }
transformations = {'scalar_inputs': scalar_inputs, 'to_scalar': to_scalar, 'one_hot': one_hot, 'one_hot_plus': one_hot_plus, 'case_by_case': case_by_case}