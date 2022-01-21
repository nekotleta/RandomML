from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, chi2, RFE, SelectFromModel
from sklearn.preprocessing import MinMaxScaler
import pandas as pds
import numpy as np

class FeatureSelector():
    """
    Class for performing feature selection for data preprocessing.
    
    Implements four different methods to identify features importances
    
        1. Correlation Pearson
        2. Chi^2 selector
        3. RFE selector with default estimator LogReg
        4. Embeded selector with default estimator LogReg
        
    Parameters
    --------
        data : dataframe
            A dataset with observations in the rows and features in the columns
        labels : array or series
            Array of labels for training the machine learning model
        k : int
            Number of features to select, default = 100
         
    Notes
    --------
    
        - All operations can be run with the `run_all` method.
    
    """
    def __init__(self, data, label, k=100):
        self.X = data
        self.y = label
        self.k = k
        self.X_norm = MinMaxScaler().fit_transform(self.X)
        self.feature_name = data.columns
        
    def cor_selector(self):
        """Pearson Correlation
        Return
        --------
        cor_support: list
            Boolean array with length equals number of X columns. True is for selected feature
        cor_feature: list
            Array of selected features
        """
        cor_list = []
        # calculate the correlation with y for each feature
        for i in self.X.columns.tolist():
            cor = np.corrcoef(self.X[i], self.y)[0, 1]
            cor_list.append(cor)
        # replace NaN with 0
        cor_list = [0 if np.isnan(i) else i for i in cor_list]
        cor_feature = self.X.iloc[:,np.argsort(np.abs(cor_list))[-self.k:]].columns.tolist()
        cor_support = [True if i in cor_feature else False for i in self.feature_name]
        return cor_support, cor_feature
    
    def chi_selector(self):
        """Chi^2 selected K best
        Return
        --------
        chi_support: list
            Boolean array with length equals number of X columns. True is for selected feature
        chi_feature: list
            Array of selected features
        """
        chi_selector = SelectKBest(chi2, k=self.k)
        chi_selector.fit(self.X_norm, self.y)
        chi_support = chi_selector.get_support()
        chi_feature = self.X.loc[:,chi_support].columns.tolist()
        return chi_support, chi_feature
    
    def rfe_selector(self, estimator=LogisticRegression(), params=None):
        """RFE selected with default estimator LogisticRegression
        Return
        --------
        rfe_support: list
            Boolean array with length equals number of X columns. True is for selected feature
        rfe_feature: list
            Array of selected features
        """
        if params is None:
            params={
                "n_features_to_select": self.k, 
                "step": 10
            }
        rfe_selector = RFE(estimator=estimator, **params)
        rfe_selector.fit(self.X_norm, self.y)
        rfe_support = rfe_selector.get_support()
        rfe_feature = self.X.loc[:,rfe_support].columns.tolist()
        return rfe_support, rfe_feature
    
    def embeded_selector(self, estimator=LogisticRegression(penalty="l1", solver='liblinear')):
        """Embeded selection with default estimator LogisticRegression. Estimator could be RandomForestClassifier, LGBMClassifier
        Return
        --------
        embeded_support: list
            Boolean array with length equals number of X columns. True is for selected feature
        embeded_feature: list
            Array of selected features
        """
        embeded_selector = SelectFromModel(estimator, threshold='1.25*median')
        embeded_selector.fit(self.X_norm, self.y)
        embeded_support = embeded_selector.get_support()
        embeded_feature = self.X.loc[:,embeded_lr_support].columns.tolist()
        return embeded_support, embeded_feature
    
    def run_all(self):
        """Embeded selection with default estimator LogisticRegression. Estimator could be RandomForestClassifier, LGBMClassifier
        Return
        --------
        feature_selection_df: pandas.DataFrame
            DataFrame after running all feature selectors sort by most selected feature
        """
        from sklearn.ensemble import RandomForestClassifier
        from lightgbm import LGBMClassifier
        # put all selection together
        print("Correlation selector starts.....")
        cor_support, _ = self.cor_selector()
        print("Chi2 selector starts.....")
        chi_support, _ = self.chi_selector()
        print("RFE selector starts.....")
        rfe_support, _ = self.rfe_selector()
        print("Embeded LogReg selector starts.....")
        embeded_lr_support, _ = self.embeded_selector()
        print("Embeded RF selector starts.....")
        embeded_rf_support, _ = self.embeded_selector(RandomForestClassifier(n_estimators=self.k))
        lgbc = LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=32, colsample_bytree=0.2,
            reg_alpha=3, reg_lambda=1, min_split_gain=0.01, min_child_weight=40)
        print("Embeded LightGBM selector starts.....")
        embeded_lgb_support, _ = self.embeded_selector(lgbc)
        
        feature_selection_df = pds.DataFrame({'Feature': self.feature_name, 
                                             'Pearson': cor_support, 
                                             'Chi-2': chi_support, 
                                             'RFE': rfe_support, 
                                             'Logistics': embeded_lr_support,
                                             'Random Forest': embeded_rf_support,
                                             'LightGBM': embeded_lgb_support})
        
        feature_selection_df['Total'] = np.sum(feature_selection_df, axis=1)
        feature_selection_df = feature_selection_df.sort_values(['Total', 'Feature'], ascending=False)
        feature_selection_df.index = range(1, len(feature_selection_df)+1)
        return feature_selection_df
