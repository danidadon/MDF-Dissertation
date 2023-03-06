import numpy as np
import pandas as pd
from numpy import percentile
import matplotlib.pyplot as plt
import matplotlib.font_manager
from pyod.utils.utility import standardizer
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
import multiprocessing as mp
from sklearn.metrics import accuracy_score
import time


# Import all models
from sklearn.model_selection import train_test_split
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.lscp import LSCP
from pyod.models.lmdd import LMDD
from pyod.models.cof import COF
from pyod.models.loci import LOCI
from pyod.models.sod import SOD
from pyod.models.rod import ROD
from pyod.models.ecod import ECOD
from pyod.models.copod import COPOD
from pyod.models.loda import LODA
from pyod.models.mad import MAD
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.deep_svdd import DeepSVDD
from pyod.models.mo_gaal import MO_GAAL
from pyod.models.so_gaal import SO_GAAL
from pyod.models.vae import VAE
from pyod.models.sos import SOS



random_state = np.random.RandomState(42)
outliers_fraction = 0.10
#detector_list = [LOF(n_neighbors=5), LOF(n_neighbors=10), LOF(n_neighbors=15),
#                 LOF(n_neighbors=20), LOF(n_neighbors=25), LOF(n_neighbors=30),
#                 LOF(n_neighbors=35), LOF(n_neighbors=40), LOF(n_neighbors=45),
#                 LOF(n_neighbors=50)]

detector_list = [LOF(n_neighbors=5), LOF(n_neighbors=10), LOF(n_neighbors=15)]


# Define nine outlier detection tools to be compared
classifiers = {
    ## TRADITIONAL ##
    ## Linear Models for Outlier Detection: ##
    'Principal Component Analysis (PCA)': PCA(
        contamination=outliers_fraction, random_state=random_state),
    'Minimum Covariance Determinant (MCD)': MCD(
        contamination=outliers_fraction, random_state=random_state),
    'One-class SVM (OCSVM)': OCSVM(contamination=outliers_fraction),
    ## Proximity-Based Outlier Detection Models: ##
    'Local Outlier Factor (LOF)':
        LOF(n_neighbors=10, contamination=outliers_fraction),
    'K Nearest Neighbors (KNN)': KNN(
        contamination=outliers_fraction),
    'Average KNN': KNN(method='mean',
                       contamination=outliers_fraction),
    'Histogram-base Outlier Detection (HBOS)': HBOS(
        contamination=outliers_fraction),
    #'Cluster-based Local Outlier Factor (CBLOF)':
    #    CBLOF(contamination=outliers_fraction,
    #          check_estimator=False, random_state=random_state),
    ## Probabilistic Models for Outlier Detection: ##
    'Angle-based Outlier Detector (ABOD)':
        ABOD(contamination=outliers_fraction),
    ## Outlier Ensembles and Combination Frameworks: ##
    'Isolation Forest': IForest(contamination=outliers_fraction,
                                random_state=random_state),
    'Feature Bagging':
        FeatureBagging(LOF(n_neighbors=10),
                       contamination=outliers_fraction,
                       random_state=random_state),
    'Locally Selective Combination (LSCP)': LSCP(
        detector_list, contamination=outliers_fraction,
        random_state=random_state),
    
    ## STATE OF THE ART ##
    ## Linear Models for Outlier Detection: ##
    #'Deviation-based Outlier Detection (LMDD)' : LMDD(
    #    contamination=outliers_fraction, random_state=random_state),
    ## Proximity-Based Outlier Detection Models: ##
    'Connectivity-Based Outlier Factor (COF)' : 
        COF(n_neighbors=10, contamination=outliers_fraction),
    'Median kNN': KNN(method='median',
                       contamination=outliers_fraction),
    #'SOD': SOD(contamination=outliers_fraction, n_neighbors=35),
    ## Probabilistic Models for Outlier Detection: ##
    #'ecod': ECOD(contamination=outliers_fraction),
    'COPOD: Copula-Based Outlier Detection': COPOD(contamination=outliers_fraction),
    'SOS':  SOS(contamination=outliers_fraction, perplexity=10),
    ## Outlier Ensembles and Combination Frameworks: ##
    'loda': LODA(contamination=outliers_fraction),
}




list1 = ['PCA', 'MCD', 'OCSVM', 'LOF', 'KNN', 'AvgKNN', 'HBOS', 'ABOD', 'IForest', 'FB', 'LSCP', 'COF', 'MedKNN', 'COPOD', 'SOS', 'LODA']
#'CBLOF',
   





def versus(args):
    
    m = len(args.columns) - 1
    
    X = args.iloc[:,0:m]
    y = args.iloc[:,m]
    X_norm = pd.DataFrame(standardizer(X))
        
    results = pd.DataFrame(list1, columns = ['Method'] )
    results['Recall'] = ''
    results['Precision'] = ''
    results['F1_Macro'] = ''
    results['AUC'] = ''
                  
    for i, (clf_name, clf) in enumerate(classifiers.items()):
        # fit the data and tag outliers
        clf.fit(X_norm)
    
        # get the prediction on the test data
        y_pred = clf.predict(X_norm)  # outlier labels (0 or 1)
    
        results.at[i, 'Recall'] = recall_score(y, y_pred)  
        results.at[i, 'Precision'] = precision_score(y, y_pred)
        results.at[i, 'F1_Macro'] = f1_score(y, y_pred, average='macro') 
        results.at[i, 'AUC'] = roc_auc_score(y, y_pred)
    
    # changes based on metric
    best_method_results = pd.DataFrame(results[results.F1_Macro == results.F1_Macro.max()])


    return best_method_results


