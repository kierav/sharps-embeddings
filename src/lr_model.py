import pandas as pd
from sklearn.calibration import CalibrationDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler,MaxAbsScaler
from sklearn.metrics import average_precision_score,roc_auc_score
from utils import split_data,print_metrics,plot_performance
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt

class LinearModel():
    """
        Logistic regression model for flare forecasting
    """
    def __init__(self,data_file:str,window:int,val_split:int=0,flare_thresh:float=1e-5,class_weight=None,features=['usflux'],**kwargs):
        self.data_file = data_file
        self.window = window
        self.flare_thresh = flare_thresh
        self.val_split = val_split
        self.scaler = StandardScaler()
        self.features = features
        self.label = 'flare'
        self.model = LogisticRegression(class_weight=class_weight,random_state=val_split,**kwargs)

    def prepare_data(self):
        # load and prep dataframe
        self.df = pd.read_csv(self.data_file)
        self.df['sample_time'] = pd.to_datetime(self.df['sample_time'])
        self.df['flare'] = (self.df['flare_intensity_in_'+str(self.window)+'h']>=self.flare_thresh).astype(int)
        self.df.dropna(axis=0,subset=self.features,inplace=True)

    def setup(self):
        # split data
        self.df_test,self.df_pseudotest,self.df_train,self.df_val = split_data(self.df,self.val_split)
        self.scaler.fit(self.df_train[self.features])
        self.X_train = self.scaler.transform(self.df_train[self.features])
        self.X_val = self.scaler.transform(self.df_val[self.features])
        self.X_pseudotest = self.scaler.transform(self.df_pseudotest[self.features])
        self.X_test = self.scaler.transform(self.df_test[self.features])
        return

    def train(self):
        self.model.fit(self.X_train,self.df_train[self.label])

    def test(self,X,y):
        ypred = self.model.predict_proba(X)
        return ypred[:,1]
        

if __name__ == "__main__":
    data_file = 'data/index_sharps.csv'
    window = 24
    flare_thresh = 1e-5
    print('Window: ',window,'h')

    feats = ['lat_fwt','lon_fwt','area_acr','usflux','meangam','meangbt','meangbz','meangbh',
         'meanjzd','totusjz','meanalp','meanjzh','totusjh','absnjzh','savncpp','meanpot',
         'totpot','meanshr','shrgt45','r_value']    # SHARPs parameters

    df_ensemble = pd.DataFrame()

    for val_split in range(5):
        model = LinearModel(data_file=data_file,window=window,flare_thresh=flare_thresh,
                            val_split=val_split,features=feats,max_iter=200)
        model.prepare_data()
        model.setup()
        model.train()
        ypred = model.test(model.X_pseudotest,model.df_pseudotest['flare'])
        y = model.df_pseudotest['flare']

        if len(df_ensemble) == 0:
            df_ensemble['ytrue'] = y
        df_ensemble['ypred'+str(val_split)] = ypred

        print_metrics(ypred,y,True)
        
    df_ensemble['ypred_median'] = df_ensemble.filter(regex='ypred[0-9]').median(axis=1)
    print_metrics(df_ensemble['ypred_median'],df_ensemble['ytrue'],True)

    plot_performance(df_ensemble,cal='ypred')
    plt.savefig('24h_Mflare_lr_performance.png')
