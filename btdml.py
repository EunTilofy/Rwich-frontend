from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import roc_auc_score, recall_score, precision_score
from sklearn.metrics import average_precision_score, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd

class Binary_Treatment_DML():
    def __init__(self,df_train,df_test,cat_features,con_features,y_model_conf,t_model_conf,res_model_conf,kfold,Y,T):
        self.df_train,self.df_test = df_train,df_test
        self.cat_features,self.con_features = cat_features,con_features
        self.features = cat_features + con_features
        self.y_model_conf,self.t_model_conf,self.res_model_conf = y_model_conf,t_model_conf,res_model_conf
        self.kfold = kfold # kfold cross fit
        self.Y,self.T = Y,T # Y~outcome, T~treatment
        self.res_model = CatBoostRegressor(**res_model_conf) # residual model using catboost(deal with categorical features)
        self.res_model.name = 'res_model' 
        self.y_model = CatBoostClassifier(**self.y_model_conf)
        self.y_model.name = 'y_model'
        self.cali_res_model_p = None
        self.cali_res_model_i = None
        self.cali_y_model_p = None
        self.cali_y_model_i = None
  
        
        
    def first_stage_fit(self,df_train,df_pred):
        # T_train = df_train[df_train[self.T]==1] # treatment group train
        # C_train = df_train[df_train[self.T]==0].sample(n=T_train.shape[0]) # control group train
        # T_pred = df_pred[df_pred[self.T]==1] # treatment group predict
        # C_pred = df_pred[df_pred[self.T]==0].sample(n=T_pred.shape[0]) # control group predict
        # df_train = pd.concat([T_train,C_train],axis=0) # concat train df
        # df_pred = pd.concat([T_pred,C_pred],axis=0) # concat pred df
        self.y_model = CatBoostClassifier(**self.y_model_conf).fit(df_train[self.features], df_train[self.Y])
        t_model = CatBoostClassifier(**self.t_model_conf).fit(df_train[self.features], df_train[self.T])

        y_pred = self.y_model.predict_proba(df_pred[self.features])[:, 1]
        t_pred = t_model.predict_proba(df_pred[self.features])[:, 1]
        y_res = np.array(df_pred[self.Y]) - y_pred
        t_res = np.array(df_pred[self.T]) - t_pred
        sample_weight = t_res**2 # non-params model sample weight
        res_label = y_res / t_res # res model fit on residual ratio
        return sample_weight,res_label,df_pred

    def res_model_prep(self,df_all):
        sample_weight_final = np.array([])
        res_label_final = np.array([])
        df_final = pd.DataFrame()
        kf = KFold(n_splits=self.kfold, shuffle=True,random_state=0)
        for train_index,pred_index in kf.split(df_all):
            df_train, df_pred = df_all.iloc[train_index].reset_index(drop=True), df_all.iloc[pred_index].reset_index(drop=True)
            sample_weight,res_label,df_pred = self.first_stage_fit(df_train,df_pred)
            sample_weight_final = np.concatenate([sample_weight_final,sample_weight],axis=0)
            res_label_final = np.concatenate([res_label_final,res_label],axis=0)
            df_final = pd.concat([df_final, df_pred], ignore_index=True)
        return sample_weight_final,res_label_final,df_final

    def fit(self):
        sample_weight_final,res_label_final,df_final = self.res_model_prep(self.df_train)
        self.res_model.fit(df_final[self.features],res_label_final,sample_weight=sample_weight_final)

    def predict_y(self, data):#预测y的标签 
        return self.y_model.predict(data[self.features])# 默认阈值 0.5
    
    def predict_outcome(self, data,if_train = False,if_cali = False):#预测y的概率值
        y_train_pred = self.y_model.predict_proba(data[self.features])[:, 1]
        if (if_train == True) and (if_cali == True):
            self.cali_y_model_p = self.calibrate_predict_y_train(data,self.y_model,'y_model', type_cali ='Platt')
            self.cali_y_model_i = self.calibrate_predict_y_train(data,self.y_model,'y_model',type_cali ='Isotonic')
            return self.cali_y_model_p.predict_proba(y_train_pred.reshape(-1, 1))[:, 1],self.cali_y_model_i.predict(y_train_pred)
        elif (if_train == False) and (if_cali == True):
            return self.cali_y_model_p.predict_proba(y_train_pred.reshape(-1, 1))[:, 1],self.cali_y_model_i.predict(y_train_pred)
        else:
            return self.y_model.predict_proba(data[self.features])[:, 1]

    
    def predict(self,data,if_train = False,if_cali = False):#预测uplift的概率值
        y_train_pred = self.res_model.predict(data[self.features])
        if (if_train == True) and (if_cali == True):
            self.cali_res_model_p = self.calibrate_predict_y_train(data,self.res_model,'res_model', type_cali ='Platt')
            self.cali_res_model_i = self.calibrate_predict_y_train(data,self.res_model,'res_model', type_cali ='Isotonic')
            return self.cali_res_model_p.predict_proba(y_train_pred.reshape(-1, 1))[:, 1],self.cali_res_model_i.predict(y_train_pred)
        elif (if_train == False) and (if_cali == True):
            return self.cali_res_model_p.predict_proba(y_train_pred.reshape(-1, 1))[:, 1],self.cali_res_model_i.predict(y_train_pred)
        else:
            return self.res_model.predict(data[self.features])

    def calibrate_predict_y_train(self,data_train,model,model_name,type_cali ='Platt'):
        if model_name =='res_model':
            y_pred_raw = model.predict(data_train[self.features])
        elif model_name =='y_model':
            y_pred_raw = model.predict_proba(data_train[self.features])[:, 1]
        if type_cali == 'Platt' :
            calibrated_model_p = LogisticRegression()
            calibrated_model_p.fit(y_pred_raw.reshape(-1, 1), data_train[self.Y])
            return calibrated_model_p
        if type_cali == 'Isotonic':
            calibrated_model_i = IsotonicRegression(out_of_bounds='clip')
            calibrated_model_i.fit(y_pred_raw, data_train[self.Y])
            return calibrated_model_i

    def get_feature_importance(self):
        Feature = pd.DataFrame()
        f_i = self.res_model.get_feature_importance()
        Feature['feature'] = self.features
        Feature['importance'] = f_i
        return Feature.sort_values(by='importance',ascending=False)


