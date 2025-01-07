import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score


class Binary_Treatment_DML():
    def __init__(self,df_train,df_test,cat_features,con_features,y_model_conf,t_model_conf,res_model_conf,kfold,Y,T):
        self.df_train,self.df_test = df_train,df_test
        self.cat_features,self.con_features = cat_features,con_features
        self.features = cat_features + con_features
        self.y_model_conf,self.t_model_conf,self.res_model_conf = y_model_conf,t_model_conf,res_model_conf
        self.kfold = kfold # kfold cross fit
        self.Y,self.T = Y,T # Y~outcome, T~treatment
        self.res_model = CatBoostRegressor(**res_model_conf) # residual model using catboost(deal with categorical features)
        self.y_model = None
        
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
        y_model_train_auc = roc_auc_score(df_train[self.Y], self.y_model.predict(df_train[self.features]))
        y_model_train_precision = precision_score(df_train[self.Y], self.y_model.predict(df_train[self.features]))
        y_model_train_recall = recall_score(df_train[self.Y], self.y_model.predict(df_train[self.features]))
        y_model_pred_auc = roc_auc_score(df_pred[self.Y], self.y_model.predict(df_pred[self.features]))
        y_model_pred_precision = precision_score(df_pred[self.Y], self.y_model.predict(df_pred[self.features]))
        y_model_pred_recall = recall_score(df_pred[self.Y], self.y_model.predict(df_pred[self.features]))
        print({'Y Model Train AUC': y_model_train_auc, 'Y Model Train Precision': y_model_train_precision, 'Y Model Train Recall': y_model_train_recall})
        print({'Y Model Test AUC': y_model_pred_auc, 'Y Model Test Precision': y_model_pred_precision, 'Y Model Test Recall': y_model_pred_recall})

        t_pred = t_model.predict_proba(df_pred[self.features])[:, 1]
        t_model_train_auc = roc_auc_score(df_train[self.T], t_model.predict(df_train[self.features]))
        t_model_train_precision = precision_score(df_train[self.T], t_model.predict(df_train[self.features]))
        t_model_train_recall = recall_score(df_train[self.T], t_model.predict(df_train[self.features]))
        t_model_pred_auc = roc_auc_score(df_pred[self.T], t_model.predict(df_pred[self.features]))
        t_model_pred_precision = precision_score(df_pred[self.T], t_model.predict(df_pred[self.features]))
        t_model_pred_recall = recall_score(df_pred[self.T], t_model.predict(df_pred[self.features]))
        print({'T Model Train AUC': t_model_train_auc, 'T Model Train Precision': t_model_train_precision, 'T Model Train Recall': t_model_train_recall})
        print({'T Model Test AUC': t_model_pred_auc, 'T Model Test Precision': t_model_pred_precision, 'T Model Test Recall': t_model_pred_recall})
        
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

    def predict_outcome(self, data):
        return self.y_model.predict_proba(data[self.features])[:, 1]
    
    def predict(self,data):
        return self.res_model.predict(data[self.features])
    
    def get_feature_importance(self):
        Feature = pd.DataFrame()
        f_i = self.res_model.get_feature_importance()
        Feature['feature'] = self.features
        Feature['importance'] = f_i
        return Feature.sort_values(by='importance',ascending=False)