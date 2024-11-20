import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import roc_auc_score, recall_score, precision_score
import pandas as pd
import numpy as np

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

import pickle
import pandas as pd
def load_model():
    with open('DML_model.pkl', 'rb') as f:
        DML = pickle.load(f)
    return DML
DML = load_model()

from sklearn.preprocessing import LabelEncoder

# def load_label_encoder(file_path="label_encoder.pkl"):
#     with open(file_path, "rb") as f:
#         return pickle.load(f)
# labelEncoderDict = load_label_encoder()

# def encode_input_data(input_data, labelEncoderDict):
#     for feature, mapping in labelEncoderDict.items():
#         if feature in input_data:
#             input_data[feature] = mapping.get(input_data[feature], np.nan) 
#     return input_data

import gradio as gr

def predict(*inputs):
    cate_features = ['性别（0:女；1:男）', '一线治疗时病理分级（绝大多数是1-2级和3a级，少数3b级和转化为大B细胞的）', '治疗时ECOG（体力评分，一般>2是预后不良因素）', '治疗时B症状(0:无；1:有)' , '首次开始治疗前分期（一般1-2称为局限期，3-4进展期）', 
               '治疗时是否骨髓受累（骨髓穿刺明确，有是预后不良因素）', '治疗时单个淋巴结是否大于6cm（大于6预后不良因素）',
                '分类1(1-chop;2-fc;3-rchop;4-rfc;5-cvp;6-rcvp;7-放疗；8-R单药；9-放化疗；10-RB;11-GCHOP;12-WW;13-姑息对症)', '评效分组1',
                '启动一线治疗原因（新补充）']
    
#     mapping_dict = {
# #                 '2：任何淋巴结或者结外肿块直径≥7cm': 2, 
#                 '4：脾大': 4, 
# #                 '9：其他（请记录具体因素）':9,
# #                '1：受累淋巴结区≥3个，直径≥3cm':1, '8：患者意愿':8, '5：胸腔积液、腹水':5,
# #                '3：B症状':3, '6：白细胞＜1.0x10^9/L或血小板＜100x10^9/L':6, '7：白血病（恶性细胞＞5x10^9/L）':7
#     }
    
    labelEncoderDict = {'性别（0:女；1:男）': {0: 0, 1: 1}, '一线治疗时病理分级（绝大多数是1-2级和3a级，少数3b级和转化为大B细胞的）': {'1-2级': 0, '1-2级，局灶3a级': 1, '3a级': 2, '3a级，可能有转化': 3, '3b级': 4, '3级': 5, '3级 ': 6, 'TFL,DLBCL': 7, 'tFL,DLBCL': 8, np.nan: 9}, '治疗时ECOG（体力评分，一般>2是预后不良因素）': {0.0: 0, 1.0: 1, 2.0: 2, 3.0: 3, 4.0: 4, np.nan: 5}, '治疗时B症状(0:无；1:有)': {0.0: 0, 1.0: 1, np.nan: 2}, '首次开始治疗前分期（一般1-2称为局限期，3-4进展期）': {0.0: 0, 1.0: 1, 2.0: 2, 3.0: 3, 4.0: 4, np.nan: 5}, '治疗时是否骨髓受累（骨髓穿刺明确，有是预后不良因素）': {0.0: 0, 1.0: 1, np.nan: 2}, '治疗时单个淋巴结是否大于6cm（大于6预后不良因素）': {0.0: 0, 1.0: 1, np.nan: 2}, '分类1(1-chop;2-fc;3-rchop;4-rfc;5-cvp;6-rcvp;7-放疗；8-R单药；9-放化疗；10-RB;11-GCHOP;12-WW;13-姑息对症)': {1: 0, 3: 1, 4: 2, 6: 3, 9: 4, 10: 5, 11: 6}, '评效分组1': {'CR': 0, 'PD': 1, 'PR': 2, 'SD': 3, np.nan: 4}, '启动一线治疗原因（新补充）': {4.0: 0, np.nan: 1}}
    
    processed_inputs = []
    for value in inputs:
        if value == "" or value is None:
            processed_inputs.append(np.nan)
        else:
            processed_inputs.append(value)
    inputs = processed_inputs
    
    
    input_data = {
        'ID': '',
        '性别（0:女；1:男）': 1 if inputs[0] == "男" else 0,
        '最大病灶cm（首次治疗）': float(inputs[1]) if pd.notna(inputs[1]) else np.nan,
        'SUVmax（首次治疗）': float(inputs[2]) if pd.notna(inputs[2]) else np.nan,
        '治疗时ECOG（体力评分，一般>2是预后不良因素）': float(inputs[3]) if pd.notna(inputs[3]) else np.nan,
        '治疗时B症状(0:无；1:有)': 1.0 if inputs[4] == "有" else 0.0,
        '首次开始治疗前分期（一般1-2称为局限期，3-4进展期）': float(inputs[5]) if pd.notna(inputs[5]) else np.nan,
        '启动一线治疗年龄（一般60为cutoff）': float(inputs[6]) if pd.notna(inputs[6]) else np.nan,
        '一线治疗时病理分级（绝大多数是1-2级和3a级，少数3b级和转化为大B细胞的）': inputs[7] if pd.notna(inputs[7]) else np.nan,
        '一线治疗时累及淋巴结区数目（大于等于5预后不良）': float(inputs[8]) if pd.notna(inputs[8]) else np.nan,
        '治疗时是否骨髓受累（骨髓穿刺明确，有是预后不良因素）': 1.0 if inputs[9] == "是" else 0.0,
        '治疗时单个淋巴结是否大于6cm（大于6预后不良因素）': 1.0 if inputs[10] == "是" else 0.0,
        'LDH（首次治疗）>240是预后不良因素': float(inputs[11]) if pd.notna(inputs[11]) else np.nan,
        'β2微球蛋白β2-MG（首次治疗）>3是预后不良因素': inputs[12] if pd.notna(inputs[12]) else np.nan,
        'WBC（首次治疗）': float(inputs[13]) if pd.notna(inputs[13]) else np.nan,
        'HGB（首次治疗）<120是预后不良因素': float(inputs[14]) if pd.notna(inputs[14]) else np.nan,
        'PLT（首次治疗）': float(inputs[15]) if pd.notna(inputs[15]) else np.nan,
        '单核细胞绝对值（首次治疗）': float(inputs[16]) if pd.notna(inputs[16]) else np.nan,
        '淋巴细胞绝对值（首次治疗）': float(inputs[17]) if pd.notna(inputs[17]) else np.nan,
        '淋巴单核细胞比（首次治疗）': float(inputs[18]) if pd.notna(inputs[18]) else np.nan,
        '启动一线治疗原因（新补充）': 4 if inputs[19] == '4：脾大' else np.nan,
        '分类1(1-chop;2-fc;3-rchop;4-rfc;5-cvp;6-rcvp;7-放疗；8-R单药；9-放化疗；10-RB;11-GCHOP;12-WW;13-姑息对症)': int(inputs[20].split(' ')[0]) if pd.notna(inputs[20]) else np.nan,
        '评效分组1': inputs[21] if pd.notna(inputs[21]) else np.nan,
        'Rweichi': 1 if inputs[22] == "有" else 0,
        '一线PFS(确诊时间-进展时间，需要写函数)': float(inputs[23]) if pd.notna(inputs[23]) else np.nan,
        '一线后是否进展': 1 if inputs[24] == "是" else 0,
        'pod24': 1 if inputs[25] == "有" else 0,
        '治疗时FLIPI-1（最终版）': float(inputs[26]) if pd.notna(inputs[26]) else np.nan,
        '治疗FLIPI2（最终版）': float(inputs[27]) if pd.notna(inputs[27]) else np.nan,
        '一线治疗时PRIMA-PI': float(inputs[28]) if pd.notna(inputs[28]) else np.nan,
    }
    
    # print("input data = ", input_data)

    # input_data = encode_input_data(input_data, labelEncoderDict)
    
    input_df = pd.DataFrame([input_data])
    
    # print(input_df.dtypes)
    
    for f in cate_features:
        # print(f, input_df[f], labelEncoderDict[f])
        input_df[f] = input_df[f].apply(
            lambda x: labelEncoderDict[f].get(x, labelEncoderDict[f].get(np.nan, np.nan))
        )
        # print(f, input_df[f])

    
    input_df[cate_features] = input_df[cate_features].astype('str')
    
    # print("input_df = ", input_df, input_df.dtypes)
    
    # print(input_df.iloc[0])
    # print(input_df.dtypes)
    
    # with open('input_df.pkl', 'wb') as f:
    #     pickle.dump(input_df, f)

    uplift = DML.predict(input_df)
    predict_y = DML.predict_outcome(input_df)
    
    # print("result = ", uplift[0], predict_y[0])

    return uplift[0], predict_y[0]


inputs = [
    gr.Dropdown(["", "男", "女"], label="性别", value=""),
    gr.Number(label="最大病灶cm（首次治疗）", value=""),
    gr.Number(label="SUVmax（首次治疗）", value=""),
    gr.Dropdown(["", 0, 1, 2, 3, 4, 5], label="治疗时ECOG（体力评分，一般>2是预后不良因素）", value=""),
    gr.Dropdown(["", "无", "有"], label="治疗时B症状", value=""),
    gr.Dropdown(["", 0, 1, 2, 3, 4, 5], label="首次开始治疗前分期（一般1-2称为局限期，3-4进展期）", value=""),
    gr.Number(label="启动一线治疗年龄", value=""),
    gr.Dropdown(["", "1-2级", "3a级", "tFL,DLBCL", "1-2级，局灶3a级",
                 "3级", "3b级", "3a级，可能有转化",
                 "3级", "TFL,DLBCL"], label="一线治疗时病理分级", value=""),
    gr.Number(label="一线治疗时累及淋巴结区数目（大于等于5预后不良）", value=""),
    gr.Dropdown(["", "否", "是"], label="治疗时是否骨髓受累（骨髓穿刺明确，有是预后不良因素）", value=""),
    gr.Dropdown(["", "否", "是"], label="治疗时单个淋巴结是否大于6cm（大于6预后不良因素）", value=""),
    gr.Number(label="LDH（首次治疗）>240是预后不良因素", value=""),
    gr.Textbox(label="β2微球蛋白β2-MG（首次治疗）>3是预后不良因素", value=""),
    gr.Number(label="WBC（首次治疗）", value=""),
    gr.Number(label="HGB（首次治疗）<120是预后不良因素", value=""),
    gr.Number(label="PLT（首次治疗）", value=""),
    gr.Number(label="单核细胞绝对值（首次治疗）", value=""),
    gr.Number(label="淋巴细胞绝对值（首次治疗）", value=""),
    gr.Number(label="淋巴单核细胞比（首次治疗）", value=""),
    gr.Dropdown([
        "",
        "1 ：受累淋巴结区≥3个，直径≥3cm",
        "2 ：任何淋巴结或者结外肿块直径≥7cm",
        "3 ：B症状",
        "4 ：脾大",
        "5 ：胸腔积液、腹水",
        "6 ：白细胞＜1.0x10^9/L或血小板＜100x10^9/L",
        "7 ：白血病（恶性细胞＞5x10^9/L）"
        "8 ：患者意愿",
        "9 ：其他（请记录具体因素）",
    ], label="启动一线治疗原因（新补充）", value=""),
    gr.Dropdown([
        "",
        '1 : chop',
        '2 : fc', 
        '3 : rchop',
        '4 : rfc',
        '5 : cvp',
        '6 : rcvp',
        '7 : 放疗',
        '8 : R单药',
        '9 : 放化疗', 
        '10 : RB', 
        '11 : GCHOP',
        '12 : WW',
        '13 : 姑息对症'
    ], label="分类1", value=""),
    gr.Dropdown(['', 'CR', 'PR', 'PD', 'SD'], label="评效分组1", value=""),
    gr.Dropdown(["", "无", "有"], label="R维持分类(Rweichi)", value=""),
    gr.Number(label="一线PFS(确诊时间-进展时间)", value=""),
    gr.Dropdown(["", "否", "是"], label="一线后是否进展", value=""),
    gr.Dropdown(["", "无", "有", "非免疫化疗", "随访<24个月"], label="是否POD24", value=""),
    gr.Dropdown(["", 0, 1, 2, 3, 4, 5], label="治疗时FLIPI-1（最终版）", value=""),
    gr.Dropdown(["", 0, 1, 2, 3, 4, 5], label="治疗FLIPI2（最终版）", value=""),
    gr.Dropdown(["", 0, 1, 2], label="一线治疗时PRIMA-PI", value="")
]


outputs = [
    gr.Textbox(label="Uplift"),
    gr.Textbox(label="Predict Y"),
]

gr.Interface(fn=predict, inputs=inputs, outputs=outputs, live=False).launch(share=True, debug=True)

