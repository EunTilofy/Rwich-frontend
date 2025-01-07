from btdml import Binary_Treatment_DML
from flask import Flask, render_template, request
from sklearn import svm
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import matplotlib

app = Flask(__name__)

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# Load the model
with open('DML_model.pkl', 'rb') as f:
    DML = pickle.load(f)
    
with open('svm_model.pkl', 'rb') as f:
    SVM = pickle.load(f)
    
with open('predict_y.pkl', 'rb') as f:
    predict_y_samples = pickle.load(f)
with open('uplift.pkl', 'rb') as f:
    uplift_samples = pickle.load(f)

labelEncoderDict = {
    '性别（0:女；1:男）': {0: 0, 1: 1},
    '一线治疗时病理分级（绝大多数是1-2级和3a级，少数3b级和转化为大B细胞的）': {'1-2级': 0, '1-2级，局灶3a级': 1, '3a级': 2, '3a级，可能有转化': 3, '3b级': 4, '3级': 5, '3级 ': 6, 'TFL,DLBCL': 7, 'tFL,DLBCL': 8, np.nan: 9},
    '治疗时ECOG（体力评分，一般>2是预后不良因素）': {0.0: 0, 1.0: 1, 2.0: 2, 3.0: 3, 4.0: 4, np.nan: 5},
    '治疗时B症状(0:无；1:有)': {0.0: 0, 1.0: 1, np.nan: 2},
    '首次开始治疗前分期（一般1-2称为局限期，3-4进展期）': {0.0: 0, 1.0: 1, 2.0: 2, 3.0: 3, 4.0: 4, np.nan: 5},
    '治疗时是否骨髓受累（骨髓穿刺明确，有是预后不良因素）': {0.0: 0, 1.0: 1, np.nan: 2},
    '治疗时单个淋巴结是否大于6cm（大于6预后不良因素）': {0.0: 0, 1.0: 1, np.nan: 2},
    '分类1(1-chop;2-fc;3-rchop;4-rfc;5-cvp;6-rcvp;7-放疗；8-R单药；9-放化疗；10-RB;11-GCHOP;12-WW;13-姑息对症)': {1: 0, 3: 1, 4: 2, 6: 3, 9: 4, 10: 5, 11: 6},
    '评效分组1': {'CR': 0, 'PD': 1, 'PR': 2, 'SD': 3, np.nan: 4},
    '启动一线治疗原因（新补充）': {4.0: 0, np.nan: 1}
}

cate_features = ['性别（0:女；1:男）', '一线治疗时病理分级（绝大多数是1-2级和3a级，少数3b级和转化为大B细胞的）', '治疗时ECOG（体力评分，一般>2是预后不良因素）', '治疗时B症状(0:无；1:有)' , '首次开始治疗前分期（一般1-2称为局限期，3-4进展期）', 
               '治疗时是否骨髓受累（骨髓穿刺明确，有是预后不良因素）', '治疗时单个淋巴结是否大于6cm（大于6预后不良因素）',
                '分类1(1-chop;2-fc;3-rchop;4-rfc;5-cvp;6-rcvp;7-放疗；8-R单药；9-放化疗；10-RB;11-GCHOP;12-WW;13-姑息对症)', '评效分组1',
                '启动一线治疗原因（新补充）']

def predict(inputs):
    processed_inputs = []
    for value in inputs:
        if value == "" or value is None:
            processed_inputs.append(np.nan)
        else:
            processed_inputs.append(value)
    inputs = processed_inputs
    
    input_data = {
        'ID': inputs[29],
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
    
    input_df = pd.DataFrame([input_data])
    
    for f in cate_features:
        input_df[f] = input_df[f].apply(
            lambda x: labelEncoderDict[f].get(x, labelEncoderDict[f].get(np.nan, np.nan))
        )
    
    input_df[cate_features] = input_df[cate_features].astype('str')
        
    uplift = DML.predict(input_df)[0]
    predict_y = DML.predict_outcome(input_df)[0]
    
    tmp = pd.DataFrame([[uplift, predict_y]], columns=['uplift', 'predict_y'])
    tmp_class = SVM.predict(tmp)[0]
    conc =  "不推荐" if tmp_class == 0 else ("非常推荐" if tmp_class == 1 else "推荐")
    
    fig, ax = plt.subplots(figsize = (24, 12))
    ax.hist(predict_y_samples, bins = 20, color = 'blue', alpha = 0.7)
    ax.set_title('24个月进展风险分布') 
    ax.grid(True)
    ax.axvline(x=predict_y, color='red', linestyle = 'dashed', linewidth=2)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    url1 = base64.b64encode(buf.getvalue()).decode('utf8')
    buf.close()
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize = (24, 12))
    ax.hist(uplift_samples, bins = 20, color = 'blue', alpha = 0.7)
    ax.set_title('R维持获益等级分布') 
    ax.grid(True)
    ax.axvline(x=uplift, color='red', linestyle = 'dashed', linewidth=2)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    url2 = base64.b64encode(buf.getvalue()).decode('utf8')
    buf.close()
    plt.close(fig)
    
    percentile_y = 100 * ((predict_y_samples <= predict_y).mean())
    predict_y = f"24 个月内的进展风险高于 {percentile_y:.2f}% 的人群。"
    
    # print(max(uplift_samples), uplift)
    
    percentile_up = 100 * ((uplift_samples <= uplift).mean())
    uplift = f"预计采用 R 维持的获益等级高于 {percentile_up:.2f}% 的人群。"
    
    return uplift, predict_y, conc, url1, url2

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', result=None)

@app.route('/predict', methods=['POST'])
def predict_route():
    gender = request.form['gender']
    max_tumor_size = request.form['max_tumor_size']
    suv_max = request.form['suv_max']
    ecog_score = request.form['ecog_score']
    b_symptoms = request.form['b_symptoms']
    initial_stage = request.form['initial_stage']
    treatment_age = request.form['treatment_age']
    pathology_grade = request.form['pathology_grade']
    lymph_node_count = request.form['lymph_node_count']
    bone_marrow_involvement = request.form['bone_marrow_involvement']
    lymph_node_size = request.form['lymph_node_size']
    ldh = request.form['ldh']
    beta2_mg = request.form['beta2_mg']
    wbc = request.form['wbc']
    hgb = request.form['hgb']
    plt = request.form['plt']
    monocyte = request.form['monocyte']
    lymphocyte = request.form['lymphocyte']
    lymph_monocyte_ratio = request.form['lymph_monocyte_ratio']
    reason_for_treatment = request.form['reason_for_treatment']
    classification = request.form['classification']
    efficacy_group = request.form['efficacy_group']
    rweichi = request.form['rweichi']
    fps = request.form['fps']
    progress_after = request.form['progress_after']
    pod24 = request.form['pod24']
    FLIPI_1 = request.form['FLIPI_1']
    FLIPI_2= request.form['FLIPI_2']
    FLIPI_PI = request.form['FLIPI_PI']
    patient_id = request.form['patient_id']

    inputs = [
        gender, max_tumor_size, suv_max, ecog_score, b_symptoms, initial_stage, treatment_age, pathology_grade,
        lymph_node_count, bone_marrow_involvement, lymph_node_size, ldh, beta2_mg, wbc, hgb, plt, monocyte,
        lymphocyte, lymph_monocyte_ratio, reason_for_treatment, classification, efficacy_group, rweichi,
        fps, progress_after, pod24, FLIPI_1, FLIPI_2, FLIPI_PI, patient_id
    ]
    
    result = predict(inputs)
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)