from btdml import Binary_Treatment_DML
from flask import Flask, render_template, request, jsonify
from sklearn import svm
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import io
import base64
import matplotlib

app = Flask(__name__)

matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
matplotlib.use('Agg')

# Load the model
with open('DML_model_6.pkl', 'rb') as f:
    DML_6 = pickle.load(f)

with open('DML_model_12.pkl', 'rb') as f:
    DML_12 = pickle.load(f)
    
with open('DML_model_24.pkl', 'rb') as f:
    DML_24 = pickle.load(f)
    
with open('DML_model_36.pkl', 'rb') as f:
    DML_36 = pickle.load(f)
    
with open('predict_y_6.pkl', 'rb') as f:
    predict_y_6 = pickle.load(f)

with open('predict_y_12.pkl', 'rb') as f:
    predict_y_12 = pickle.load(f)
    
with open('predict_y_24.pkl', 'rb') as f:
    predict_y_24 = pickle.load(f)
    
with open('predict_y_36.pkl', 'rb') as f:
    predict_y_36 = pickle.load(f)
    
with open('svm_model.pkl', 'rb') as f:
    SVM = pickle.load(f)

with open('uplift.pkl', 'rb') as f:
    uplift_samples = pickle.load(f)
    
ori_df = pd.read_pickle('df_filtered.pkl')

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

cate_features = ['性别（0:女；1:男）', '一线治疗时病理分级（绝大多数是1-2级和3a级，少数3b级和转化为大B细胞的）', 
                 '治疗时ECOG（体力评分，一般>2是预后不良因素）', '治疗时B症状(0:无；1:有)' , '首次开始治疗前分期（一般1-2称为局限期，3-4进展期）', 
               '治疗时是否骨髓受累（骨髓穿刺明确，有是预后不良因素）', '治疗时单个淋巴结是否大于6cm（大于6预后不良因素）',
                '分类1(1-chop;2-fc;3-rchop;4-rfc;5-cvp;6-rcvp;7-放疗；8-R单药；9-放化疗；10-RB;11-GCHOP;12-WW;13-姑息对症)', '评效分组1',
                '启动一线治疗原因（新补充）']

# features = ['最大病灶cm（首次治疗）', 'SUVmax（首次治疗）', '启动一线治疗年龄（一般60为cutoff）', 
#                  '一线治疗时累及淋巴结区数目（大于等于5预后不良）', 'LDH（首次治疗）>240是预后不良因素', 
#                  'β2微球蛋白β2-MG（首次治疗）>3是预后不良因素', 'WBC（首次治疗）', 'HGB（首次治疗）<120是预后不良因素',
#                    'PLT（首次治疗）', '单核细胞绝对值（首次治疗）', '淋巴细胞绝对值（首次治疗）', '淋巴单核细胞比（首次治疗）']

need_features = [
    ['启动一线治疗年龄（一般60为cutoff）', 1],
    # ['治疗时ECOG（体力评分，一般>2是预后不良因素）', 0],
    # ['治疗时B症状(0:无；1:有)', 0],
    # ['一线治疗时病理分级（绝大多数是1-2级和3a级，少数3b级和转化为大B细胞的）', 0],
    # ['首次开始治疗前分期（一般1-2称为局限期，3-4进展期）', 0],
    ['一线治疗时累及淋巴结区数目（大于等于5预后不良）', 1],
    # ['治疗时是否骨髓受累（骨髓穿刺明确，有是预后不良因素）', 0],
    ['最大病灶cm（首次治疗）', 1],
    ['SUVmax（首次治疗）', 1],
    ['WBC（首次治疗）', 1],
    ['单核细胞绝对值（首次治疗）', 1],
    ['淋巴细胞绝对值（首次治疗）', 1],
    ['HGB（首次治疗）<120是预后不良因素', 1],
    ['PLT（首次治疗）', 1],
    ['LDH（首次治疗）>240是预后不良因素', 1],
    ['β2微球蛋白β2-MG（首次治疗）>3是预后不良因素', 1]
]

con_features_show = [
    '(1) 启动一线治疗时年龄（岁）',
    # '(2) 启动一线治疗时ECOG评分（体力评分，一般>2是预后不良因素）',
    # '(3) 启动一线治疗时是否存在B症状', 
    # '(4) 启动一线治疗时病理分级',
    # '(5) 启动一线治疗时AnnArbor分期',
    '(2) 启动一线治疗时受累淋巴结区数目（大于等于5预后不良）',
    # '(7) 启动一线治疗时骨髓是否受累',
    '(3) 启动一线治疗时最大病灶直径（cm）',
    '(4) 启动一线治疗时病灶的最高 SUVmax',
    '(5) 启动一线治疗时白细胞（WBC）数值（10^9/L）',
    '(6) 启动一线治疗时单核细胞绝对值（10^9/L）',
    '(7) 启动一线治疗时淋巴细胞绝对值（10^9/L）',
    '(8) 启动一线治疗时淋巴单核细胞比',
    '(9) 启动一线治疗时血红蛋白（HGB）数值（g/L）（120是预后不良因素）',
    '(10) 启动一线治疗时血小板（PLT）数值（10^9/L）',
    '(11) 启动一线治疗时乳酸脱氢酶（LDH）数值（IU/L）',
    '(12) 启动一线治疗时β2微球蛋白（β2-MG）（>3mg/L 是预后不良因素）',
                    ]

def get_survival(val, average_val):
    fig, ax = plt.subplots(figsize = (8, 4))
    ax.set_xlabel('month')
    ax.set_ylabel('risk of progress')
    x = [6, 12, 24, 36]
    # val = [ 1-x for x in val]
    ax.plot(x, val, marker = 'o', linestyle = '-', color = 'r', label = 'patient')
    ax.plot(x, average_val, marker = '*', linestyle = '--', color = 'grey', label = 'average')
    ax.set_xticks(x)
    for x in average_val:
        val.append(x)
    ax.set_yticks(val)
    ax.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format = 'png')
    url = base64.b64encode(buf.getvalue()).decode('utf8')
    buf.close()
    plt.close(fig)
    return url

with open("Z.pkl", "rb") as f:
    Z = pickle.load(f)

with open("X.pkl", "rb") as f:
    X = pickle.load(f)

with open("y_combined.pkl", "rb") as f:
    y_combined = pickle.load(f)

def get_svm(y, x):
    x1_min, x1_max, x2_min, x2_max = [-0.4831802622349537, 0.4370673107224893, -0.04382664337769359, 1.004533718080136]
    xx, yy = np.meshgrid(np.arange(x1_min, x1_max, 0.01),
                     np.arange(x2_min, x2_max, 0.01))
    Z.reshape(xx.shape)
    cmap = ListedColormap(['blue', 'orange', 'green', 'red'])
    fig, ax = plt.subplots(figsize = (10, 8))
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
    colors = ['blue', 'orange', 'green', 'red']
    labels = ['Rweichi=0, pod24=0', 'Rweichi=0, pod24=1', 'Rweichi=1, pod24=0', 'Rweichi=1, pod24=1']
    for i in range(4):
        ax.scatter(X[y_combined == i]['uplift'], X[y_combined == i]['predict_y'], 
            color=colors[i], label=labels[i], edgecolor='k', s=70, alpha=0.7)
    # ax.set_xlabel('uplift')
    # ax.set_ylabel('predict_y')
    ax.set_title('SVM Decision Boundaries for Four Classes')
    
    ax.scatter([x], [y], color='cyan', label=f'the patient',
               edgecolor='yellow', linewidth=0.5, s=400, marker='*', alpha=1)
    # 横向箭头（获益等级，朝左）
    ax.arrow(x1_max, x2_min - 0.03, x1_min - x1_max, 0, head_width=0.03, head_length=0.04, fc='black', ec='black')
    ax.text((x1_min + x1_max) / 2, x2_min - 0.07, 'The Level of Benefit Is Better', fontsize=12, color='black', ha='center')
    
    # 纵向箭头（进展风险，移到顶部）
    ax.arrow(x1_min - 0.03, x2_min, 0, - x2_min + x2_max, head_width=0.03, head_length=0.04, fc='black', ec='black')
    ax.text(x1_min - 0.07, (x2_min + x2_max) / 2, 'The Risk of Progress Is Greater', fontsize=12, color='black', rotation=90, va='center')
    
    # 调整图例位置到右下角
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
    
    # 调整布局
    plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.2)
    
    buf = io.BytesIO()
    plt.savefig(buf, format = 'png')
    url = base64.b64encode(buf.getvalue()).decode('utf8')
    buf.close()
    plt.close(fig)
    return url

def get_rwch(uplift_val, conc, per_up):
    # 获益值 ｜ 推荐情况
    # uplift_val（保留两位小数） ｜ conc（直接显示）
    # f"预计采用 R 维持的获益等级高于 {per_up:.2f}% 的人群。"
    data = [
        ["获益值", "推荐情况"],
        ["{:.2f}".format(uplift_val), conc],
        [f"预计采用 R 维持的获益等级高于 {per_up:.2f}% 的人群。"]
    ]
    fig, ax = plt.subplots(figsize = (8, 3))
    ax.axis('off')
    cell_width = 0.5
    cell_height = 0.5
    for i in range(4):
        ax.add_line(Line2D([0, 2 * cell_width], [i * cell_height, i * cell_height], color='black'))
    for j in range(3):
        if j == 1:
            ax.add_line(Line2D([j * cell_width, j * cell_width], [cell_height, 3 * cell_height], color='black'))
        else:
            ax.add_line(Line2D([j * cell_width, j * cell_width], [0, 3 * cell_height], color='black'))
    ax.text(0.5 * cell_width, 2.5 * cell_height, data[0][0], ha='center', va='center', fontproperties='STHeiti', fontsize = 20)
    ax.text(1.5 * cell_width, 2.5 * cell_height, data[0][1], ha='center', va='center', fontproperties='STHeiti', fontsize = 20)
    ax.text(0.5 * cell_width, 1.5 * cell_height, data[1][0], ha='center', va='center', fontproperties='STHeiti', fontsize = 20)
    ax.text(1.5 * cell_width, 1.5 * cell_height, data[1][1], ha='center', va='center', fontproperties='STHeiti', fontsize = 20)
    ax.text(1.0 * cell_width, 0.5 * cell_height, data[2][0], ha='center', va='center', fontproperties='STHeiti', fontsize = 20)
    ax.set_xlim(0, 2 * cell_width)
    ax.set_ylim(0, 3 * cell_height)
    buf = io.BytesIO()
    plt.savefig(buf, format = 'png')
    url = base64.b64encode(buf.getvalue()).decode('utf8')
    buf.close()
    plt.close(fig)
    return url

from matplotlib.font_manager import FontProperties
    
from matplotlib.lines import Line2D
def get_pod(pod_val, pod_per):
    # 进行时间（月） ｜ 6 ｜ 12 ｜ 24 ｜ 36
    # 进展概率 ｜ pod_val[0] | pod_val[1] | pod_val[2] | pod_val[3]
    # 超过人群百分比 ｜ pod_per[0] | pod_per[1] | pod_per[2] | pod_per[3]
    data = [
        ['进行时间(月)', 6,12,24,36],
        ['进展概率', "{:.2f}".format(pod_val[0]) , "{:.2f}".format(pod_val[1]) , "{:.2f}".format(pod_val[2]) , "{:.2f}".format(pod_val[3])],
        ['超过人群百分比(%)', "{:.2f}".format(pod_per[0]) , "{:.2f}".format(pod_per[1]) , "{:.2f}".format(pod_per[2]) , "{:.2f}".format(pod_per[3])]
    ]
    fig, ax = plt.subplots(figsize = (10, 3))
    ax.axis('off')
    cell_width = 0.07
    cell_height = 0.07
    font = FontProperties(family=['STHeiti'])
    for i in range(4):
        ax.add_line(Line2D([0, 6 * cell_width], [i * cell_height, i * cell_height], color='black'))
    ax.add_line(Line2D([0, 0], [0, 3 * cell_height], color='black'))
    for j in range(1, 7):
        if j == 1:
            ax.add_line(Line2D([j * 2 * cell_width, j * 2 * cell_width], [0, 3 * cell_height], color='black'))
        else:
            ax.add_line(Line2D([(j + 1) * cell_width, (j + 1) * cell_width], [0, 3 * cell_height], color='black'))
    for i in range(3):
        for j in range(5):
            x_pos = (j + 0.5) * (2 * cell_width if j == 0 else cell_width) + (j != 0) * cell_width
            ax.text(x_pos, (2.5 - i) * cell_height, data[i][j],
                    fontsize=20, ha='center', va='center', fontproperties=font)
    ax.set_xlim(0, 6 * cell_width)
    ax.set_ylim(0, 3 * cell_height)
    buf = io.BytesIO()
    plt.savefig(buf, format = 'png')
    url = base64.b64encode(buf.getvalue()).decode('utf8')
    buf.close()
    plt.close(fig)
    return url

def get_features(input_data, uplift_val):
    url = []

    def calculate_mean_uplift_per_feature_bucket(df, feature, n_bins=10):
        # 复制数据并删除 feature 列中的空值
        df_copy = df.copy().dropna(subset=[feature])
        # 使用 pd.qcut 进行分桶，并返回分桶标签和边界
        df_copy['feature_bucket'], bins = pd.qcut(df_copy[feature], q=n_bins, labels=False, retbins=True)
        # 计算每个分桶的 uplift 均值
        mean_uplift_per_feature_bucket = df_copy.groupby('feature_bucket')['uplift'].mean()
        return mean_uplift_per_feature_bucket, bins

    def vis_combined_plots(df, feature, val, has_val, uplift_val, bucket='bucket', n_bins=10):
        # 计算每个特征分桶的 uplift 均值和分桶边界
        mean_uplift_per_feature_bucket, bins = calculate_mean_uplift_per_feature_bucket(df, feature, n_bins)
        
        # 确保 bins 是数值类型
        bins = np.array(bins, dtype=float)  # 强制转换为 float 类型
        
        # 计算每个 uplift 分桶的特征均值
        df_cleaned = df
        mean_feature_per_bucket = df_cleaned.groupby(bucket)[feature].mean()
        
        # 创建画布
        fig, axes = plt.subplots(1, 2, figsize=(24, 10))
        
        # 图1：特征值分布直方图（10 分桶）
        df[feature].hist(bins=10, color='lightcoral', edgecolor='black', ax=axes[0])
        axes[0].set_ylabel('Count')
        axes[1].set_xlabel('Feature Value')
        if has_val:
            axes[0].axvline(x=val, color='blue', linestyle='dashed', linewidth=2)
            axes[0].text(val, axes[0].get_ylim()[1], ' patient', ha='center', va='bottom')
        
        # 图2：Mean Uplift per Feature Bucket
        # 找到 val 所在的桶
        colors = ["skyblue"] * n_bins
        if has_val:
            val_bin_index = np.digitize(val, bins) - 1
            if 0 <= val_bin_index < n_bins:
                colors[val_bin_index] = "orange"  # 将 val 所在的桶颜色标为橙色
        
        axes[1].bar(range(n_bins), mean_uplift_per_feature_bucket.values, color=colors)
        axes[1].set_xlabel('Feature Bucket')
        axes[1].set_ylabel('Mean Uplift')
        
        # 设置 x 轴标签为分桶范围
        bin_labels = [f'{bins[i]:.2f} - {bins[i+1]:.2f}' for i in range(len(bins)-1)]
        axes[1].set_xticks(range(n_bins))
        axes[1].set_xticklabels(bin_labels, rotation=45)
        for i, v in enumerate(mean_uplift_per_feature_bucket.values):
            axes[1].text(i, v, f'{v:.2f}', ha='center', va='bottom')
        
        # # 图3：Mean of Feature per Uplift Bucket
        # # 找到 uplift_val 所在的桶
        # uplift_bin_index = np.digitize(uplift_val, mean_feature_per_bucket.index) - 1
        # colors = ['green'] * len(mean_feature_per_bucket)
        # if 0 <= uplift_bin_index < len(mean_feature_per_bucket):
        #     colors[uplift_bin_index] = "red"  # 将 uplift_val 所在的桶颜色标为红色
        
        # axes[2].bar(mean_feature_per_bucket.index, mean_feature_per_bucket.values, color=colors)
        # axes[2].set_xlabel('Bucket')
        # for i, v in enumerate(mean_feature_per_bucket.values):
        #     axes[2].text(i, v, f'{v:.2f}', ha='center', va='bottom')
        
        # 调整布局并保存图像
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        url = base64.b64encode(buf.getvalue()).decode('utf8')
        buf.close()
        plt.close(fig)
        return url
    
    for [f, _] in zip(need_features, con_features_show) :
        val = 0
        has_val = False
        fe = f[0]
        con = f[1]
        if not pd.isna(input_data.loc[0, fe]):
            val = input_data.loc[0, fe]
            has_val = True
            _ = _ + " (患者值 ：" + str(val) + ")"
        else :
            _ = _ + " (表中未提供患者数据)"
        
        if con:
            url.append([_, vis_combined_plots(ori_df, fe, val, has_val, uplift_val)])
    
    return url
    
    
def solve(pod_val, pod_per, uplift_val, conc, per_up, df, all_pod_val):
    # 一、2*4 表格

    # 1. pod - 6 12 24 36 的概率
    # 2. 分别位于 top 百分数
    
    pod_url = get_pod(pod_val, pod_per)
    
    # 二、survival 分析 曲线  
    url_survival = get_survival(pod_val, all_pod_val)
    
    # 三、Rweichi 2*2 表格
        # uplift 获益性，推荐值
        # uplift 的 top 百分数
    url_rwch = get_rwch(uplift_val, conc, per_up)
    
    # 四、2维图
    url_svm = get_svm(pod_val[2], uplift_val)
    
    # 五、重要指标

    # 1. 重要指标全人群分布柱状图，用竖线显示患者的位置
    # 2. 重要指标全人群获益性分桶图，用竖线显示患者的位置
    
    urls_fea = get_features(df, uplift_val)
    
    
    return pod_url, url_survival, url_rwch, url_svm, urls_fea

def main_solve(df):
    uplift_val = []
    pod_val = []
    pod_per = []
    all_pod_val = []
    # df.drop('Rweichi', axis=1)
    uplift_val.append(DML_6.predict(df, if_train=True, if_cali=False)[0])
    uplift_val.append(DML_12.predict(df, if_train=True, if_cali=False)[0])
    uplift_val.append(DML_24.predict(df, if_train=True, if_cali=False)[0])
    uplift_val.append(DML_36.predict(df, if_train=True, if_cali=False)[0])
    
    pod_val.append(DML_6.predict_outcome(df, if_train=False, if_cali=True)[0][0])
    pod_val.append(DML_12.predict_outcome(df, if_train=False, if_cali=True)[0][0])
    pod_val.append(DML_24.predict_outcome(df, if_train=False, if_cali=True)[0][0])
    pod_val.append(DML_36.predict_outcome(df, if_train=False, if_cali=True)[0][0])
    # print ("pod_val[0] = ", pod_val[2])
    pod_per.append(100 * ((predict_y_6 <= pod_val[0]).mean()))
    pod_per.append(100 * ((predict_y_12 <= pod_val[1]).mean()))
    pod_per.append(100 * ((predict_y_24 <= pod_val[2]).mean()))
    pod_per.append(100 * ((predict_y_36 <= pod_val[3]).mean()))
    all_pod_val.append(predict_y_6.mean())
    all_pod_val.append(predict_y_12.mean())
    all_pod_val.append(predict_y_24.mean())
    all_pod_val.append(predict_y_36.mean())
    
    tmp = pd.DataFrame([[uplift_val[2], pod_val[2]]], columns=['uplift', 'predict_y'])
    tmp_class = SVM.predict(tmp)[0]
    conc =  "不推荐" if tmp_class == 0 else ("非常推荐" if tmp_class == 1 else "推荐")
    
    per_up = 100 * ((uplift_samples >= uplift_val[2]).mean())
    # uplift = f"预计采用 R 维持的获益等级高于 {per_up:.2f}% 的人群。"
    
    return solve(pod_val, pod_per, uplift_val[2], conc, per_up, df, all_pod_val)

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
        '治疗时单个淋巴结是否大于6cm（大于6预后不良因素）': (1.0 if float(inputs[1])>=6.0 else 0.0) if pd.notna(inputs[1]) else np.nan,
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
        # 'Rweichi': 1 if inputs[22] == "有" else 0,
        # '一线PFS(确诊时间-进展时间，需要写函数)': float(inputs[23]) if pd.notna(inputs[23]) else np.nan,
        # '一线后是否进展': 1 if inputs[24] == "是" else 0,
        # 'pod24': 1 if inputs[25] == "有" else 0,
        # '治疗时FLIPI-1（最终版）': float(inputs[26]) if pd.notna(inputs[26]) else np.nan,
        # '治疗FLIPI2（最终版）': float(inputs[27]) if pd.notna(inputs[27]) else np.nan,
        # '一线治疗时PRIMA-PI': float(inputs[28]) if pd.notna(inputs[28]) else np.nan,
    }
    
    input_df = pd.DataFrame([input_data])
    
    for f in cate_features:
        input_df[f] = input_df[f].apply(
            lambda x: labelEncoderDict[f].get(x, labelEncoderDict[f].get(np.nan, np.nan))
        )
    
    input_df[cate_features] = input_df[cate_features].astype('str')
    
    return main_solve(input_df)

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
    # lymph_node_size = request.form['lymph_node_size']
    lymph_node_size = None
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
    # rweichi = request.form['rweichi']
    rweichi = None
    # fps = request.form['fps']
    fps = None
    progress_after = None
    # progress_after = request.form['progress_after']
    pod24 = None
    FLIPI_1 = request.form['FLIPI_1']
    FLIPI_2= request.form['FLIPI_2']
    PRIMA_PI = request.form['PRIMA_PI']
    patient_id = request.form['patient_id']
    
    predict_pod24 = request.form['predict_pod24']
    predict_uplift = request.form['predict_uplift']
    conc_human_0 = request.form['conc_human_0']
    
    if reason_for_treatment == '9 ：其他（请记录具体因素）':
        reason_for_treatment = request.form['other_reason']

    inputs = [
        gender, max_tumor_size, suv_max, ecog_score, b_symptoms, initial_stage, treatment_age, pathology_grade,
        lymph_node_count, bone_marrow_involvement, lymph_node_size, ldh, beta2_mg, wbc, hgb, plt, monocyte,
        lymphocyte, lymph_monocyte_ratio, reason_for_treatment, classification, efficacy_group, rweichi,
        fps, progress_after, pod24, FLIPI_1, FLIPI_2, PRIMA_PI, patient_id
    ]
    return jsonify(result = predict(inputs))


import csv
# 定义 CSV 文件的列顺序
csv_columns = [
    'patient_id', 'gender', 'max_tumor_size', 'suv_max', 'ecog_score', 'b_symptoms',
    'initial_stage', 'treatment_age', 'pathology_grade', 'lymph_node_count',
    'bone_marrow_involvement', 'lymph_node_size', 'ldh', 'beta2_mg', 'wbc', 'hgb',
    'plt', 'monocyte', 'lymphocyte', 'lymph_monocyte_ratio', 'reason_for_treatment',
    'classification', 'efficacy_group', 'rweichi', 'fps', 'progress_after', 'pod24',
    'FLIPI_1', 'FLIPI_2', 'PRIMA_PI', 'predict_pod24', 'predict_uplift', 'conc_human_0',
    'conc_human_1'
]
csv_file = 'save.csv'


@app.route('/save', methods=['POST'])
def save_route():
    patient_id = request.form['patient_id']
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
    lymph_node_size = None
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
    rweichi = None
    fps = None
    progress_after = None
    pod24 = None
    FLIPI_1 = request.form['FLIPI_1']
    FLIPI_2= request.form['FLIPI_2']
    PRIMA_PI = request.form['PRIMA_PI']
    patient_id = request.form['patient_id']
    
    predict_pod24 = request.form['predict_pod24']
    predict_uplift = request.form['predict_uplift']
    conc_human_0 = request.form['conc_human_0']
    conc_human_1 = request.form['conc_human_1']
    
    if reason_for_treatment == '9 ：其他（请记录具体因素）':
        reason_for_treatment = request.form['other_reason']
        
    data = {
        'patient_id': patient_id,
        'gender': gender,
        'max_tumor_size': max_tumor_size,
        'suv_max': suv_max,
        'ecog_score': ecog_score,
        'b_symptoms': b_symptoms,
        'initial_stage': initial_stage,
        'treatment_age': treatment_age,
        'pathology_grade': pathology_grade,
        'lymph_node_count': lymph_node_count,
        'bone_marrow_involvement': bone_marrow_involvement,
        'lymph_node_size': lymph_node_size,
        'ldh': ldh,
        'beta2_mg': beta2_mg,
        'wbc': wbc,
        'hgb': hgb,
        'plt': plt,
        'monocyte': monocyte,
        'lymphocyte': lymphocyte,
        'lymph_monocyte_ratio': lymph_monocyte_ratio,
        'reason_for_treatment': reason_for_treatment,
        'classification': classification,
        'efficacy_group': efficacy_group,
        'rweichi': rweichi,
        'fps': fps,
        'progress_after': progress_after,
        'pod24': pod24,
        'FLIPI_1': FLIPI_1,
        'FLIPI_2': FLIPI_2,
        'PRIMA_PI': PRIMA_PI,
        'predict_pod24': predict_pod24,
        'predict_uplift': predict_uplift,
        'conc_human_0': conc_human_0,
        'conc_human_1': conc_human_1
    }

    
    try:
        with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=csv_columns)
            # writer.writeheader()  # 写入表头
            writer.writerow(data)  # 写入数据
        print(f"数据已成功写入 {csv_file}")
    except Exception as e:
        print(f"写入 CSV 文件时出错: {e}")

    return jsonify(result = None)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port="8081", debug=True, threaded = False)