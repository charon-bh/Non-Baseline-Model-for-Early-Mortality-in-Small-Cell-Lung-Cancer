import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import warnings
import re
from io import BytesIO
from PIL import Image
warnings.filterwarnings('ignore')

# ===================== 基础配置 =====================
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False    
plt.rcParams['font.family'] = 'DejaVu Sans'   
st.set_page_config(
    page_title="Non-Baseline Model for Predicting Early Mortality in Small Cell Lung Cancer",
    page_icon="📊",
    layout="wide"
)

# 【特征】
CATEGORICAL_FEATURES = [
    'Sex', 'Income_Group', 'T_Stage', 'N_Stage', 'M_Stage',
    'Brain_metastasis', 'Liver_metastasis', 'Urbanization_Level',
    'Radiation', 'Chemotherapy', 'Bone_metastasis'
]
CONTINUOUS_FEATURES = ['Age']
selected_features = CATEGORICAL_FEATURES + CONTINUOUS_FEATURES  

# ===================== 核心：特征值映射字典（完整覆盖所有特征） =====================
FEATURE_VALUE_MAPPING = {
    # 基础特征
    'Sex': {0: 'Female', 1: 'Male'},
    'Income_Group': {0: '<60000', 1: '60000-70000', 2: '70000-80000', 3: '>80000'},
    'Age': None,  # 连续特征无需映射
    
    # 分期特征
    'T_Stage': {0: 'T1', 1: 'T2', 2: 'T3', 3: 'T4'},
    'N_Stage': {0: 'N0', 1: 'N1', 2: 'N2', 3: 'N3'},
    'M_Stage': {0: 'M0', 1: 'M1'},
    
    # 转移特征
    'Brain_metastasis': {0: 'No', 1: 'Yes'},
    'Liver_metastasis': {0: 'No', 1: 'Yes'},
    'Bone_metastasis': {0: 'No', 1: 'Yes'},
    
    # 其他特征
    'Urbanization_Level': {0: 'High urbanization', 1: 'low urbanization', 2: 'medium urbanization'},
    'Radiation': {0: 'No/Unknown', 1: 'Yes'},
    'Chemotherapy': {0: 'No/Unknown', 1: 'Yes'},
    
    # 目标变量
    'Outcome': {0: 'No', 1: 'Yes'}
}

# ===================== 工具函数 =====================
def clean_text(text):
    """清理文本中的控制字符和异常符号"""
    # 移除所有控制字符 
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)
    # 替换全角符号为半角
    text = text.replace('：', ':').replace('，', ',').replace('。', '.')
    return text

def get_feature_text(feat_name, feat_value):
    """根据特征名和数字值，返回对应的文字描述"""
    if feat_name == 'Age':
        return str(feat_value)
    mapping = FEATURE_VALUE_MAPPING.get(feat_name, {})
    return mapping.get(feat_value, str(feat_value))

# 重构特征选项
FEATURE_OPTIONS = {}
for feat in CATEGORICAL_FEATURES:
    sorted_items = sorted(FEATURE_VALUE_MAPPING[feat].items(), key=lambda x: x[0])
    FEATURE_OPTIONS[feat] = [item[1] for item in sorted_items]  # 下拉框显示文本
    FEATURE_OPTIONS[f"{feat}_values"] = [item[0] for item in sorted_items]  # 对应数字值

# 特殊处理Age
FEATURE_OPTIONS['Age'] = list(range(20, 86))

# 文件路径
MODEL_PATH = "LightGBM_Optimal_Model.pkl"
TRAIN_DATA_PATH = "traindata.csv"

# ===================== 初始化模型和SHAP解释器 =====================
@st.cache_resource
def init_model():
    """加载模型和初始化SHAP解释器"""
    try:
        # 1. 加载模型
        model_dict = joblib.load(MODEL_PATH)
        model = model_dict['model']
        optimal_threshold = model_dict['optimal_threshold']
        st.success("✅ Model loaded successfully!")
        
        # 2. 加载训练集作为背景数据
        train_df = pd.read_csv(TRAIN_DATA_PATH, encoding="GBK")
        X_train = train_df[selected_features].copy()
        background_data = shap.sample(X_train, 100, random_state=42) if len(X_train) > 100 else X_train
        
        # 3. 初始化SHAP解释器
        explainer = shap.TreeExplainer(
            model=model,
            data=background_data,
            model_output="raw"
        )
        
        # 适配基准值
        base_value = explainer.expected_value
        base_value = base_value[1] if isinstance(base_value, list) else base_value
        base_value = float(base_value)
        
        return model, optimal_threshold, explainer, base_value
    
    except Exception as e:
        st.error(f"❌ Initialization failed: {str(e)}")
        return None, None, None, None

# ===================== 绘制SHAP力图（贡献分析） =====================
def plot_shap_force_plot(base_value, shap_values, feature_values, prob_1):
    """
    自定义SHAP力图：用水平条形图展示特征贡献
    红色=推高概率 | 蓝色=拉低概率 | 按贡献绝对值排序
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # 数据准备
    features = selected_features
    contributions = shap_values
    
    # 按贡献绝对值降序排序
    abs_contrib = np.abs(contributions)
    sorted_idx = np.argsort(abs_contrib)[::-1]
    sorted_feats = [features[i] for i in sorted_idx]
    sorted_contrib = contributions[sorted_idx]
    
    # 核心修改：将数字值转换为文字描述
    sorted_vals = [get_feature_text(features[i], feature_values[i]) for i in sorted_idx]
    
    # 绘制条形图
    colors = ['#d62728' if c > 0 else '#1f77b4' for c in sorted_contrib]
    bars = ax.barh(range(len(sorted_feats)), sorted_contrib, color=colors, alpha=0.8)
    
    # 样式设置
    ax.set_yticks(range(len(sorted_feats)))
    ax.set_yticklabels([f"{feat} (value: {val})" for feat, val in zip(sorted_feats, sorted_vals)], fontsize=11)
    ax.invert_yaxis()  # 重要的特征在上方
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)  # 零刻度线
    
    
    outcome_1_text = get_feature_text('Outcome', 1)
    xlabel_text = clean_text(f'SHAP contribution values (positive values increase the probability of Outcome={outcome_1_text}, negative values decrease it)')
    title_text = clean_text(f'SHAP Feature Contribution Analysis - Probability of Predicting Outcome={outcome_1_text}: {prob_1:.4f}')
    
    ax.set_xlabel(xlabel_text, fontsize=12)
    ax.set_title(title_text, fontsize=16, pad=20)
    ax.grid(axis='x', alpha=0.3)
    
    # 标注贡献值
    for i, bar in enumerate(bars):
        width = bar.get_width()
        label_x = width + 0.005 if width > 0 else width - 0.005
        ax.text(label_x, bar.get_y() + bar.get_height()/2, 
                f'{width:.4f}', va='center', ha='left' if width > 0 else 'right', fontsize=10)
    
    # 添加基准值和最终概率说明（使用半角符号）
    baseline_text = clean_text(f'Baseline value (average forecast): {base_value:.4f} | Final Prediction Probability: {prob_1:.4f}')
    ax.text(0.02, 0.02, baseline_text, 
            transform=ax.transAxes, fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # 保存为图片
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    return img

# ===================== 主页面 =====================
def main():
    st.title("📊 Non-Baseline Model for Predicting Early Mortality in Small Cell Lung Cancer")
    st.markdown("### Single-Sample SHAP Feature Contribution Plot")
    st.markdown("---")
    
    # 初始化
    model, optimal_threshold, explainer, base_value = init_model()
    if model is None:
        st.stop()
    
    # 左右分栏
    col_left, col_right = st.columns([1, 2])
    
    # ---------------------- 左侧：特征输入 ----------------------
    with col_left:
        st.subheader("🔍 Patient Characteristics Input")
        feature_values = []  # 存储传给模型的数字值
        
        # 11个分类特征输入（显示文字，存储数字）
        for feat in CATEGORICAL_FEATURES:
            display_options = FEATURE_OPTIONS[feat]
            value_options = FEATURE_OPTIONS[f"{feat}_values"]
            
            # 下拉框选择（显示文字）
            selected_text = st.selectbox(feat, display_options, index=0, key=f"input_{feat}")
            # 找到对应的数字值
            selected_idx = display_options.index(selected_text)
            selected_value = value_options[selected_idx]
            feature_values.append(selected_value)
        
        # 1个连续特征（Age）
        age_val = st.slider("Age", 20, 85, 50, key="input_Age")
        feature_values.append(age_val)
        
        # 预测按钮
        predict_btn = st.button("🚀 Execute Prediction & Generate SHAP Maps", type="primary")
    
    # ---------------------- 右侧：预测结果 + SHAP力图 ----------------------
    with col_right:
        st.subheader("📈 Prediction Results ")
        
        if not predict_btn:
            st.info("👈 After entering patient characteristics, click the button to view prediction results and SHAP feature contribution plots.")
        else:
            with st.spinner("Predicting and calculating SHAP values..."):
                # 1. 构造输入数据
                X_input = pd.DataFrame([feature_values], columns=selected_features)
                
                # 2. 模型预测
                prob = model.predict_proba(X_input)[0]
                prob_0, prob_1 = prob[0], prob[1]
                pred_class = 1 if prob_1 >= optimal_threshold else 0
                
                # 3. 计算SHAP值
                shap_output = explainer.shap_values(X_input)
                if isinstance(shap_output, list) and len(shap_output) == 2:
                    shap_values = shap_output[1][0]
                elif isinstance(shap_output, np.ndarray) and shap_output.ndim == 2:
                    shap_values = shap_output[0]
                else:
                    shap_values = np.array(shap_output).flatten()
                
                # ========== 显示预测结果（映射为文字） ==========
                outcome_0_text = get_feature_text('Outcome', 0)
                outcome_1_text = get_feature_text('Outcome', 1)
                pred_class_text = get_feature_text('Outcome', pred_class)
                
                st.markdown(f"""
                <div style="background: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                    <h4 style="margin:0 0 15px 0; color:#e74c3c;">Optimal Classification Threshold: {optimal_threshold:.3f}</h4>
                    <div style="display: flex; justify-content: space-around; align-items: center;">
                        <div style="text-align:center;">
                            <h5 style="color:#1f77b4; margin:0;">Outcome={outcome_0_text} Probability</h5>
                            <span style="font-size:2.5em; font-weight:bold; color:#1f77b4;">{prob_0*100:.2f}%</span>
                        </div>
                        <div style="text-align:center;">
                            <h5 style="color:#d62728; margin:0;">Outcome={outcome_1_text} Probability</h5>
                            <span style="font-size:2.5em; font-weight:bold; color:#d62728;">{prob_1*100:.2f}%</span>
                        </div>
                    </div>
                    <h3 style="color:{'#e74c3c' if pred_class==1 else '#27ae60'}; text-align:center; margin:15px 0 0 0;">
                        Prediction Category: {pred_class} ({pred_class_text}) - {'High-risk' if pred_class==1 else 'Lower-risk'}
                    </h3>
                </div>
                """, unsafe_allow_html=True)
                
                # ========== 显示SHAP力图 ==========
                st.subheader("SHAP Feature Contribution Plot")
                shap_fig = plot_shap_force_plot(base_value, shap_values, feature_values, prob_1)
                st.image(shap_fig, width=1000)
                st.markdown(f"💡 Red features → Increase the probability of Outcome={outcome_1_text} | Blue features → Decrease the probability")

# ===================== 执行主函数 =====================
if __name__ == "__main__":
    main()