import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import warnings
from io import BytesIO
from PIL import Image
warnings.filterwarnings('ignore')

# ===================== 基础配置 =====================
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False    
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

# 特征可选值
FEATURE_OPTIONS = {
    'Sex': [0, 1],
    'Income_Group': [0, 1, 2, 3],
    'T_Stage': [0, 1, 2, 3],
    'N_Stage': [0, 1, 2, 3],
    'M_Stage': [0, 1],
    'Brain_metastasis': [0, 1],
    'Liver_metastasis': [0, 1],
    'Urbanization_Level': [0, 1, 2],
    'Radiation': [0, 1],
    'Chemotherapy': [0, 1],
    'Bone_metastasis': [0, 1],
    'Age': list(range(20, 86))
}

# 文件路径
MODEL_PATH = "LightGBM_Optimal_Model.pkl"
TRAIN_DATA_PATH = "traindata.csv"

# ===================== 初始化模型和SHAP解释器 =====================
@st.cache_resource
def init_model():
    """加载模型和初始化SHAP解释器"""
    # 1. 加载模型
    try:
        model_dict = joblib.load(MODEL_PATH)
        model = model_dict['model']
        optimal_threshold = model_dict['optimal_threshold']
        st.success("✅ Model loaded successfully！")
    except Exception as e:
        st.error(f"❌ 模型加载失败：{str(e)}")
        return None, None, None
    
    # 2. 加载训练集作为背景数据
    try:
        train_df = pd.read_csv(TRAIN_DATA_PATH, encoding="GBK")
        X_train = train_df[selected_features].copy()
        background_data = shap.sample(X_train, 100, random_state=42) if len(X_train) > 100 else X_train
    except Exception as e:
        st.error(f"❌ 训练集加载失败：{str(e)}")
        return None, None, None
    
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
    sorted_vals = [feature_values[i] for i in sorted_idx]
    
    # 绘制条形图
    colors = ['#d62728' if c > 0 else '#1f77b4' for c in sorted_contrib]
    bars = ax.barh(range(len(sorted_feats)), sorted_contrib, color=colors, alpha=0.8)
    
    # 样式设置
    ax.set_yticks(range(len(sorted_feats)))
    ax.set_yticklabels([f"{feat} (value：{val})" for feat, val in zip(sorted_feats, sorted_vals)], fontsize=11)
    ax.invert_yaxis()  # 重要的特征在上方
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)  # 零刻度线
    ax.set_xlabel('SHAP contribution values (positive values increase the probability of Outcome=1, negative values decrease it)）', fontsize=12)
    ax.set_title(f'SHAP Feature Contribution Analysis - Probability of Predicting Outcome=1：{prob_1:.4f}', fontsize=16, pad=20)
    ax.grid(axis='x', alpha=0.3)
    
    # 标注贡献值
    for i, bar in enumerate(bars):
        width = bar.get_width()
        label_x = width + 0.005 if width > 0 else width - 0.005
        ax.text(label_x, bar.get_y() + bar.get_height()/2, 
                f'{width:.4f}', va='center', ha='left' if width > 0 else 'right', fontsize=10)
    
    # 添加基准值和最终概率说明
    ax.text(0.02, 0.02, f'Baseline value (average forecast)：{base_value:.4f} | Final Prediction Probability：{prob_1:.4f}', 
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
        feature_values = []
        
        # 11个分类特征
        for feat in CATEGORICAL_FEATURES:
            val = st.selectbox(feat, FEATURE_OPTIONS[feat], index=0, key=f"input_{feat}")
            feature_values.append(val)
        
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
                
                # ========== 显示预测结果 ==========
                st.markdown(f"""
                <div style="background: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                    <h4 style="margin:0 0 15px 0; color:#e74c3c;">Optimal Classification Threshold：{optimal_threshold:.3f}</h4>
                    <div style="display: flex; justify-content: space-around; align-items: center;">
                        <div style="text-align:center;">
                            <h5 style="color:#1f77b4; margin:0;">Outcome=0 Probability</h5>
                            <span style="font-size:2.5em; font-weight:bold; color:#1f77b4;">{prob_0*100:.2f}%</span>
                        </div>
                        <div style="text-align:center;">
                            <h5 style="color:#d62728; margin:0;">Outcome=1 Probability</h5>
                            <span style="font-size:2.5em; font-weight:bold; color:#d62728;">{prob_1*100:.2f}%</span>
                        </div>
                    </div>
                    <h3 style="color:{'#e74c3c' if pred_class==1 else '#27ae60'}; text-align:center; margin:15px 0 0 0;">
                        Prediction Category：{pred_class} ({'High-risk' if pred_class==1 else 'Lower-risk'})
                    </h3>
                </div>
                """, unsafe_allow_html=True)
                
                # ========== 显示SHAP力图 ==========
                st.subheader("SHAP Feature Contribution Plot")
                shap_fig = plot_shap_force_plot(base_value, shap_values, feature_values, prob_1)
                st.image(shap_fig, width=1000)  # 替换废弃的use_column_width参数
                st.markdown("💡 Red features → Increase the probability of Outcome=1 | Blue features → Decrease the probability")

# ===================== 执行主函数 =====================
if __name__ == "__main__":
    main()