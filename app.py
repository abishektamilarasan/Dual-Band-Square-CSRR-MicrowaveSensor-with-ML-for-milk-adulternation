import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from milk_model import BidirectionalMilkModel
from PIL import Image

st.set_page_config(
    page_title="Milk Adulteration Microwave Sensor AI Platform",
    page_icon="🥛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38BDF8, #818CF8, #C084FC);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #94A3B8;
        font-weight: 500;
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        background-color: #EEF2FF;
        color: #4F46E5;
    }
</style>
""", unsafe_allow_html=True)

# Cache model instance
@st.cache_resource
def load_model():
    return BidirectionalMilkModel('milk_sensor_dataset.csv')

model = load_model()

# Header
st.markdown('<div class="main-title">🥛 Milk Sensor AI Analysis & Bidirectional Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">CSRR Microwave Resonator Sensor Characterization • Forward Inference • Inverse Synthesis • Research Reviews</div>', unsafe_allow_html=True)

# Navigation Tabs
tab_forward, tab_inverse, tab_data_insights, tab_report_review = st.tabs([
    "🎯 Forward Predictor (Sensor -> Adulterant & Conc)",
    "🔄 Inverse Predictor (Adulterant & Conc -> Sensor Params)",
    "📊 Dataset & Calibration Curves",
    "📑 Research Report & Review Gallery"
])


# TAB 1: FORWARD PREDICTION

with tab_forward:
    st.subheader("Predict Adulterant Type & Concentration from Microwave Response")
    st.info("Input measured resonant frequency ($f_1$), notch depth ($S_{21}$ dip), and bandwidth ($BW$) or Quality factor ($Q_1$). The model classifies the adulterant and predicts arbitrary/unseen continuous concentrations.")

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.markdown("### 🎛️ Sensor Inputs")
        
        # Preset selector for quick demo
        preset_choice = st.selectbox(
            "Load Sample Values from Dataset (Optional):",
            ["Custom Input"] + model.df['sample_label'].tolist()
        )

        if preset_choice != "Custom Input":
            preset_row = model.df[model.df['sample_label'] == preset_choice].iloc[0]
            default_f1 = float(preset_row['f1_GHz'])
            default_depth = float(preset_row['depth1_dB'])
            default_q1 = float(preset_row['Q1'])
            default_bw = float(preset_row['BW_MHz'])
        else:
            default_f1 = 1.9560
            default_depth = -22.500
            default_q1 = 9.00
            default_bw = 217.3

        f1_input = st.number_input("Resonant Frequency $f_1$ (GHz):", min_value=1.5, max_value=2.5, value=default_f1, step=0.001, format="%.4f")
        depth_input = st.number_input("Notch Depth / S21 Dip (dB):", min_value=-40.0, max_value=-5.0, value=default_depth, step=0.1, format="%.3f")
        
        bw_mode = st.radio("Specify Bandwidth or Q-Factor:", ["Quality Factor (Q1)", "3-dB Bandwidth (BW in MHz)"], horizontal=True)
        
        if bw_mode == "Quality Factor (Q1)":
            q1_input = st.number_input("Loaded Quality Factor ($Q_1$):", min_value=1.0, max_value=60.0, value=default_q1, step=0.1, format="%.2f")
            bw_mhz_val = (f1_input / q1_input) * 1000.0
            st.caption(f"Calculated Bandwidth $BW$: **{bw_mhz_val:.2f} MHz**")
            bw_input = None
        else:
            bw_input = st.number_input("Bandwidth $BW$ (MHz):", min_value=10.0, max_value=1000.0, value=default_bw, step=1.0, format="%.1f")
            q1_input = (f1_input * 1000.0) / bw_input
            st.caption(f"Calculated Quality Factor $Q_1$: **{q1_input:.2f}**")

        predict_btn = st.button("🚀 Run Forward Prediction", type="primary", use_container_width=True)

    with col2:
        st.markdown("### 💡 AI Diagnosis & Prediction Result")
        if predict_btn or preset_choice != "Custom Input":
            res = model.predict_forward(f1_input, depth_input, Q1=q1_input, BW_MHz=bw_input)
            
            adulterant = res['predicted_adulterant']
            conc = res['predicted_concentration']
            
            unit = "ml added" if adulterant == "Water" else ("g added" if adulterant == "Sugar" else ("M / frac" if adulterant == "MixedElectrolyte" else ""))
            
            st.markdown(f"""
            <div style="background-color: #F0FDF4; border: 2px solid #86EFAC; border-radius: 12px; padding: 20px; margin-bottom: 20px;">
                <h4 style="margin: 0; color: #166534;">Detected Material / Adulterant</h4>
                <p style="font-size: 1.8rem; font-weight: 700; color: #15803D; margin: 4px 0 10px 0;">{adulterant}</p>
                <h4 style="margin: 0; color: #166534;">Estimated Concentration</h4>
                <p style="font-size: 2.2rem; font-weight: 800; color: #166534; margin: 4px 0 0 0;">
                    {conc} <span style="font-size: 1.2rem; font-weight: 500;">{unit}</span>
                </p>
                <p style="font-size: 0.85rem; color: #15803D; margin-top: 8px;">Continuous estimation supported even for concentrations not in the original dataset.</p>
            </div>
            """, unsafe_allow_html=True)

            # Probabilities chart
            probs_df = pd.DataFrame(list(res['class_probabilities'].items()), columns=['Material', 'Probability'])
            fig_prob = px.bar(
                probs_df, x='Material', y='Probability', 
                title="Classification Confidence", 
                text='Probability', 
                color='Probability',
                color_continuous_scale="Blues",
                range_y=[0, 1.05]
            )
            fig_prob.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig_prob.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_prob, use_container_width=True)


# TAB 2: INVERSE PREDICTION

with tab_inverse:
    st.subheader("Predict Sensor Parameters for Any Adulterant & Concentration")
    st.info("Input an adulterant and an arbitrary concentration (e.g. 42 ml Water, 7.8g Sugar, or 0.05M Electrolyte). The surrogate physics model will predict the corresponding microwave resonant parameters.")

    inv_col1, inv_col2 = st.columns([1, 1.2])

    with inv_col1:
        st.markdown("### 🧪 Target Sample Parameters")
        target_adulterant = st.selectbox("Select Adulterant:", ["Water", "Sugar", "MixedElectrolyte", "Milk", "SyntheticMilk"])
        
        if target_adulterant == "Water":
            conc_val = st.slider("Water Concentration (ml):", min_value=0.0, max_value=120.0, value=42.0, step=0.5)
            unit_lbl = "ml"
        elif target_adulterant == "Sugar":
            conc_val = st.slider("Sugar Concentration (g):", min_value=0.0, max_value=25.0, value=7.5, step=0.25)
            unit_lbl = "g"
        elif target_adulterant == "MixedElectrolyte":
            conc_val = st.slider("Mixed Electrolyte Concentration:", min_value=0.00, max_value=0.20, value=0.05, step=0.005, format="%.3f")
            unit_lbl = "M"
        else:
            conc_val = 0.0
            st.caption("Reference sample without adulterant.")
            unit_lbl = ""

        generate_btn = st.button("⚡ Synthesize Microwave Response", type="primary", use_container_width=True)

    with inv_col2:
        st.markdown("### 📡 Predicted Microwave Sensor Response")
        inv_res = model.predict_inverse(target_adulterant, conc_val)

        m1, m2 = st.columns(2)
        with m1:
            st.metric("Resonant Frequency (f1)", f"{inv_res['predicted_f1_GHz']:.4f} GHz")
            st.metric("Transmission Notch Depth", f"{inv_res['predicted_depth1_dB']:.2f} dB")
        with m2:
            st.metric("Quality Factor (Q1)", f"{inv_res['predicted_Q1']:.2f}")
            st.metric("Bandwidth (BW)", f"{inv_res['predicted_BW_MHz']:.2f} MHz")
            
        st.metric("Dielectric Permittivity (ε'_r real)", f"{inv_res['predicted_eps_r_real']:.2f}")

        # Simulated S21 Resonance Curve
        freqs = np.linspace(inv_res['predicted_f1_GHz'] - 0.25, inv_res['predicted_f1_GHz'] + 0.25, 300)
        # Lorentzian / S21 notch approximation: S21(f) = S0 / sqrt(1 + 4*Q^2*((f - f0)/f0)^2)
        f0 = inv_res['predicted_f1_GHz']
        q = inv_res['predicted_Q1']
        depth = inv_res['predicted_depth1_dB']
        
        # approximate S21 notch shape
        s21_sim = depth / (1.0 + 4.0 * (q**2) * ((freqs - f0)/f0)**2)**0.5
        
        fig_s21 = go.Figure()
        fig_s21.add_trace(go.Scatter(x=freqs, y=s21_sim, mode='lines', name='Synthesized S21', line=dict(color='#2563EB', width=2.5)))
        fig_s21.add_vline(x=f0, line_dash="dash", line_color="red", annotation_text=f"f1 = {f0:.4f} GHz")
        fig_s21.update_layout(
            title=f"Synthesized Transmission Response (S21) for {target_adulterant} @ {conc_val}{unit_lbl}",
            xaxis_title="Frequency (GHz)",
            yaxis_title="|S21| (dB)",
            height=300,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_s21, use_container_width=True)


# TAB 3: DATASET & CALIBRATION

with tab_data_insights:
    st.subheader("Sensor Calibration Curves & Dataset Analysis")
    
    st.dataframe(model.df, use_container_width=True)

    c_plot1, c_plot2 = st.columns(2)
    with c_plot1:
        fig_scatter = px.scatter(
            model.df, 
            x='f1_GHz', 
            y='depth1_dB', 
            color='material_type', 
            size='Q1',
            hover_name='sample_label',
            title="Sensor Space: Resonant Frequency vs. Notch Depth"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Individual Adulterant Calibration Curves
    st.markdown("### 📈 Adulterant Calibration Curves")
    c_sub1, c_sub2, c_sub3 = st.columns(3)

    with c_sub1:
        water_df = model.df[model.df['material_type'] == 'Water'].sort_values(by='concentration')
        fig_water = px.line(
            water_df, 
            x='concentration', 
            y='f1_GHz', 
            markers=True, 
            title="💧 Water: Resonant Freq vs Concentration",
            labels={'concentration': 'Water Added (ml)', 'f1_GHz': 'f1 (GHz)'},
            color_discrete_sequence=['#38BDF8']
        )
        st.plotly_chart(fig_water, use_container_width=True)

    with c_sub2:
        sugar_df = model.df[model.df['material_type'] == 'Sugar'].sort_values(by='concentration')
        fig_sugar = px.line(
            sugar_df, 
            x='concentration', 
            y='f1_GHz', 
            markers=True, 
            title="🍬 Sugar: Resonant Freq vs Concentration",
            labels={'concentration': 'Sugar Added (g)', 'f1_GHz': 'f1 (GHz)'},
            color_discrete_sequence=['#F59E0B']
        )
        st.plotly_chart(fig_sugar, use_container_width=True)

    with c_sub3:
        fig_sugar_depth = px.line(
            sugar_df, 
            x='concentration', 
            y='depth1_dB', 
            markers=True, 
            title="🍬 Sugar: Notch Depth (dB) vs Concentration",
            labels={'concentration': 'Sugar Added (g)', 'depth1_dB': 'Notch Depth (dB)'},
            color_discrete_sequence=['#EC4899']
        )
        st.plotly_chart(fig_sugar_depth, use_container_width=True)


# TAB 4: REPORT & REVIEW GALLERY

with tab_report_review:
    st.subheader("📑 Research Papers, Reviews & Sensor Documentation")
    st.markdown("Upload your review reports, CST simulation screenshots, fabrication photos, or read parsed literature.")

    uploaded_files = st.file_uploader(
        "Upload Review Photos, CST Snapshots, or Evaluation Reports:", 
        type=['png', 'jpg', 'jpeg', 'pdf', 'txt'], 
        accept_multiple_files=True
    )

    if uploaded_files:
        st.success(f"Received {len(uploaded_files)} files!")
        img_cols = st.columns(3)
        for idx, file in enumerate(uploaded_files):
            col = img_cols[idx % 3]
            with col:
                if file.type.startswith('image'):
                    image = Image.open(file)
                    st.image(image, caption=file.name, use_container_width=True)
                else:
                    st.write(f"📄 **{file.name}** ({file.size / 1024:.1f} KB)")

    # Research Papers Available in Workspace
    paper_dir = os.path.join(os.path.dirname(__file__), "Research paper")
    if os.path.exists(paper_dir):
        with st.expander("📚 Available Research Papers in Repository"):
            papers = [f for f in os.listdir(paper_dir) if f.endswith('.pdf')]
            for p in papers:
                st.markdown(f"- 📄 `{p}`")
