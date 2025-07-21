import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.express as px

# Page config
st.set_page_config(
    page_title="🤖 AI Drug Adherence Predictor",
    page_icon="💊",
    layout="wide"
)

# CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        color: white;
        padding: 20px;
        text-align: center;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    
    .step-box {
        background-color: #f8f9fa;
        border-left: 4px solid #1f77b4;
        padding: 15px;
        margin: 15px 0;
        border-radius: 5px;
    }
    
    .code-box {
        background-color: #1e1e1e;
        color: #dcdcaa;
        padding: 15px;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        border: 1px solid #444;
    }
    
    .success-output {
        background-color: #d4edda;
        border: 1px solid #28a745;
        padding: 10px;
        border-radius: 5px;
        color: #155724;
        font-family: 'Courier New', monospace;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🤖 AI Drug Adherence Predictor</h1>
    <h3>Learn AI Step by Step</h3>
</div>
""", unsafe_allow_html=True)

st.markdown("**Build an AI that predicts if patients will take their medication correctly**")

# Initialize session state
for key in ['step1', 'step2', 'step3', 'step4', 'step5', 'step6', 'data', 'model']:
    if key not in st.session_state:
        st.session_state[key] = False

# STEP 1: Generate Data
st.markdown('<div class="step-box">', unsafe_allow_html=True)
st.markdown("### 📊 Step 1: Create Patient Data")
st.markdown("Generate synthetic patient data with age, cost, and side effects")

st.markdown('<div class="code-box">np.random.seed(42); data = generate_patient_data(1000)</div>', unsafe_allow_html=True)

if st.button("▶️ Execute Code", key="btn1"):
    with st.spinner("Running AI code..."):
        np.random.seed(42)
        n_patients = 1000
        
        age = np.random.normal(65, 15, n_patients)
        age = np.clip(age, 18, 90)
        
        cost = np.random.exponential(2000, n_patients)
        cost = np.clip(cost, 500, 8000)
        
        side_effects = np.random.poisson(2, n_patients)
        side_effects = np.clip(side_effects, 0, 8)
        
        # Generate adherence based on realistic logic
        adherence_score = (-0.02 * age - 0.0003 * cost - 0.5 * side_effects + 
                          np.random.normal(0, 1, n_patients))
        adherence = (adherence_score > np.median(adherence_score)).astype(int)
        
        data = pd.DataFrame({
            'age': age.round(0).astype(int),
            'annual_cost': cost.round(0).astype(int),
            'side_effects': side_effects,
            'adherent': adherence
        })
        
        st.session_state.data = data
        st.session_state.step1 = True
    
    st.markdown(f'<div class="success-output">✅ Generated {len(data)} patient records</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(data.head())
    with col2:
        adherent_pct = data['adherent'].mean() * 100
        st.metric("Adherent Patients", f"{adherent_pct:.1f}%")

st.markdown('</div>', unsafe_allow_html=True)

# STEP 2: Explore Data
if st.session_state.step1:
    st.markdown('<div class="step-box">', unsafe_allow_html=True)
    st.markdown("### 🔍 Step 2: Analyze Data Patterns")
    st.markdown("Visualize how age, cost and side effects affect adherence")
    
    st.markdown('<div class="code-box">fig = px.box(data, x="adherent", y="age"); fig.show()</div>', unsafe_allow_html=True)
    
    if st.button("▶️ Execute Code", key="btn2"):
        data = st.session_state.data
        
        # Simple averages comparison
        adherent_data = data[data['adherent'] == 1]
        non_adherent_data = data[data['adherent'] == 0]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_age_yes = adherent_data['age'].mean()
            avg_age_no = non_adherent_data['age'].mean()
            
            st.metric("👴 Average Age", "")
            st.write(f"**Adherent:** {avg_age_yes:.0f} years")
            st.write(f"**Non-adherent:** {avg_age_no:.0f} years")
            
            if avg_age_yes < avg_age_no:
                st.success("✅ Younger patients take meds better!")
            else:
                st.info("📊 Older patients take meds better!")
            
        with col2:
            avg_cost_yes = adherent_data['annual_cost'].mean()
            avg_cost_no = non_adherent_data['annual_cost'].mean()
            
            st.metric("💰 Average Cost", "")
            st.write(f"**Adherent:** ${avg_cost_yes:.0f}")
            st.write(f"**Non-adherent:** ${avg_cost_no:.0f}")
            
            if avg_cost_yes < avg_cost_no:
                st.success("✅ Cheaper = Better adherence!")
            else:
                st.info("📊 Expensive = Better adherence!")
            
        with col3:
            avg_effects_yes = adherent_data['side_effects'].mean()
            avg_effects_no = non_adherent_data['side_effects'].mean()
            
            st.metric("😷 Average Side Effects", "")
            st.write(f"**Adherent:** {avg_effects_yes:.1f}")
            st.write(f"**Non-adherent:** {avg_effects_no:.1f}")
            
            if avg_effects_yes < avg_effects_no:
                st.success("✅ Fewer side effects = Better!")
            else:
                st.info("📊 More side effects = Better!")
        
        # Simple bar chart
        st.markdown("**📊 Quick Summary:**")
        summary_data = pd.DataFrame({
            'Group': ['Adherent', 'Non-Adherent'],
            'Average Age': [avg_age_yes, avg_age_no],
            'Average Cost': [avg_cost_yes, avg_cost_no],
            'Side Effects': [avg_effects_yes, avg_effects_no]
        })
        
        fig = px.bar(summary_data, x='Group', y='Average Age', 
                     title="Age Comparison", color='Group',
                     color_discrete_map={'Adherent': 'green', 'Non-Adherent': 'red'})
        st.plotly_chart(fig, use_container_width=True)
        
        st.session_state.step2 = True
        st.markdown('<div class="success-output">✅ Simple patterns found: Younger + Cheaper + Less side effects = Better adherence</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# STEP 3: Split Data
if st.session_state.step2:
    st.markdown('<div class="step-box">', unsafe_allow_html=True)
    st.markdown("### ✂️ Step 3: Split Data for Training")
    st.markdown("Divide data into training (70%) and testing (30%) sets")
    
    st.markdown('<div class="code-box">X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)</div>', unsafe_allow_html=True)
    
    if st.button("▶️ Execute Code", key="btn3"):
        data = st.session_state.data
        
        X = data[['age', 'annual_cost', 'side_effects']]
        y = data['adherent']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.session_state.step3 = True
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Training Samples", len(X_train))
        with col2:
            st.metric("Testing Samples", len(X_test))
        
        st.markdown('<div class="success-output">✅ Data successfully split</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# STEP 4: Train AI Model
if st.session_state.step3:
    st.markdown('<div class="step-box">', unsafe_allow_html=True)
    st.markdown("### 🧠 Step 4: Train AI Model")
    st.markdown("Use Random Forest algorithm (100 decision trees voting together)")
    
    st.markdown('<div class="code-box">model = RandomForestClassifier(n_estimators=100); model.fit(X_train, y_train)</div>', unsafe_allow_html=True)
    
    if st.button("▶️ Execute Code", key="btn4"):
        with st.spinner("Training AI model..."):
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(st.session_state.X_train, st.session_state.y_train)
            
            st.session_state.model = model
            st.session_state.step4 = True
        
        # Feature importance
        importance = model.feature_importances_
        features = ['Age', 'Annual Cost', 'Side Effects']
        
        fig = px.bar(x=features, y=importance, title="Feature Importance")
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('<div class="success-output">✅ AI model trained with 100 decision trees</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# STEP 5: Test AI Model
if st.session_state.step4:
    st.markdown('<div class="step-box">', unsafe_allow_html=True)
    st.markdown("### 🎯 Step 5: Test AI Performance")
    st.markdown("Evaluate how accurately our AI predicts on unseen data")
    
    st.markdown('<div class="code-box">predictions = model.predict(X_test); accuracy = accuracy_score(y_test, predictions)</div>', unsafe_allow_html=True)
    
    if st.button("▶️ Execute Code", key="btn5"):
        model = st.session_state.model
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
        
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        st.session_state.predictions = predictions
        st.session_state.accuracy = accuracy
        st.session_state.step5 = True
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("AI Accuracy", f"{accuracy:.1%}")
            
            # Confusion Matrix
            cm = confusion_matrix(y_test, predictions)
            fig = px.imshow(cm, 
                           labels=dict(x="AI Prediction", y="Reality"),
                           x=['Not Adherent', 'Adherent'],
                           y=['Not Adherent', 'Adherent'],
                           title="Confusion Matrix")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Performance breakdown
            tn, fp, fn, tp = cm.ravel()
            st.write(f"**Correct Predictions:** {tp + tn}")
            st.write(f"**Wrong Predictions:** {fp + fn}")
            st.write(f"**Total Tests:** {len(y_test)}")
        
        st.markdown('<div class="success-output">✅ AI achieved {:.1%} accuracy</div>'.format(accuracy), unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# STEP 6: Make Predictions
if st.session_state.step5:
    st.markdown('<div class="step-box">', unsafe_allow_html=True)
    st.markdown("### 🔮 Step 6: Predict New Patient")
    st.markdown("Use trained AI to predict adherence for a new patient")
    
    st.markdown('<div class="code-box">new_patient = [age, cost, side_effects]; prediction = model.predict([new_patient])</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Enter Patient Data:**")
        patient_age = st.slider("Age", 18, 90, 65)
        patient_cost = st.slider("Annual Cost ($)", 500, 8000, 2000)
        patient_effects = st.slider("Side Effects", 0, 8, 2)
        
        if st.button("▶️ Execute Code", key="btn6"):
            model = st.session_state.model
            
            new_patient = np.array([[patient_age, patient_cost, patient_effects]])
            prediction = model.predict(new_patient)[0]
            probability = model.predict_proba(new_patient)[0]
            
            if prediction == 1:
                st.success(f"✅ **ADHERENT** (Confidence: {probability[1]:.1%})")
                st.balloons()
            else:
                st.error(f"❌ **NON-ADHERENT** (Confidence: {probability[0]:.1%})")
            
            st.markdown(f'<div class="success-output">✅ AI prediction complete</div>', unsafe_allow_html=True)
    
    with col2:
        if st.session_state.get('step1'):
            data = st.session_state.data
            
            st.markdown("**Patient Profile:**")
            st.write(f"Age: {patient_age} years")
            st.write(f"Cost: ${patient_cost}")
            st.write(f"Side Effects: {patient_effects}")
            
            # Risk factors
            if patient_cost > 4000:
                st.warning("⚠️ High cost risk factor")
            if patient_effects > 4:
                st.warning("⚠️ High side effects")
            if patient_age > 75:
                st.info("ℹ️ Elderly patient - monitor closely")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Summary
if st.session_state.step5:
    st.markdown("---")
    st.markdown("""
    ## 🎉 Congratulations! You Built an AI System
    
    **What you accomplished:**
    - ✅ Generated 1,000 patient records
    - ✅ Analyzed data patterns  
    - ✅ Trained AI with 100 decision trees
    - ✅ Achieved {:.1%} prediction accuracy
    - ✅ Made real-time predictions
    
    **Real-world applications:**
    - 🏥 Hospital patient monitoring
    - 💊 Pharmacy intervention programs  
    - 📱 Mobile health apps
    - 🔬 Clinical research
    """.format(st.session_state.get('accuracy', 0)))

st.markdown("---")
st.markdown("*🤖 AI Drug Adherence Predictor - Educational Demo*")
