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
    page_title="MyWay Drug Adherence Predictor",
    page_icon="💊",
    layout="wide"
)

# CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #772583, #4A1A4F);
        color: white;
        padding: 30px;
        text-align: center;
        border-radius: 15px;
        margin-bottom: 30px;
    }
    
    .story-section {
        background-color: #f8f9fa;
        border-left: 5px solid #772583;
        padding: 20px;
        margin: 20px 0;
        border-radius: 10px;
    }
    
    .step-box {
        background-color: #ffffff;
        border: 2px solid #e9ecef;
        padding: 25px;
        margin: 20px 0;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
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
    
    .metric-card {
        background-color: #f8f9fa;
        border: 2px solid #28a745;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 10px 0;
    }
    
    .sanofi-logo {
        position: fixed;
        bottom: 20px;
        right: 20px;
        width: 80px;
        opacity: 0.7;
        z-index: 1000;
    }
    
    .workflow-box {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        border: 1px solid #dee2e6;
        padding: 20px;
        margin: 15px 0;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Sanofi Logo
st.markdown("""
<img src="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD//gA7Q1JFQVRPUjogZ2QtanBlZyB2MS4wICh1c2luZyBJSkcgSlBFRyB2ODApLCBxdWFsaXR5ID0gODUK/9sAQwAFAwQEBAMFBAQEBQUFBgcMCAcHBwcPCwsJDBEPEhIRDxERExYcFxMUGhURERghGBodHR8fHxMXIiQiHiQcHh8e/9sAQwEFBQUHBgcOCAgOHhQRFB4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4e/8AAEQgAUABQAwEiAAIRAQMRAf/EAB8AAAEFAQEBAQEBAAAAAAAAAAABAgMEBQYHCAkKC//EALUQAAIBAwMEAgECBQYGZzCBCyEAATIDBBUFBhIHFwhhCAkiFCsBARACEQMTFjFBMiQIZDVHFyJhgZEVcaGxwUIjNEKCkWJy0bIkM0OhwfDBtTVSgtJj4uvBZIaGdI9t/EAB8BAAMBAQEBAQEBAQEAAAAAAAABAgMEBQYHCAkKC//EALURAAIBAgQEAwQHBQQEAAECdwABAgMRBAUhMQYSQVEyYXEHgZEiI6GxwRMkQYKS0fAUVHLRovASY3MzNTdEYXOhQ8FP//aAAAwDAQACEQMRAD8A=" class="sanofi-logo" alt="Sanofi Logo">
""", unsafe_allow_html=True)

# Header with story context
st.markdown("""
<div class="main-header">
    <h1>MyWay Drug Adherence AI Predictor</h1>
    <h3>Your Journey as a Data Scientist</h3>
    <p>Build an AI system that can predict patient medication adherence</p>
</div>
""", unsafe_allow_html=True)

# Story Introduction
st.markdown("""
<div class="story-section">
    <h2>The Challenge</h2>
    <p><strong>MyWay</strong> is Sanofi's innovative patient support program that helps patients navigate their treatment journey. However, we face a critical challenge: <strong>50% of patients don't take their medications as prescribed.</strong></p>
    
    <p>Poor medication adherence leads to:</p>
    <ul>
        <li>Worsened patient outcomes</li>
        <li>Increased healthcare costs (€125 billion annually in Europe)</li>
        <li>Higher hospitalization rates</li>
    </ul>
    
    <p><strong>Your Mission:</strong> As a data scientist, you'll build an AI system that can identify patients at risk of poor adherence before it happens, enabling proactive interventions.</p>
</div>
""", unsafe_allow_html=True)

# Data Science Workflow
st.markdown("""
<div class="workflow-box">
    <h3>The Data Science Workflow</h3>
    <p>Every data scientist follows these core steps:</p>
    <strong>1. Understand the Problem</strong> → <strong>2. Collect & Explore Data</strong> → <strong>3. Prepare Data</strong> → <strong>4. Build Model</strong> → <strong>5. Evaluate Performance</strong> → <strong>6. Deploy & Predict</strong>
</div>
""", unsafe_allow_html=True)

# Initialize session state
for key in ['step1', 'step2', 'step3', 'step4', 'step5', 'step6', 'data', 'model']:
    if key not in st.session_state:
        st.session_state[key] = False

# STEP 1: Data Collection
st.markdown('<div class="step-box">', unsafe_allow_html=True)
st.markdown("### Step 1: Data Collection - Understanding Patient Patterns")
st.markdown("""**The Data Scientist's First Task:** Gather and generate realistic patient data that mirrors real-world adherence patterns.

In the real MyWay program, we would analyze:
- Patient demographics and medical history
- Treatment costs and insurance coverage  
- Reported side effects and quality of life measures
- Historical adherence patterns

For this demo, we'll generate 1,000 synthetic patient records with key adherence factors.""")

st.markdown('<div class="code-box">np.random.seed(42); data = generate_realistic_patient_data(1000)</div>', unsafe_allow_html=True)

if st.button("Execute: Generate Patient Dataset", key="btn1"):
    with st.spinner("Generating realistic patient data..."):
        np.random.seed(42)
        n_patients = 1000
        
        # Generate realistic patient data
        age = np.random.normal(65, 15, n_patients)
        age = np.clip(age, 18, 90)
        
        cost = np.random.exponential(2000, n_patients)
        cost = np.clip(cost, 500, 8000)
        
        side_effects = np.random.poisson(2, n_patients)
        side_effects = np.clip(side_effects, 0, 8)
        
        # Generate adherence based on clinical insights
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
    
    st.markdown(f'<div class="success-output">Dataset Created: {len(data)} patient records ready for analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(data.head(10))
    with col2:
        adherent_pct = data['adherent'].mean() * 100
        st.metric("Baseline Adherence Rate", f"{adherent_pct:.1f}%")
        st.markdown("**Key Variables:**")
        st.write("• Age: Patient age in years")
        st.write("• Annual Cost: Treatment cost per year")  
        st.write("• Side Effects: Number of reported side effects")
        st.write("• Adherent: 1=Yes, 0=No")

st.markdown('</div>', unsafe_allow_html=True)

# STEP 2: Data Analysis
if st.session_state.step1:
    st.markdown('<div class="step-box">', unsafe_allow_html=True)
    st.markdown("### Step 2: Exploratory Data Analysis - Finding the Patterns")
    st.markdown("""**The Data Scientist's Analysis:** Before building any model, we need to understand what drives adherence.

This is where domain expertise meets data science. Clinical research suggests that cost, side effects, and patient age are key factors.""")
    
    st.markdown('<div class="code-box">analyze_adherence_patterns(data)</div>', unsafe_allow_html=True)
    
    if st.button("Execute: Analyze Adherence Drivers", key="btn2"):
        data = st.session_state.data
        
        adherent_data = data[data['adherent'] == 1]
        non_adherent_data = data[data['adherent'] == 0]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_age_yes = adherent_data['age'].mean()
            avg_age_no = non_adherent_data['age'].mean()
            st.metric("Average Age", "")
            st.write(f"**Adherent:** {avg_age_yes:.0f} years")
            st.write(f"**Non-adherent:** {avg_age_no:.0f} years")
            if avg_age_yes < avg_age_no:
                st.success("Insight: Younger patients show better adherence")
            
        with col2:
            avg_cost_yes = adherent_data['annual_cost'].mean()
            avg_cost_no = non_adherent_data['annual_cost'].mean()
            st.metric("Average Annual Cost", "")
            st.write(f"**Adherent:** ${avg_cost_yes:.0f}")
            st.write(f"**Non-adherent:** ${avg_cost_no:.0f}")
            if avg_cost_yes < avg_cost_no:
                st.success("Insight: Lower costs improve adherence")
            
        with col3:
            avg_effects_yes = adherent_data['side_effects'].mean()
            avg_effects_no = non_adherent_data['side_effects'].mean()
            st.metric("Average Side Effects", "")
            st.write(f"**Adherent:** {avg_effects_yes:.1f}")
            st.write(f"**Non-adherent:** {avg_effects_no:.1f}")
            if avg_effects_yes < avg_effects_no:
                st.success("Insight: Fewer side effects = better adherence")
        
        st.session_state.step2 = True
        st.markdown('<div class="success-output">Key Pattern Identified: Age, Cost, and Side Effects are the primary adherence drivers</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# STEP 3: Data Preparation
if st.session_state.step2:
    st.markdown('<div class="step-box">', unsafe_allow_html=True)
    st.markdown("### Step 3: Data Preparation - Setting Up for Machine Learning")
    st.markdown("""**Critical Data Science Step:** Split the data into training and testing sets.

**Training Set (70%):** Used to teach the AI model patterns
**Test Set (30%):** Used to evaluate how well the model performs on unseen data

This prevents overfitting and ensures our model will work on real patients.""")
    
    st.markdown('<div class="code-box">X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3)</div>', unsafe_allow_html=True)
    
    if st.button("Execute: Prepare Training Data", key="btn3"):
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
            st.metric("Training Set", f"{len(X_train)} patients")
            st.write("Used to train the AI model")
        with col2:
            st.metric("Test Set", f"{len(X_test)} patients") 
            st.write("Used to validate performance")
        
        st.markdown('<div class="success-output">Data successfully prepared for machine learning</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# STEP 4: Model Building & Training
if st.session_state.step3:
    st.markdown('<div class="step-box">', unsafe_allow_html=True)
    st.markdown("### Step 4: AI Model Development - Building the Prediction Engine")
    
    st.markdown("""**Algorithm Selection:** For this healthcare application, we'll use Random Forest - an ensemble method that combines multiple decision trees.

**Why Random Forest for Healthcare?**
- Handles mixed data types (age, cost, categorical symptoms)
- Provides interpretable results that clinicians can understand  
- Robust against outliers and missing data
- Gives confidence scores for predictions""")
    
    # Show decision tree visualization
    try:
        st.image("decision_tree.png", 
                 caption="Example Decision Tree: How AI decides patient adherence",
                 use_column_width=True)
        
        st.markdown("""**How the Algorithm Works:**
- Each tree asks different questions about the patient
- Trees vote on the final prediction  
- 100 trees provide more reliable predictions than 1 tree
- The majority vote determines adherence likelihood""")
        
    except FileNotFoundError:
        st.markdown("""**Random Forest Concept:**
```
Tree 1: Age < 60? → Cost < $2000? → Side Effects < 2? → ADHERENT
Tree 2: Side Effects < 3? → Age < 65? → Cost < $3000? → ADHERENT  
Tree 3: Cost < $1500? → Age < 70? → Side Effects < 4? → ADHERENT
... (97 more trees)

Final Vote: 65 trees say "ADHERENT", 35 say "NON-ADHERENT"
Result: ADHERENT (65% confidence)
```""")
    
    st.markdown('<div class="code-box">model = RandomForestClassifier(n_estimators=100); model.fit(X_train, y_train)</div>', unsafe_allow_html=True)
    
    if st.button("Execute: Train AI Model", key="btn4"):
        with st.spinner("Training Random Forest with 100 decision trees..."):
            progress_bar = st.progress(0)
            import time
            
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress((i + 1))
            
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(st.session_state.X_train, st.session_state.y_train)
            
            st.session_state.model = model
            st.session_state.step4 = True
        
        # Show what the AI learned
        importance = model.feature_importances_
        features = ['Age', 'Annual Cost', 'Side Effects']
        
        st.markdown("**AI Learning Results - Feature Importance:**")
        
        for feature, imp in zip(features, importance):
            st.write(f"**{feature}:** {imp:.1%} importance")
            st.progress(imp)
        
        most_important = features[np.argmax(importance)]
        st.markdown('<div class="success-output">Model Training Complete: {} identified as the strongest predictor</div>'.format(most_important), unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# STEP 5: Model Evaluation
if st.session_state.step4:
    st.markdown('<div class="step-box">', unsafe_allow_html=True)
    st.markdown("### Step 5: Performance Evaluation - Validating Clinical Utility")
    st.markdown("""**The Critical Question:** Can our AI system make accurate predictions on patients it has never seen before?

**Clinical Validation:** We test on the 30% of data the model never saw during training. This simulates real-world deployment.""")
    
    st.markdown('<div class="code-box">predictions = model.predict(X_test); clinical_accuracy = accuracy_score(y_test, predictions)</div>', unsafe_allow_html=True)
    
    if st.button("Execute: Validate Model Performance", key="btn5"):
        model = st.session_state.model
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
        
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        st.session_state.predictions = predictions
        st.session_state.accuracy = accuracy
        st.session_state.step5 = True
        
        # Clinical performance metrics
        cm = confusion_matrix(y_test, predictions)
        tn, fp, fn, tp = cm.ravel()
        
        # Main accuracy display
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h2 style="color: #28a745; margin: 0;">{accuracy:.1%}</h2>
                <h3 style="margin: 5px 0;">Clinical Accuracy</h3>
                <p style="margin: 0; color: #666;">Correctly predicted {tp + tn} out of {len(y_test)} patients</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Clinical interpretation
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Correct Predictions:**")
            st.metric("Adherent Patients Identified", f"{tp}")
            st.metric("Non-Adherent Patients Identified", f"{tn}")
            
        with col2:
            st.markdown("**Prediction Errors:**")
            st.metric("Missed Non-Adherent Patients", f"{fn}")
            st.metric("False Adherence Predictions", f"{fp}")
        
        # Clinical impact assessment
        if accuracy > 0.8:
            st.success("Excellent performance - Ready for clinical deployment")
        elif accuracy > 0.7:
            st.warning("Good performance - Consider additional refinement")
        else:
            st.error("Performance needs improvement")
        
        st.markdown('<div class="success-output">Clinical Validation Complete: {:.1%} accuracy on unseen patients</div>'.format(accuracy), unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# STEP 6: Real-time Prediction
if st.session_state.step5:
    st.markdown('<div class="step-box">', unsafe_allow_html=True)
    st.markdown("### Step 6: Clinical Deployment - Real Patient Predictions")
    st.markdown("""**MyWay Integration:** Your AI model is now ready to help healthcare providers identify at-risk patients in real-time.

**Clinical Workflow:** When a new patient enrolls in MyWay, enter their information below to get an adherence prediction.""")
    
    st.markdown('<div class="code-box">adherence_prediction = predict_patient_adherence(patient_data)</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**New Patient Intake:**")
        patient_age = st.slider("Patient Age", 18, 90, 65)
        patient_cost = st.slider("Annual Treatment Cost ($)", 500, 8000, 2000)
        patient_effects = st.slider("Reported Side Effects", 0, 8, 2)
        
        if st.button("Execute: Predict Adherence Risk", key="btn6"):
            model = st.session_state.model
            
            new_patient = np.array([[patient_age, patient_cost, patient_effects]])
            prediction = model.predict(new_patient)[0]
            probability = model.predict_proba(new_patient)[0]
            
            if prediction == 1:
                st.success(f"LIKELY ADHERENT - Confidence: {probability[1]:.1%}")
                st.markdown("**Recommendation:** Standard MyWay support program")
            else:
                st.error(f"AT RISK FOR NON-ADHERENCE - Confidence: {probability[0]:.1%}")
                st.markdown("**Recommendation:** Enhanced support with personalized interventions")
    
    with col2:
        st.markdown("**Patient Risk Profile:**")
        st.write(f"Age: {patient_age} years")
        st.write(f"Annual Cost: ${patient_cost:,}")
        st.write(f"Side Effects: {patient_effects}")
        
        # Risk factor analysis
        risk_factors = []
        if patient_cost > 4000:
            risk_factors.append("High treatment cost")
        if patient_effects > 4:
            risk_factors.append("Multiple side effects")
        if patient_age > 75:
            risk_factors.append("Advanced age")
            
        if risk_factors:
            st.warning("**Risk Factors Identified:**")
            for factor in risk_factors:
                st.write(f"• {factor}")
        else:
            st.success("No major risk factors identified")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Impact Summary
if st.session_state.step5:
    st.markdown("---")
    st.markdown("""
    ## Mission Accomplished: Your AI System is Ready
    
    **What You've Built:**
    - Analyzed 1,000 patient records to identify adherence patterns
    - Trained an AI model with {:.1%} clinical accuracy  
    - Created a real-time prediction system for MyWay
    - Enabled proactive patient interventions
    

    """.format(st.session_state.get('accuracy', 0)))

st.markdown("---")
st.markdown("*MyWay Drug Adherence AI - Transforming Patient Care Through Data Science*")
