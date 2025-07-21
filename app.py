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
    
    .metric-card {
        background-color: #f8f9fa;
        border: 2px solid #28a745;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 10px 0;
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

# STEP 4: AI Deep Dive & Training
if st.session_state.step3:
    st.markdown('<div class="step-box">', unsafe_allow_html=True)
    st.markdown("### 🧠 Step 4: Deep Dive into AI")
    
    # AI Categories explanation
    st.markdown("**🤖 3 Main Types of AI:**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🎯 Supervised Learning**
        - Has labeled data (answers)
        - Learns from examples
        - Makes predictions
        - **← We use this!**
        """)
        
    with col2:
        st.markdown("""
        **🔍 Unsupervised Learning**
        - No labels/answers
        - Finds hidden patterns
        - Groups similar data
        - Example: Customer segments
        """)
        
    with col3:
        st.markdown("""
        **🎮 Reinforcement Learning**
        - Learns by trial/error
        - Gets rewards/penalties
        - Improves over time
        - Example: Game AI
        """)
    
    st.info("💡 **Our case:** Supervised Learning → Classification (Yes/No adherence)")
    
    # Algorithm choice
    st.markdown("**🔧 Popular Classification Algorithms:**")
    
    algo_cols = st.columns(4)
    with algo_cols[0]:
        if st.button("🌳 Random Forest", key="rf_btn", help="Multiple decision trees voting"):
            st.session_state.chosen_algo = "Random Forest"
    with algo_cols[1]:
        if st.button("🧮 Logistic Regression", key="lr_btn", help="Linear probability model"):
            st.session_state.chosen_algo = "Logistic Regression"
    with algo_cols[2]:
        if st.button("🎯 SVM", key="svm_btn", help="Support Vector Machine"):
            st.session_state.chosen_algo = "SVM"
    with algo_cols[3]:
        if st.button("🏘️ KNN", key="knn_btn", help="K-Nearest Neighbors"):
            st.session_state.chosen_algo = "KNN"
    
    # Default choice
    if 'chosen_algo' not in st.session_state:
        st.session_state.chosen_algo = "Random Forest"
    
    st.success(f"**Selected:** {st.session_state.chosen_algo} ✅")
    
    # Random Forest Explanation
    if st.session_state.chosen_algo == "Random Forest":
        st.markdown("**🌳 How Random Forest Works:**")
        
        # Visual explanation with emojis
        st.markdown("""
        ```
        🌳 Tree 1: "Age < 60? → Yes=Adherent, No=Check cost"
        🌳 Tree 2: "Cost < $3000? → Yes=Adherent, No=Non-adherent"  
        🌳 Tree 3: "Side effects < 3? → Yes=Adherent, No=Check age"
        🌳 ... (97 more trees)
        
        📊 Final Vote: 60 trees say "Adherent", 40 say "Non-adherent"
        ✅ Result: ADHERENT (60% confidence)
        ```
        """)
        
        # Simple diagram
        st.markdown("**🗳️ Voting Process:**")
        vote_data = pd.DataFrame({
            'Prediction': ['Adherent', 'Non-Adherent'], 
            'Trees Voting': [65, 35],
            'Percentage': ['65%', '35%']
        })
        st.dataframe(vote_data, use_container_width=True)
        
        st.info("💡 **Why Random Forest?** More trees = more opinions = better accuracy!")
    
    # Training section
    st.markdown("---")
    st.markdown("**🚀 Let's Train Our AI:**")
    
    st.markdown('<div class="code-box">model = RandomForestClassifier(n_estimators=100); model.fit(X_train, y_train)</div>', unsafe_allow_html=True)
    
    if st.button("▶️ Execute Code", key="btn4"):
        with st.spinner("🧠 Training AI model..."):
            
            # Progress bar simulation
            progress_bar = st.progress(0)
            import time
            
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress((i + 1))
                if i % 20 == 0:
                    st.text(f"Training tree {i+1}/100...")
            
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(st.session_state.X_train, st.session_state.y_train)
            
            st.session_state.model = model
            st.session_state.step4 = True
        
        st.success("🎉 100 decision trees trained successfully!")
        
        # Feature importance with simple explanation
        importance = model.feature_importances_
        features = ['Age', 'Annual Cost', 'Side Effects']
        
        st.markdown("**🎯 What the AI learned (Feature Importance):**")
        
        for i, (feature, imp) in enumerate(zip(features, importance)):
            st.write(f"**{feature}:** {imp:.1%} importance")
            st.progress(imp)
        
        # Simple interpretation
        most_important = features[np.argmax(importance)]
        st.info(f"🏆 **Most important factor:** {most_important}")
        
        st.markdown('<div class="success-output">✅ AI learned: {} is the key factor for adherence prediction!</div>'.format(most_important), unsafe_allow_html=True)
    
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
        
        # Simple, business-friendly results
        cm = confusion_matrix(y_test, predictions)
        tn, fp, fn, tp = cm.ravel()
        
        total_patients = len(y_test)
        correct_predictions = tp + tn
        wrong_predictions = fp + fn
        
        # Calculate simple metrics
        correctly_predicted_adherent = tp
        correctly_predicted_non_adherent = tn
        missed_adherent_patients = fn  # We said no, but they were adherent
        false_alarms = fp  # We said yes, but they weren't adherent
        
        st.markdown("### 📊 AI Performance Results")
        
        # Main accuracy metric - large and prominent
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h2 style="color: #28a745; margin: 0;">🎯 {accuracy:.1%}</h2>
                <h3 style="margin: 5px 0;">Overall Accuracy</h3>
                <p style="margin: 0; color: #666;">Got {correct_predictions} out of {total_patients} predictions right</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Detailed breakdown in simple terms
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### ✅ What We Got Right")
            st.metric("✅ Correctly Identified Adherent Patients", f"{correctly_predicted_adherent}")
            st.metric("✅ Correctly Identified Non-Adherent Patients", f"{correctly_predicted_non_adherent}")
            st.success(f"**Total Correct:** {correct_predictions} patients")
            
        with col2:
            st.markdown("### ❌ What We Got Wrong")
            st.metric("⚠️ Missed Adherent Patients", f"{missed_adherent_patients}")
            st.metric("🚨 False Alarms", f"{false_alarms}")
            st.error(f"**Total Wrong:** {wrong_predictions} patients")
        
        # Business impact explanation
        st.markdown("### 💼 Business Impact")
        
        col1, col2 = st.columns(2)
        with col1:
            if missed_adherent_patients > 0:
                st.warning(f"⚠️ **Risk:** {missed_adherent_patients} adherent patients might not get needed support")
            else:
                st.success("✅ No adherent patients were missed!")
        
        with col2:
            if false_alarms > 0:
                st.info(f"💰 **Cost:** {false_alarms} patients might receive unnecessary interventions")
            else:
                st.success("✅ No unnecessary interventions!")
        
        # Overall assessment
        if accuracy > 0.8:
            st.success("🎉 **Excellent Performance!** This AI is ready for deployment.")
        elif accuracy > 0.7:
            st.warning("⚠️ **Good Performance** - Consider fine-tuning before deployment.")
        else:
            st.error("❌ **Needs Improvement** - More training data or different approach needed.")
        
        st.markdown('<div class="success-output">✅ AI testing complete - {:.1%} accuracy achieved</div>'.format(accuracy), unsafe_allow_html=True)
    
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
    - ✅ Analyzed data patterns inside 1,000 patient records
    - ✅ Trained AI with 100 decision trees
    - ✅ Achieved {:.1%} prediction accuracy
    - ✅ Made real-time predictions
    """.format(st.session_state.get('accuracy', 0)))

st.markdown("---")
st.markdown("*🤖 AI Drug Adherence Predictor - Educational Demo*")
