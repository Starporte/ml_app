import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import sys
from io import StringIO

# Configuration
st.set_page_config(
    page_title="ML Tutorial for Sanofi",
    page_icon="🧬",
    layout="wide"
)

# CSS simple
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #0056b3, #007bff);
        color: white;
        padding: 20px;
        text-align: center;
        border-radius: 10px;
        margin-bottom: 30px;
    }
    
    .step-box {
        background-color: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 15px;
        margin: 15px 0;
        border-radius: 5px;
    }
    
    .code-output {
        background-color: #e8f5e8;
        border: 1px solid #28a745;
        padding: 10px;
        border-radius: 5px;
        font-family: monospace;
        white-space: pre-wrap;
    }
    
    .error-output {
        background-color: #f8d7da;
        border: 1px solid #dc3545;
        padding: 10px;
        border-radius: 5px;
        font-family: monospace;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1> You Can DO AI !</h1>
    <h3>Interactive Adherence Drug Prediction - DLBT Session</h3>
</div>
""", unsafe_allow_html=True)

# Introduction
st.markdown("""
## 🎯 What you'll learn in 30 minutes

This **guided tutorial** teaches you machine learning basics through a simple drug discovery example. 
Each step has **10 lines of code maximum** and runs instantly.

**How it works:** Read the explanation → Run the code → See results → Move to next step
""")

# Code execution function
def run_code(code):
    old_stdout = sys.stdout
    sys.stdout = captured = StringIO()
    
    context = {
        'pd': pd, 'np': np, 'train_test_split': train_test_split,
        'LinearRegression': LinearRegression, 'r2_score': r2_score
    }
    
    try:
        exec(code, context)
        output = captured.getvalue()
        return True, output, context
    except Exception as e:
        return False, str(e), context
    finally:
        sys.stdout = old_stdout

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = {}

# Step 1
st.markdown('<div class="step-box">', unsafe_allow_html=True)
st.markdown("### 📊 Step 1: Create Drug Data")
st.markdown("**What:** Generate synthetic drug properties (molecular weight, solubility) and their effectiveness")

code1 = st.text_area("Step 1 Code:", value="""# Create synthetic drug data
import numpy as np
import pandas as pd

np.random.seed(42)
molecular_weight = np.random.normal(300, 50, 100)  # Drug weight
solubility = np.random.normal(0.5, 0.2, 100)      # Water solubility
effectiveness = (molecular_weight * 0.1 + solubility * 50 + 
                np.random.normal(0, 5, 100))       # Drug effectiveness

data = pd.DataFrame({
    'molecular_weight': molecular_weight,
    'solubility': solubility, 
    'effectiveness': effectiveness
})

print("Drug dataset created!")
print(data.head())""", height=200, key="code1")

if st.button("▶️ Run Step 1", key="run1"):
    success, output, context = run_code(code1)
    st.session_state.results['step1'] = context
    if success:
        st.markdown(f'<div class="code-output">{output}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="error-output">Error: {output}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Step 2
st.markdown('<div class="step-box">', unsafe_allow_html=True)
st.markdown("### 🔬 Step 2: Prepare Training Data")
st.markdown("**What:** Split data into training and testing sets (like testing drugs in lab vs clinic)")

code2 = st.text_area("Step 2 Code:", value="""# Split data for training and testing
from sklearn.model_selection import train_test_split

# Use data from Step 1 (uncomment if running separately)
# data = st.session_state.results['step1']['data']

X = data[['molecular_weight', 'solubility']]  # Input features
y = data['effectiveness']                      # Target to predict

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print("Data split completed!")""", height=180, key="code2")

if st.button("▶️ Run Step 2", key="run2"):
    # Merge previous context
    if 'step1' in st.session_state.results:
        code2_with_data = code2.replace("# data = st.session_state.results['step1']['data']", 
                                       f"data = {st.session_state.results['step1']['data'].to_dict()}")
        code2_with_data = "data = pd.DataFrame(" + str(st.session_state.results['step1']['data'].to_dict()) + ")\n" + code2
    else:
        code2_with_data = code2
    
    success, output, context = run_code(code2_with_data)
    st.session_state.results['step2'] = context
    if success:
        st.markdown(f'<div class="code-output">{output}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="error-output">Error: {output}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Step 3
st.markdown('<div class="step-box">', unsafe_allow_html=True)
st.markdown("### 🤖 Step 3: Train AI Model")
st.markdown("**What:** Teach the AI to predict drug effectiveness from molecular properties")

code3 = st.text_area("Step 3 Code:", value="""# Train machine learning model
from sklearn.linear_model import LinearRegression

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

print("🎯 Model trained successfully!")
print(f"Model learned from {len(X_train)} drug samples")

# Show what the model learned
print("\\nModel insights:")
print(f"Molecular weight importance: {model.coef_[0]:.3f}")
print(f"Solubility importance: {model.coef_[1]:.3f}")""", height=160, key="code3")

if st.button("▶️ Run Step 3", key="run3"):
    success, output, context = run_code(code3)
    st.session_state.results['step3'] = context
    if success:
        st.markdown(f'<div class="code-output">{output}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="error-output">Error: {output}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Step 4
st.markdown('<div class="step-box">', unsafe_allow_html=True)
st.markdown("### 🔮 Step 4: Make Predictions")
st.markdown("**What:** Use the trained AI to predict effectiveness of new, unseen drugs")

code4 = st.text_area("Step 4 Code:", value="""# Make predictions on test drugs
from sklearn.metrics import r2_score

# Predict effectiveness of test drugs
y_pred = model.predict(X_test)

print("🔮 Predictions made!")
print("\\nSample predictions:")
for i in range(5):
    actual = y_test.iloc[i]
    predicted = y_pred[i]
    print(f"Drug {i+1}: Actual={actual:.1f}, Predicted={predicted:.1f}")

# Calculate accuracy
accuracy = r2_score(y_test, y_pred)
print(f"\\n📊 Model accuracy: {accuracy:.3f} (0=bad, 1=perfect)")""", height=180, key="code4")

if st.button("▶️ Run Step 4", key="run4"):
    success, output, context = run_code(code4)
    st.session_state.results['step4'] = context
    if success:
        st.markdown(f'<div class="code-output">{output}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="error-output">Error: {output}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Step 5
st.markdown('<div class="step-box">', unsafe_allow_html=True)
st.markdown("### 🎯 Step 5: Design New Drug")
st.markdown("**What:** Use your AI model to design a new drug with optimal properties")

code5 = st.text_area("Step 5 Code:", value="""# Design new drug using AI predictions
import numpy as np

# Test different drug compositions
new_drugs = pd.DataFrame({
    'molecular_weight': [250, 300, 350, 400],
    'solubility': [0.3, 0.5, 0.7, 0.9]
})

# Predict their effectiveness
predicted_effectiveness = model.predict(new_drugs)

print("🧬 New drug candidates:")
for i, row in new_drugs.iterrows():
    effectiveness = predicted_effectiveness[i]
    print(f"Drug {i+1}: Weight={row['molecular_weight']}, "
          f"Solubility={row['solubility']}, "
          f"Predicted effectiveness={effectiveness:.1f}")

best_drug = np.argmax(predicted_effectiveness)
print(f"\\n🏆 Best candidate: Drug {best_drug + 1}")""", height=200, key="code5")

if st.button("▶️ Run Step 5", key="run5"):
    success, output, context = run_code(code5)
    if success:
        st.markdown(f'<div class="code-output">{output}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="error-output">Error: {output}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Conclusion
st.markdown("---")
st.markdown("""
## 🎉 Congratulations!

You just completed a **machine learning drug discovery pipeline** in 5 steps:

1. ✅ **Data Creation** - Generated drug properties
2. ✅ **Data Preparation** - Split training/testing
3. ✅ **Model Training** - Taught AI to predict effectiveness  
4. ✅ **Prediction** - Tested on new drugs
5. ✅ **Drug Design** - Found optimal candidate

### 🚀 Real Applications at Sanofi:
- **Molecular property prediction**
- **Drug-target interaction modeling**
- **Clinical trial optimization**
- **Side effect prediction**

*This simplified example demonstrates the core ML workflow used in pharmaceutical research.*
""")
