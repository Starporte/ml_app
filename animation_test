import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import time

st.set_page_config(page_title="🎬 Animation Tests", layout="wide")

st.title("🎬 Decision Tree Animation Tests")
st.markdown("Testing 3 different animation approaches")

# CSS Animations
st.markdown("""
<style>
    .tree-node {
        width: 100px;
        height: 100px;
        background: linear-gradient(45deg, #4CAF50, #45a049);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
        color: white;
        margin: 10px auto;
        animation: grow 2s ease-in-out;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    
    .tree-branch {
        width: 60px;
        height: 60px;
        background: linear-gradient(45deg, #FF9800, #F57C00);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 16px;
        color: white;
        margin: 5px auto;
        animation: slideIn 1s ease-in-out;
        animation-delay: 0.5s;
        opacity: 0;
        animation-fill-mode: forwards;
    }
    
    .tree-leaf {
        width: 40px;
        height: 40px;
        background: linear-gradient(45deg, #2196F3, #1976D2);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 12px;
        color: white;
        margin: 5px auto;
        animation: fadeIn 1s ease-in-out;
        animation-delay: 1.5s;
        opacity: 0;
        animation-fill-mode: forwards;
    }
    
    @keyframes grow {
        0% { 
            transform: scale(0) rotate(0deg); 
            opacity: 0;
        }
        50% {
            transform: scale(1.2) rotate(180deg);
        }
        100% { 
            transform: scale(1) rotate(360deg); 
            opacity: 1;
        }
    }
    
    @keyframes slideIn {
        0% { 
            transform: translateX(-100px); 
            opacity: 0;
        }
        100% { 
            transform: translateX(0); 
            opacity: 1;
        }
    }
    
    @keyframes fadeIn {
        0% { 
            opacity: 0; 
            transform: translateY(20px);
        }
        100% { 
            opacity: 1; 
            transform: translateY(0);
        }
    }
    
    .decision-path {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        margin: 10px 0;
        border-radius: 10px;
        animation: typewriter 3s steps(40) forwards;
        overflow: hidden;
        white-space: nowrap;
        border-right: 3px solid white;
        width: 0;
    }
    
    @keyframes typewriter {
        to { 
            width: 100%; 
            border-right: none;
        }
    }
    
    .voting-box {
        background: #f0f8ff;
        border: 2px solid #4CAF50;
        border-radius: 15px;
        padding: 20px;
        margin: 10px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(76, 175, 80, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(76, 175, 80, 0); }
        100% { box-shadow: 0 0 0 0 rgba(76, 175, 80, 0); }
    }
    
    .forest-container {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 10px;
        padding: 20px;
    }
    
    .mini-tree {
        width: 30px;
        height: 30px;
        background: #4CAF50;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 10px;
        animation: treeGrow var(--delay, 0s) 0.5s ease-in-out forwards;
        opacity: 0;
        transform: scale(0);
    }
    
    @keyframes treeGrow {
        to { opacity: 1; transform: scale(1); }
    }
</style>
""", unsafe_allow_html=True)

# Option 1: Manim-style with CSS (Simulation)
st.header("🎨 Option 1: CSS Animations (Manim-style)")
st.markdown("**Simulating Manim-like animations with pure CSS**")

if st.button("🚀 Start Decision Tree Animation", key="css_anim"):
    st.markdown("### 🌳 Building Decision Tree Step by Step")
    
    # Root node
    st.markdown('<div class="tree-node">🎯 Root<br>Age?</div>', unsafe_allow_html=True)
    
    time.sleep(1)
    
    # Branch nodes
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="tree-branch">< 65<br>💊</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="tree-branch">> 65<br>💰</div>', unsafe_allow_html=True)
    
    time.sleep(1)
    
    # Leaf nodes
    cols = st.columns(4)
    with cols[0]:
        st.markdown('<div class="tree-leaf">✅<br>YES</div>', unsafe_allow_html=True)
    with cols[1]:
        st.markdown('<div class="tree-leaf">❌<br>NO</div>', unsafe_allow_html=True)
    with cols[2]:
        st.markdown('<div class="tree-leaf">✅<br>YES</div>', unsafe_allow_html=True)
    with cols[3]:
        st.markdown('<div class="tree-leaf">❌<br>NO</div>', unsafe_allow_html=True)
    
    # Decision path animation
    st.markdown('<div class="decision-path">🔍 Decision Path: Age=70 → >65 → Check Cost → $1500 → ✅ ADHERENT</div>', unsafe_allow_html=True)

# Option 2: Plotly Animations
st.header("📊 Option 2: Plotly Interactive Animations")

if st.button("📈 Create Plotly Tree Animation", key="plotly_anim"):
    # Simulate tree building with animated scatter plot
    frames_data = []
    
    # Frame 1: Root node
    frames_data.append({
        'x': [0], 'y': [0], 
        'text': ['Root: Age?'], 
        'size': [50],
        'color': ['red'],
        'frame': 0
    })
    
    # Frame 2: Add branches
    frames_data.append({
        'x': [0, -2, 2], 'y': [0, -1, -1],
        'text': ['Root: Age?', '< 65 years', '> 65 years'],
        'size': [50, 40, 40],
        'color': ['red', 'orange', 'orange'],
        'frame': 1
    })
    
    # Frame 3: Add leaves
    frames_data.append({
        'x': [0, -2, 2, -3, -1, 1, 3], 
        'y': [0, -1, -1, -2, -2, -2, -2],
        'text': ['Root: Age?', '< 65 years', '> 65 years', 
                'ADHERENT', 'NON-ADHERENT', 'ADHERENT', 'NON-ADHERENT'],
        'size': [50, 40, 40, 30, 30, 30, 30],
        'color': ['red', 'orange', 'orange', 'green', 'blue', 'green', 'blue'],
        'frame': 2
    })
    
    # Create animated figure
    df_frames = pd.DataFrame([
        {'x': x, 'y': y, 'text': text, 'size': size, 'color': color, 'frame': frame}
        for frame_data in frames_data
        for x, y, text, size, color in zip(
            frame_data['x'], frame_data['y'], frame_data['text'], 
            frame_data['size'], frame_data['color']
        )
        for frame in [frame_data['frame']]
    ])
    
    fig = px.scatter(df_frames, x='x', y='y', text='text', size='size',
                     color='color', animation_frame='frame',
                     title='Decision Tree Construction',
                     range_x=[-4, 4], range_y=[-3, 1])
    
    fig.update_traces(textposition="middle center", textfont_size=12)
    fig.update_layout(showlegend=False, height=500)
    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 2000
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Random Forest voting animation
    st.markdown("### 🗳️ Random Forest Voting Process")
    
    # Create voting animation data
    trees_data = []
    for i in range(0, 101, 10):  # 0, 10, 20, ..., 100
        adherent_votes = int(i * 0.65)  # 65% vote adherent
        non_adherent_votes = i - adherent_votes
        
        trees_data.append({
            'Trees_Processed': i,
            'Adherent_Votes': adherent_votes,
            'Non_Adherent_Votes': non_adherent_votes,
            'Confidence': adherent_votes / max(i, 1) if i > 0 else 0
        })
    
    df_voting = pd.DataFrame(trees_data)
    
    fig_voting = px.bar(df_voting, x='Trees_Processed', 
                       y=['Adherent_Votes', 'Non_Adherent_Votes'],
                       title='Random Forest Voting Progress',
                       labels={'value': 'Votes', 'variable': 'Prediction'},
                       color_discrete_map={
                           'Adherent_Votes': 'green',
                           'Non_Adherent_Votes': 'red'
                       })
    
    st.plotly_chart(fig_voting, use_container_width=True)

# Option 3: Advanced CSS with Interactive Elements
st.header("🎭 Option 3: Advanced CSS + Interactive")

if st.button("🎪 Launch Interactive Forest", key="advanced_css"):
    st.markdown("### 🌲 Random Forest Simulation")
    
    # Dynamic forest creation
    st.markdown('<div class="voting-box">🗳️ <strong>100 Trees Voting in Real-Time</strong></div>', unsafe_allow_html=True)
    
    # Create mini forest with staggered animations
    forest_html = '<div class="forest-container">'
    for i in range(100):
        delay = i * 0.05  # Stagger animation
        forest_html += f'<div class="mini-tree" style="--delay: {delay}s">🌳</div>'
    forest_html += '</div>'
    
    st.markdown(forest_html, unsafe_allow_html=True)
    
    # Voting progress
    progress_placeholder = st.empty()
    votes_placeholder = st.empty()
    
    adherent_votes = 0
    non_adherent_votes = 0
    
    for i in range(1, 101):
        time.sleep(0.1)  # Simulate processing time
        
        # Random voting (65% adherent probability)
        if np.random.random() < 0.65:
            adherent_votes += 1
        else:
            non_adherent_votes += 1
        
        # Update progress
        progress_placeholder.progress(i / 100)
        
        # Update vote count
        confidence = adherent_votes / i
        votes_placeholder.markdown(f"""
        **Tree {i}/100 voted:**
        - ✅ Adherent: {adherent_votes} ({adherent_votes/i:.1%})
        - ❌ Non-Adherent: {non_adherent_votes} ({non_adherent_votes/i:.1%})
        
        **Current Prediction: {'✅ ADHERENT' if confidence > 0.5 else '❌ NON-ADHERENT'}** 
        (Confidence: {max(confidence, 1-confidence):.1%})
        """)
    
    # Final result
    if adherent_votes > non_adherent_votes:
        st.success(f"🎉 Final Decision: **ADHERENT** ({adherent_votes}/100 trees voted)")
        st.balloons()
    else:
        st.error(f"❌ Final Decision: **NON-ADHERENT** ({non_adherent_votes}/100 trees voted)")

# Comparison
st.header("🔍 Comparison")
st.markdown("""
| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| **CSS Animation** | 🎨 Beautiful, smooth, no dependencies | ⚠️ Limited interactivity | Static explanations |
| **Plotly Animation** | 📊 Data-driven, interactive, built-in Streamlit | 🐌 Can be slow with lots of data | Data visualizations |
| **Advanced CSS + Python** | 🎭 Highly customizable, real-time updates | 🔧 More complex to implement | Interactive demos |
""")

st.markdown("---")
st.markdown("**🚀 Recommendation:** Combine all three for maximum impact!")
st.markdown("- CSS for smooth UI animations")  
st.markdown("- Plotly for data-driven charts")
st.markdown("- Advanced CSS+Python for interactive simulations")
