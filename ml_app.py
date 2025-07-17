import streamlit as st
import pandas as pd
import numpy as np

st.title("🤖 ML pour Business - Formation 40 min")
st.write("Analyse prédictive simple sans librairies complexes")

# Upload de données
uploaded_file = st.file_uploader("Chargez vos données CSV", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("📊 Aperçu des données:")
    st.dataframe(df.head())
    
    # Statistiques descriptives
    st.write("📈 Statistiques:")
    st.write(df.describe())
    
    # Sélection des colonnes
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    if categorical_columns:
        target_column = st.selectbox("Choisissez la colonne à analyser:", categorical_columns)
        
        if target_column and numeric_columns:
            feature_column = st.selectbox("Choisissez la variable explicative:", numeric_columns)
            
            # Analyse simple par segmentation
            if st.button("🔍 Analyser"):
                st.write(f"📊 Analyse: {feature_column} vs {target_column}")
                
                # Segmentation simple
                segments = df.groupby(target_column)[feature_column].agg(['mean', 'count', 'std']).round(2)
                st.write("Segments identifiés:")
                st.dataframe(segments)
                
                # Règle de décision simple
                target_values = df[target_column].unique()
                if len(target_values) == 2:
                    group_means = df.groupby(target_column)[feature_column].mean()
                    threshold = group_means.mean()
                    
                    st.write(f"🎯 Règle de décision simple:")
                    st.write(f"Si {feature_column} > {threshold:.2f} → Prédiction: {group_means.idxmax()}")
                    st.write(f"Si {feature_column} ≤ {threshold:.2f} → Prédiction: {group_means.idxmin()}")
                    
                    # Test de la règle
                    st.write("🧪 Tester la règle:")
                    test_value = st.number_input(f"Entrez une valeur pour {feature_column}:")
                    if test_value:
                        prediction = group_means.idxmax() if test_value > threshold else group_means.idxmin()
                        st.success(f"Prédiction: {prediction}")
                
                # Visualisation
                st.write("📈 Visualisation:")
                chart_data = df.groupby(target_column)[feature_column].mean()
                st.bar_chart(chart_data)

# Dataset d'exemple
st.sidebar.markdown("## 📋 Dataset d'exemple")
if st.sidebar.button("Générer données exemple"):
    # Créer données exemple
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'age': np.random.randint(20, 65, 100),
        'income': np.random.randint(30000, 100000, 100),
        'years_customer': np.random.randint(1, 10, 100)
    })
    
    # Logique pour target (simplifié)
    sample_data['premium_buyer'] = np.where(
        (sample_data['income'] > 60000) & (sample_data['years_customer'] > 3), 
        'yes', 'no'
    )
    
    st.sidebar.download_button(
        label="📥 Télécharger données exemple",
        data=sample_data.to_csv(index=False),
        file_name='customers_example.csv',
        mime='text/csv'
    )
