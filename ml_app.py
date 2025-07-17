import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

st.title("🎯 AutoML pour Business - Comme Teachable Machine")

# Interface exactement comme Teachable Machine
uploaded_file = st.file_uploader(
    "📁 Upload your data", 
    type=['csv'],
    help="Glissez-déposez votre fichier CSV ici"
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Données chargées avec succès !")
    
    # Aperçu données
    with st.expander("👀 Aperçu des données"):
        st.dataframe(df.head())
    
    # Sélection target (comme Teachable Machine)
    st.markdown("### 🎯 Que voulez-vous prédire ?")
    target_col = st.selectbox(
        "Choisissez la colonne cible :",
        df.columns,
        help="C'est ce que votre IA va apprendre à prédire"
    )
    
    # Sélection features
    st.markdown("### 📊 En utilisant ces informations :")
    feature_cols = st.multiselect(
        "Sélectionnez les variables explicatives :",
        [col for col in df.columns if col != target_col],
        default=[col for col in df.columns if col != target_col][:3]
    )
    
    # Bouton magique (comme Teachable Machine)
    if st.button("🚀 Créer mon modèle IA", type="primary") and feature_cols:
        
        with st.spinner("🤖 L'IA apprend vos données..."):
            try:
                # ML simplifié
                X = df[feature_cols].select_dtypes(include=[np.number])
                y = df[target_col]
                
                # Nettoyage données
                X = X.dropna()
                y = y[X.index]
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                model = RandomForestClassifier(random_state=42)
                model.fit(X_train, y_train)
                
                score = model.score(X_test, y_test)
                
                # Résultats
                st.success("🎉 Votre modèle IA est prêt !")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("🎯 Précision", f"{score:.1%}")
                with col2:
                    st.metric("📊 Prédictions testées", len(X_test))
                
                # Importance des variables (simple)
                importances = model.feature_importances_
                importance_df = pd.DataFrame({
                    'Variable': X.columns,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                st.markdown("### 🔍 Variables les plus importantes :")
                st.dataframe(importance_df)
                
                # Test de prédiction
                st.markdown("### 🔮 Testez votre modèle :")
                
                prediction_inputs = {}
                for col in X.columns:
                    prediction_inputs[col] = st.number_input(
                        f"{col}:",
                        value=float(X[col].mean())
                    )
                
                if st.button("🔮 Prédire !"):
                    pred_df = pd.DataFrame([prediction_inputs])
                    prediction = model.predict(pred_df)[0]
                    st.success(f"🎯 Prédiction : **{prediction}**")
                    
            except Exception as e:
                st.error(f"Erreur : {str(e)}")
                st.info("Vérifiez que vos données contiennent des colonnes numériques.")

# Exemple de données
st.markdown("---")
if st.button("📊 Générer des données d'exemple"):
    np.random.seed(42)
    n_samples = 1000
    
    example_data = pd.DataFrame({
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.randint(20000, 150000, n_samples),
        'years_customer': np.random.randint(0, 20, n_samples),
        'nb_products': np.random.randint(1, 5, n_samples),
        'satisfaction': np.random.randint(1, 6, n_samples),
        'will_buy_premium': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    })
    
    st.dataframe(example_data.head())
    
    csv = example_data.to_csv(index=False)
    st.download_button(
        label="💾 Télécharger les données d'exemple",
        data=csv,
        file_name='donnees_exemple.csv',
        mime='text/csv'
    )
