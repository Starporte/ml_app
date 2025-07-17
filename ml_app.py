# Créez ce fichier : ml_app.py
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title("🤖 ML pour Business - Formation 40 min")

# Upload de données
uploaded_file = st.file_uploader("Chargez vos données CSV", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Aperçu des données:")
    st.dataframe(df.head())
    
    # Sélection target
    target_column = st.selectbox("Choisissez la colonne à prédire:", df.columns)
    
    if st.button("🚀 Entraîner le modèle"):
        # ML automatique
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        
        # Encodage automatique
        X_encoded = pd.get_dummies(X)
        
        # Split et entraînement
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2)
        
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        
        # Prédictions
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        st.success(f"Modèle entraîné ! Précision: {accuracy:.2%}")
        
        # Importance des variables
        feature_importance = pd.DataFrame({
            'feature': X_encoded.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        st.bar_chart(feature_importance.set_index('feature')['importance'])
