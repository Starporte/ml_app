# Je vous crée un template "Teachable Machine-like"
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

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
    if st.button("🚀 Créer mon modèle IA", type="primary"):
        
        # Spinners pour effet visuel
        with st.spinner("🤖 L'IA apprend vos données..."):
            
            # ML simplifié
            X = df[feature_cols]
            y = df[target_col]
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)
            
            score = model.score(X_test, y_test)
            
        # Résultats (style Teachable Machine)
        st.success("🎉 Votre modèle IA est prêt !")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🎯 Précision", f"{score:.1%}")
        with col2:
            st.metric("📊 Prédictions testées", len(X_test))
        
        # Importance des variables
        importances = model.feature_importances_
        feature_importance = pd.DataFrame({
            'Variable': feature_cols,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        st.markdown("### 🔍 Variables les plus importantes :")
        fig = px.bar(
            feature_importance, 
            x='Importance', 
            y='Variable',
            orientation='h'
        )
        st.plotly_chart(fig)
        
        # Section prédiction
        st.markdown("### 🔮 Testez votre modèle :")
        
        # Inputs dynamiques pour prédiction
        prediction_inputs = {}
        cols = st.columns(len(feature_cols))
        
        for i, col in enumerate(feature_cols):
            if df[col].dtype in ['int64', 'float64']:
                prediction_inputs[col] = cols[i].number_input(
                    f"{col}:",
                    value=float(df[col].mean())
                )
            else:
                prediction_inputs[col] = cols[i].selectbox(
                    f"{col}:",
                    df[col].unique()
                )
        
        # Prédiction en temps réel
        if st.button("🔮 Prédire !"):
            pred_df = pd.DataFrame([prediction_inputs])
            prediction = model.predict(pred_df)[0]
            probability = model.predict_proba(pred_df)[0].max()
            
            st.success(f"🎯 Prédiction : **{prediction}** (Confiance: {probability:.1%})")
