import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go

# Configuration de la page
st.set_page_config(
    page_title="🏥 Prédiction d'Adhérence Médicamenteuse",
    page_icon="💊",
    layout="wide"
)

# CSS pour améliorer l'apparence
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        color: white;
        padding: 20px;
        text-align: center;
        border-radius: 10px;
        margin-bottom: 30px;
    }
    
    .step-box {
        background-color: #f8f9fa;
        border-left: 4px solid #1f77b4;
        padding: 15px;
        margin: 15px 0;
        border-radius: 5px;
    }
    
    .metric-card {
        background-color: #e8f4fd;
        border: 1px solid #1f77b4;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    
    .success-box {
        background-color: #d4edda;
        border: 1px solid #28a745;
        padding: 10px;
        border-radius: 5px;
        color: #155724;
    }
</style>
""", unsafe_allow_html=True)

# En-tête principal
st.markdown("""
<div class="main-header">
    <h1>🏥 Tutoriel : Prédiction d'Adhérence Médicamenteuse</h1>
    <h3>Apprendre le Machine Learning avec un cas concret</h3>
</div>
""", unsafe_allow_html=True)

# Introduction
st.markdown("""
## 🎯 Objectif de ce tutoriel

Ce tutoriel vous apprend à **prédire si un patient va bien suivre son traitement** en utilisant le Machine Learning. 
Nous allons utiliser des données patient simples pour construire un modèle intelligent.

**Ce que vous allez apprendre :**
- Comment préparer des données médicales
- Comment entraîner un modèle de Machine Learning  
- Comment évaluer les performances du modèle
- Comment faire des prédictions sur de nouveaux patients

---
""")

# Initialisation des données en session state
if 'data_generated' not in st.session_state:
    st.session_state.data_generated = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# ÉTAPE 1 : Génération des données
st.markdown('<div class="step-box">', unsafe_allow_html=True)
st.markdown("### 📊 Étape 1 : Création des données patients")
st.markdown("""
**Explication :** Nous allons créer un jeu de données fictif représentant des patients avec :
- Âge du patient
- Coût du traitement annuel  
- Nombre d'effets secondaires
- Si le patient adhère au traitement (OUI/NON)
""")

if st.button("🎲 Générer les données patients", key="generate_data"):
    with st.spinner("Génération des données..."):
        # Génération des données
        np.random.seed(42)
        n_patients = 1000
        
        # Variables explicatives
        age = np.random.normal(65, 15, n_patients)
        age = np.clip(age, 18, 90)  # Âge entre 18 et 90 ans
        
        cout_annuel = np.random.exponential(2000, n_patients)
        cout_annuel = np.clip(cout_annuel, 500, 8000)  # Coût entre 500€ et 8000€
        
        effets_secondaires = np.random.poisson(2, n_patients)
        effets_secondaires = np.clip(effets_secondaires, 0, 8)  # 0 à 8 effets
        
        # Variable cible (adhérence) - logique métier
        # Plus le patient est jeune, moins d'effets secondaires, coût faible = meilleure adhérence
        score_adherence = (
            -0.02 * age +  # Plus jeune = mieux
            -0.0003 * cout_annuel +  # Moins cher = mieux  
            -0.5 * effets_secondaires +  # Moins d'effets = mieux
            np.random.normal(0, 1, n_patients)  # Bruit aléatoire
        )
        
        adherence = (score_adherence > np.median(score_adherence)).astype(int)
        
        # Création du DataFrame
        data = pd.DataFrame({
            'age': age.round(0).astype(int),
            'cout_annuel': cout_annuel.round(0).astype(int),
            'effets_secondaires': effets_secondaires,
            'adherence': adherence
        })
        
        st.session_state.data = data
        st.session_state.data_generated = True
    
    st.success("✅ Données générées avec succès !")

if st.session_state.data_generated:
    data = st.session_state.data
    
    # Affichage des données
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Aperçu des données :**")
        st.dataframe(data.head(10))
        
    with col2:
        st.markdown("**Statistiques :**")
        adherent_count = data['adherence'].sum()
        total_count = len(data)
        
        st.metric("Total patients", total_count)
        st.metric("Patients adhérents", f"{adherent_count} ({adherent_count/total_count*100:.1f}%)")
        st.metric("Patients non-adhérents", f"{total_count-adherent_count} ({(total_count-adherent_count)/total_count*100:.1f}%)")

st.markdown('</div>', unsafe_allow_html=True)

# ÉTAPE 2 : Exploration des données
if st.session_state.data_generated:
    st.markdown('<div class="step-box">', unsafe_allow_html=True)
    st.markdown("### 🔍 Étape 2 : Explorer les données")
    st.markdown("""
    **Explication :** Avant d'entraîner notre modèle, nous devons comprendre nos données.
    Regardons comment l'âge, le coût et les effets secondaires influencent l'adhérence.
    """)
    
    if st.button("📈 Créer les graphiques d'exploration", key="explore_data"):
        data = st.session_state.data
        
        # Graphiques
        col1, col2 = st.columns(2)
        
        with col1:
            # Graphique âge vs adhérence
            fig1 = px.box(data, x='adherence', y='age', 
                         labels={'adherence': 'Adhérence (0=Non, 1=Oui)', 'age': 'Âge'},
                         title="Distribution de l'âge selon l'adhérence")
            st.plotly_chart(fig1, use_container_width=True)
            
        with col2:
            # Graphique coût vs adhérence  
            fig2 = px.box(data, x='adherence', y='cout_annuel',
                         labels={'adherence': 'Adhérence (0=Non, 1=Oui)', 'cout_annuel': 'Coût annuel (€)'},
                         title="Distribution du coût selon l'adhérence")
            st.plotly_chart(fig2, use_container_width=True)
            
        # Graphique effets secondaires
        fig3 = px.box(data, x='adherence', y='effets_secondaires',
                     labels={'adherence': 'Adhérence (0=Non, 1=Oui)', 'effets_secondaires': 'Nombre d\'effets secondaires'},
                     title="Distribution des effets secondaires selon l'adhérence")
        st.plotly_chart(fig3, use_container_width=True)
        
        st.markdown("""
        **💡 Observations :**
        - Les patients plus jeunes ont tendance à mieux adhérer
        - Un coût élevé réduit l'adhérence  
        - Plus d'effets secondaires = moins d'adhérence
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ÉTAPE 3 : Entraînement du modèle
if st.session_state.data_generated:
    st.markdown('<div class="step-box">', unsafe_allow_html=True)
    st.markdown("### 🤖 Étape 3 : Entraîner le modèle d'IA")
    st.markdown("""
    **Explication :** Nous allons utiliser un algorithme appelé **Random Forest** (Forêt Aléatoire).
    
    **Qu'est-ce que Random Forest ?**
    - Imagine 100 médecins qui donnent chacun leur opinion
    - Chaque médecin regarde les données différemment  
    - Le diagnostic final = vote majoritaire des 100 médecins
    - Plus fiable qu'un seul médecin !
    """)
    
    # Choix de l'algorithme
    algo_choice = st.selectbox(
        "Choisissez l'algorithme :",
        ["Random Forest (Recommandé)", "Régression Logistique (Simple)"]
    )
    
    if st.button("🚀 Entraîner le modèle", key="train_model"):
        data = st.session_state.data
        
        with st.spinner("Entraînement en cours..."):
            # Préparation des données
            X = data[['age', 'cout_annuel', 'effets_secondaires']]
            y = data['adherence']
            
            # Division train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
            
            # Choix du modèle
            if "Random Forest" in algo_choice:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model_name = "Random Forest"
            else:
                model = LogisticRegression(random_state=42)
                model_name = "Régression Logistique"
            
            # Entraînement
            model.fit(X_train, y_train)
            
            # Prédictions
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Sauvegarde
            st.session_state.model = model
            st.session_state.model_name = model_name
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.y_pred = y_pred
            st.session_state.accuracy = accuracy
            st.session_state.model_trained = True
        
        st.success(f"✅ Modèle {model_name} entraîné avec succès !")
        st.metric("Précision du modèle", f"{accuracy:.1%}")
        
        # Importance des variables (pour Random Forest)
        if "Random Forest" in algo_choice:
            importance = model.feature_importances_
            feature_names = ['Âge', 'Coût annuel', 'Effets secondaires']
            
            fig_imp = px.bar(
                x=feature_names, y=importance,
                title="Importance des variables dans la prédiction",
                labels={'x': 'Variables', 'y': 'Importance'}
            )
            st.plotly_chart(fig_imp, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ÉTAPE 4 : Évaluation du modèle
if st.session_state.get('model_trained', False):
    st.markdown('<div class="step-box">', unsafe_allow_html=True)
    st.markdown("### 📊 Étape 4 : Évaluer les performances")
    st.markdown("""
    **Explication :** Nous devons vérifier si notre modèle fait de bonnes prédictions.
    Nous utilisons une **matrice de confusion** pour voir les erreurs.
    """)
    
    if st.button("📈 Analyser les performances", key="evaluate_model"):
        y_test = st.session_state.y_test
        y_pred = st.session_state.y_pred
        accuracy = st.session_state.accuracy
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Matrice de confusion
            cm = confusion_matrix(y_test, y_pred)
            
            fig_cm = px.imshow(
                cm, 
                labels=dict(x="Prédiction", y="Réalité", color="Nombre"),
                x=['Non-adhérent', 'Adhérent'],
                y=['Non-adhérent', 'Adhérent'],
                title="Matrice de confusion"
            )
            fig_cm.update_layout(width=400, height=400)
            st.plotly_chart(fig_cm, use_container_width=True)
            
        with col2:
            st.markdown("**📊 Résultats :**")
            st.metric("Précision globale", f"{accuracy:.1%}")
            
            # Calcul des métriques détaillées
            tn, fp, fn, tp = cm.ravel()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            st.metric("Précision (patients adhérents)", f"{precision:.1%}")
            st.metric("Rappel (patients adhérents)", f"{recall:.1%}")
            
            st.markdown("""
            **💡 Interprétation :**
            - **Précision** : Sur 100 patients prédits adhérents, combien le sont vraiment ?
            - **Rappel** : Sur 100 patients vraiment adhérents, combien sont détectés ?
            """)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ÉTAPE 5 : Faire des prédictions
if st.session_state.get('model_trained', False):
    st.markdown('<div class="step-box">', unsafe_allow_html=True)
    st.markdown("### 🎯 Étape 5 : Prédire pour un nouveau patient")
    st.markdown("""
    **Explication :** Maintenant, utilisons notre modèle entraîné pour prédire 
    si un nouveau patient va adhérer à son traitement.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Saisissez les informations du patient :**")
        
        patient_age = st.slider("Âge du patient", 18, 90, 55)
        patient_cout = st.slider("Coût annuel du traitement (€)", 500, 8000, 2000)
        patient_effets = st.slider("Nombre d'effets secondaires", 0, 8, 2)
        
        if st.button("🔮 Prédire l'adhérence", key="predict_patient"):
            model = st.session_state.model
            
            # Prédiction
            patient_data = np.array([[patient_age, patient_cout, patient_effets]])
            prediction = model.predict(patient_data)[0]
            proba = model.predict_proba(patient_data)[0]
            
            # Affichage du résultat
            if prediction == 1:
                st.success(f"✅ **Patient ADHÉRENT** (probabilité: {proba[1]:.1%})")
                st.balloons()
            else:
                st.error(f"❌ **Patient NON-ADHÉRENT** (probabilité: {proba[0]:.1%})")
                
            # Recommandations
            st.markdown("**💡 Recommandations :**")
            if patient_cout > 4000:
                st.warning("💰 Coût élevé : Considérer une aide financière")
            if patient_effets > 4:
                st.warning("😷 Beaucoup d'effets secondaires : Ajuster le traitement")
            if patient_age > 75:
                st.info("👴 Patient âgé : Prévoir un suivi renforcé")
    
    with col2:
        # Profil du patient
        st.markdown("**👤 Profil du patient :**")
        
        patient_data_display = pd.DataFrame({
            'Caractéristique': ['Âge', 'Coût annuel', 'Effets secondaires'],
            'Valeur': [f"{patient_age} ans", f"{patient_cout} €", patient_effets]
        })
        st.table(patient_data_display)
        
        # Comparaison avec la population
        if 'data' in st.session_state:
            data = st.session_state.data
            st.markdown("**📊 Comparaison avec la population :**")
            
            age_percentile = (data['age'] <= patient_age).mean() * 100
            cout_percentile = (data['cout_annuel'] <= patient_cout).mean() * 100
            effets_percentile = (data['effets_secondaires'] <= patient_effets).mean() * 100
            
            st.write(f"• Âge : {age_percentile:.0f}e percentile")
            st.write(f"• Coût : {cout_percentile:.0f}e percentile") 
            st.write(f"• Effets : {effets_percentile:.0f}e percentile")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Conclusion
st.markdown("---")
st.markdown("""
## 🎉 Félicitations !

Vous avez créé votre premier modèle de Machine Learning pour prédire l'adhérence médicamenteuse !

### 📚 Ce que vous avez appris :
1. **Préparer des données** : Créer un jeu de données représentatif
2. **Explorer les données** : Comprendre les relations entre variables
3. **Entraîner un modèle** : Utiliser Random Forest pour apprendre
4. **Évaluer les performances** : Mesurer la qualité des prédictions
5. **Faire des prédictions** : Utiliser le modèle sur de nouveaux cas

### 🏥 Applications réelles :
- **Optimisation des traitements** : Identifier les patients à risque
- **Personnalisation** : Adapter les protocoles selon le profil patient
- **Prévention** : Intervenir avant l'arrêt du traitement
- **Allocation des ressources** : Prioriser le suivi médical

### 🚀 Pour aller plus loin :
- Ajouter plus de variables (historique, comorbidités, socio-économiques)
- Tester d'autres algorithmes (XGBoost, Neural Networks)
- Valider sur de vraies données médicales
- Intégrer dans un système d'aide à la décision clinique
""")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    💊 Tutoriel Machine Learning - Prédiction d'Adhérence Médicamenteuse<br>
    🔬 Pour l'éducation et la recherche médicale
</div>
""", unsafe_allow_html=True)
