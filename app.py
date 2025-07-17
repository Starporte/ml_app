import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import json

# Configuration de la page
st.set_page_config(
    page_title="Mon Notebook Streamlit",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé pour ressembler à un notebook Jupyter
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 30px;
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 10px;
    }
    
    .cell-header {
        background-color: #f0f2f6;
        padding: 10px;
        border-left: 4px solid #1f77b4;
        margin: 20px 0 10px 0;
        font-weight: bold;
    }
    
    .code-cell {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #e1e5e9;
        margin: 10px 0;
    }
    
    .output-cell {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #d1d5db;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Titre principal
st.markdown('<h1 class="main-header">📊 Mon Notebook Streamlit</h1>', unsafe_allow_html=True)

# Sidebar pour la navigation
st.sidebar.title("🗂️ Navigation")
sections = [
    "📋 Introduction",
    "📊 Analyse des données",
    "📈 Visualisations",
    "🔍 Exploration interactive",
    "🧮 Calculs et modélisation",
    "📝 Conclusions"
]

selected_section = st.sidebar.selectbox("Choisir une section:", sections)

# Section 1: Introduction
if selected_section == "📋 Introduction":
    st.markdown('<div class="cell-header">Cellule 1: Introduction et imports</div>', unsafe_allow_html=True)
    
    st.markdown("**Objectif du notebook:**")
    st.write("Ce notebook Streamlit démontre l'analyse de données interactive avec différentes visualisations et calculs.")
    
    with st.expander("📦 Voir les imports"):
        st.code("""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import json
        """, language="python")
    
    st.markdown('<div class="cell-header">Cellule 2: Configuration initiale</div>', unsafe_allow_html=True)
    
    # Paramètres configurables
    st.subheader("⚙️ Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        sample_size = st.slider("Taille de l'échantillon", 100, 10000, 1000)
        random_seed = st.number_input("Seed aléatoire", value=42)
    
    with col2:
        chart_theme = st.selectbox("Thème des graphiques", ["plotly", "seaborn", "matplotlib"])
        show_code = st.checkbox("Afficher le code", value=True)

# Section 2: Analyse des données
elif selected_section == "📊 Analyse des données":
    st.markdown('<div class="cell-header">Cellule 3: Génération et chargement des données</div>', unsafe_allow_html=True)
    
    # Génération de données d'exemple
    np.random.seed(42)
    
    # Créer un dataset d'exemple
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
    data = {
        'date': dates,
        'ventes': np.random.normal(1000, 200, len(dates)) + np.sin(np.arange(len(dates)) * 2 * np.pi / 365) * 100,
        'temperature': np.random.normal(15, 10, len(dates)) + np.sin(np.arange(len(dates)) * 2 * np.pi / 365) * 10,
        'region': np.random.choice(['Nord', 'Sud', 'Est', 'Ouest'], len(dates)),
        'produit': np.random.choice(['A', 'B', 'C'], len(dates))
    }
    
    df = pd.DataFrame(data)
    df['ventes'] = df['ventes'].clip(lower=0)  # Pas de ventes négatives
    
    st.write("**Aperçu des données générées:**")
    st.dataframe(df.head(10))
    
    # Statistiques descriptives
    st.markdown('<div class="cell-header">Cellule 4: Statistiques descriptives</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Statistiques des ventes:**")
        st.dataframe(df['ventes'].describe())
    
    with col2:
        st.write("**Informations sur le dataset:**")
        st.write(f"- Nombre de lignes: {len(df):,}")
        st.write(f"- Nombre de colonnes: {len(df.columns)}")
        st.write(f"- Période: {df['date'].min()} à {df['date'].max()}")
        st.write(f"- Valeurs manquantes: {df.isnull().sum().sum()}")

# Section 3: Visualisations
elif selected_section == "📈 Visualisations":
    st.markdown('<div class="cell-header">Cellule 5: Visualisations temporelles</div>', unsafe_allow_html=True)
    
    # Régénération des données (simplifiée pour cohérence)
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
    df = pd.DataFrame({
        'date': dates,
        'ventes': np.random.normal(1000, 200, len(dates)) + np.sin(np.arange(len(dates)) * 2 * np.pi / 365) * 100,
        'temperature': np.random.normal(15, 10, len(dates)) + np.sin(np.arange(len(dates)) * 2 * np.pi / 365) * 10,
        'region': np.random.choice(['Nord', 'Sud', 'Est', 'Ouest'], len(dates)),
        'produit': np.random.choice(['A', 'B', 'C'], len(dates))
    })
    df['ventes'] = df['ventes'].clip(lower=0)
    
    # Graphique temporel interactif
    fig = px.line(df.sample(500), x='date', y='ventes', 
                  title="Évolution des ventes dans le temps",
                  labels={'ventes': 'Ventes (€)', 'date': 'Date'})
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('<div class="cell-header">Cellule 6: Analyse par région</div>', unsafe_allow_html=True)
    
    # Analyse par région
    ventes_region = df.groupby('region')['ventes'].agg(['mean', 'sum', 'count']).round(2)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_bar = px.bar(ventes_region.reset_index(), x='region', y='mean',
                        title="Ventes moyennes par région",
                        labels={'mean': 'Ventes moyennes (€)', 'region': 'Région'})
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        fig_pie = px.pie(ventes_region.reset_index(), values='sum', names='region',
                        title="Répartition du chiffre d'affaires")
        st.plotly_chart(fig_pie, use_container_width=True)

# Section 4: Exploration interactive
elif selected_section == "🔍 Exploration interactive":
    st.markdown('<div class="cell-header">Cellule 7: Filtres interactifs</div>', unsafe_allow_html=True)
    
    # Régénération des données
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
    df = pd.DataFrame({
        'date': dates,
        'ventes': np.random.normal(1000, 200, len(dates)) + np.sin(np.arange(len(dates)) * 2 * np.pi / 365) * 100,
        'temperature': np.random.normal(15, 10, len(dates)) + np.sin(np.arange(len(dates)) * 2 * np.pi / 365) * 10,
        'region': np.random.choice(['Nord', 'Sud', 'Est', 'Ouest'], len(dates)),
        'produit': np.random.choice(['A', 'B', 'C'], len(dates))
    })
    df['ventes'] = df['ventes'].clip(lower=0)
    
    # Filtres
    col1, col2, col3 = st.columns(3)
    
    with col1:
        regions_selectionnees = st.multiselect("Régions", df['region'].unique(), default=df['region'].unique())
    
    with col2:
        produits_selectionnes = st.multiselect("Produits", df['produit'].unique(), default=df['produit'].unique())
    
    with col3:
        date_range = st.date_input("Période", 
                                  value=(df['date'].min(), df['date'].max()),
                                  min_value=df['date'].min(),
                                  max_value=df['date'].max())
    
    # Filtrer les données
    df_filtered = df[
        (df['region'].isin(regions_selectionnees)) &
        (df['produit'].isin(produits_selectionnes)) &
        (df['date'] >= pd.Timestamp(date_range[0])) &
        (df['date'] <= pd.Timestamp(date_range[1]))
    ]
    
    st.markdown('<div class="cell-header">Cellule 8: Résultats filtrés</div>', unsafe_allow_html=True)
    
    # Métriques
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Nombre de records", f"{len(df_filtered):,}")
    
    with col2:
        st.metric("Ventes totales", f"{df_filtered['ventes'].sum():,.0f} €")
    
    with col3:
        st.metric("Ventes moyennes", f"{df_filtered['ventes'].mean():,.0f} €")
    
    with col4:
        st.metric("Température moyenne", f"{df_filtered['temperature'].mean():.1f}°C")
    
    # Graphique de corrélation
    if len(df_filtered) > 0:
        fig_scatter = px.scatter(df_filtered, x='temperature', y='ventes', 
                               color='region', size='ventes',
                               title="Corrélation Température vs Ventes",
                               labels={'temperature': 'Température (°C)', 'ventes': 'Ventes (€)'})
        st.plotly_chart(fig_scatter, use_container_width=True)

# Section 5: Calculs et modélisation
elif selected_section == "🧮 Calculs et modélisation":
    st.markdown('<div class="cell-header">Cellule 9: Calculs statistiques</div>', unsafe_allow_html=True)
    
    # Régénération des données
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
    df = pd.DataFrame({
        'date': dates,
        'ventes': np.random.normal(1000, 200, len(dates)) + np.sin(np.arange(len(dates)) * 2 * np.pi / 365) * 100,
        'temperature': np.random.normal(15, 10, len(dates)) + np.sin(np.arange(len(dates)) * 2 * np.pi / 365) * 10,
        'region': np.random.choice(['Nord', 'Sud', 'Est', 'Ouest'], len(dates)),
        'produit': np.random.choice(['A', 'B', 'C'], len(dates))
    })
    df['ventes'] = df['ventes'].clip(lower=0)
    
    # Calculs statistiques
    correlation = df['ventes'].corr(df['temperature'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Corrélation Ventes-Température:**")
        st.write(f"Coefficient de corrélation: {correlation:.3f}")
        
        if abs(correlation) > 0.5:
            st.success("Corrélation forte")
        elif abs(correlation) > 0.3:
            st.warning("Corrélation modérée")
        else:
            st.info("Corrélation faible")
    
    with col2:
        st.write("**Analyse de tendance:**")
        df_monthly = df.groupby(df['date'].dt.to_period('M'))['ventes'].mean()
        trend = "Croissante" if df_monthly.iloc[-1] > df_monthly.iloc[0] else "Décroissante"
        st.write(f"Tendance générale: {trend}")
        st.write(f"Variation: {((df_monthly.iloc[-1] / df_monthly.iloc[0]) - 1) * 100:.1f}%")
    
    st.markdown('<div class="cell-header">Cellule 10: Prédictions simples</div>', unsafe_allow_html=True)
    
    # Prédiction simple basée sur la moyenne mobile
    window = st.slider("Fenêtre de moyenne mobile", 7, 90, 30)
    
    df_sorted = df.sort_values('date')
    df_sorted['moving_avg'] = df_sorted['ventes'].rolling(window=window).mean()
    
    # Prédiction pour les prochains jours
    last_avg = df_sorted['moving_avg'].iloc[-1]
    
    st.write(f"**Prédiction basée sur moyenne mobile ({window} jours):**")
    st.write(f"Ventes prévues pour demain: {last_avg:.0f} €")
    
    # Graphique avec moyenne mobile
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(x=df_sorted['date'], y=df_sorted['ventes'], 
                                 mode='lines', name='Ventes réelles', opacity=0.7))
    fig_pred.add_trace(go.Scatter(x=df_sorted['date'], y=df_sorted['moving_avg'], 
                                 mode='lines', name=f'Moyenne mobile ({window}j)', 
                                 line=dict(width=3)))
    
    fig_pred.update_layout(title="Ventes avec moyenne mobile",
                          xaxis_title="Date",
                          yaxis_title="Ventes (€)")
    
    st.plotly_chart(fig_pred, use_container_width=True)

# Section 6: Conclusions
elif selected_section == "📝 Conclusions":
    st.markdown('<div class="cell-header">Cellule 11: Résumé de l\'analyse</div>', unsafe_allow_html=True)
    
    st.write("**Principales conclusions de l'analyse:**")
    
    conclusions = [
        "📊 Les données montrent une variabilité naturelle des ventes avec une composante saisonnière",
        "🌡️ La corrélation entre température et ventes nécessite une analyse plus approfondie",
        "🗺️ Les performances varient significativement selon les régions",
        "📈 La tendance générale peut être modélisée avec des moyennes mobiles",
        "🔍 L'analyse interactive permet une exploration flexible des données"
    ]
    
    for conclusion in conclusions:
        st.write(f"- {conclusion}")
    
    st.markdown('<div class="cell-header">Cellule 12: Prochaines étapes</div>', unsafe_allow_html=True)
    
    st.write("**Recommandations pour approfondir l'analyse:**")
    
    recommendations = [
        "Collecter plus de données historiques pour améliorer les modèles",
        "Implémenter des modèles de machine learning plus sophistiqués",
        "Analyser les facteurs externes (événements, promotions, etc.)",
        "Développer un dashboard en temps réel",
        "Automatiser les rapports et alertes"
    ]
    
    for i, rec in enumerate(recommendations, 1):
        st.write(f"{i}. {rec}")
    
    # Bouton pour télécharger les données
    if st.button("📥 Télécharger les données analysées"):
        # Régénération pour cohérence
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
        df = pd.DataFrame({
            'date': dates,
            'ventes': np.random.normal(1000, 200, len(dates)) + np.sin(np.arange(len(dates)) * 2 * np.pi / 365) * 100,
            'temperature': np.random.normal(15, 10, len(dates)) + np.sin(np.arange(len(dates)) * 2 * np.pi / 365) * 10,
            'region': np.random.choice(['Nord', 'Sud', 'Est', 'Ouest'], len(dates)),
            'produit': np.random.choice(['A', 'B', 'C'], len(dates))
        })
        df['ventes'] = df['ventes'].clip(lower=0)
        
        csv = df.to_csv(index=False)
        st.download_button(
            label="💾 Télécharger CSV",
            data=csv,
            file_name="analyse_ventes.csv",
            mime="text/csv"
        )

# Footer
st.markdown("---")
st.markdown("*Notebook créé avec Streamlit - Mise à jour automatique des calculs*")
