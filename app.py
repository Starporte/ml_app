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
import sys
from io import StringIO
import traceback

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
    
    .code-editor {
        background-color: #1e1e1e;
        color: #d4d4d4;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #333;
        margin: 10px 0;
        font-family: 'Courier New', monospace;
    }
    
    .output-cell {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #d1d5db;
        margin: 10px 0;
    }
    
    .success-output {
        background-color: #f0f9ff;
        border-left: 4px solid #22c55e;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
    }
    
    .error-output {
        background-color: #fef2f2;
        border-left: 4px solid #ef4444;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
        color: #dc2626;
    }
</style>
""", unsafe_allow_html=True)

# Titre principal
st.markdown('<h1 class="main-header">📊 Mon Notebook Streamlit</h1>', unsafe_allow_html=True)

# Sidebar pour la navigation
st.sidebar.title("🗂️ Navigation")
sections = [
    "📋 Introduction",
    "💻 Code interactif",
    "📊 Analyse des données",
    "🔬 Expérimentations",
    "📈 Visualisations",
    "🔍 Exploration interactive",
    "🧮 Calculs et modélisation",
    "📝 Conclusions"
]

selected_section = st.sidebar.selectbox("Choisir une section:", sections)

# Fonction pour exécuter du code Python de manière sécurisée
def execute_python_code(code, global_vars=None):
    """Exécute du code Python et retourne le résultat"""
    if global_vars is None:
        global_vars = {
            'pd': pd, 'np': np, 'plt': plt, 'sns': sns, 
            'px': px, 'go': go, 'st': st, 'datetime': datetime,
            'requests': requests, 'json': json
        }
    
    # Capturer la sortie
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()
    
    try:
        # Exécuter le code
        exec(code, global_vars)
        output = captured_output.getvalue()
        return True, output, global_vars
    except Exception as e:
        error_msg = f"Erreur: {str(e)}\n{traceback.format_exc()}"
        return False, error_msg, global_vars
    finally:
        sys.stdout = old_stdout

# Initialiser les variables globales pour le code
if 'code_globals' not in st.session_state:
    st.session_state.code_globals = {
        'pd': pd, 'np': np, 'plt': plt, 'sns': sns, 
        'px': px, 'go': go, 'st': st, 'datetime': datetime,
        'requests': requests, 'json': json
    }

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

# Section 2: Code interactif
elif selected_section == "💻 Code interactif":
    st.markdown('<div class="cell-header">Cellule Code 1: Premiers pas avec Python</div>', unsafe_allow_html=True)
    
    st.write("**Exercice 1: Calculatrice Python**")
    st.write("Écrivez du code Python pour effectuer des calculs simples:")
    
    code_1 = st.text_area(
        "Code Python:",
        value="""# Calculatrice simple
a = 10
b = 5

print(f"Addition: {a} + {b} = {a + b}")
print(f"Multiplication: {a} * {b} = {a * b}")
print(f"Division: {a} / {b} = {a / b}")

# Calculer la moyenne d'une liste
nombres = [12, 45, 23, 67, 34, 89, 56]
moyenne = sum(nombres) / len(nombres)
print(f"Moyenne: {moyenne:.2f}")""",
        height=200,
        key="code_1"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("▶️ Exécuter", key="run_1"):
            success, output, globals_updated = execute_python_code(code_1, st.session_state.code_globals)
            st.session_state.code_globals = globals_updated
            
            if success:
                st.markdown('<div class="success-output">', unsafe_allow_html=True)
                st.code(output, language="")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="error-output">', unsafe_allow_html=True)
                st.code(output, language="")
                st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="cell-header">Cellule Code 2: Manipulation de données</div>', unsafe_allow_html=True)
    
    st.write("**Exercice 2: Créer et manipuler un DataFrame**")
    
    code_2 = st.text_area(
        "Code Python:",
        value="""# Créer un DataFrame
import pandas as pd
import numpy as np

# Données d'exemple
data = {
    'nom': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'age': [25, 30, 35, 28, 32],
    'ville': ['Paris', 'Lyon', 'Marseille', 'Toulouse', 'Nice'],
    'salaire': [50000, 60000, 55000, 52000, 58000]
}

df = pd.DataFrame(data)
print("DataFrame créé:")
print(df)
print(f"\\nNombre de lignes: {len(df)}")
print(f"Âge moyen: {df['age'].mean():.1f} ans")
print(f"Salaire médian: {df['salaire'].median():,} €")""",
        height=250,
        key="code_2"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("▶️ Exécuter", key="run_2"):
            success, output, globals_updated = execute_python_code(code_2, st.session_state.code_globals)
            st.session_state.code_globals = globals_updated
            
            if success:
                st.markdown('<div class="success-output">', unsafe_allow_html=True)
                st.code(output, language="")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="error-output">', unsafe_allow_html=True)
                st.code(output, language="")
                st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="cell-header">Cellule Code 3: Zone de code libre</div>', unsafe_allow_html=True)
    
    st.write("**Exercice libre: Écrivez votre propre code**")
    st.write("Utilisez les variables créées précédemment ou créez vos propres analyses:")
    
    code_free = st.text_area(
        "Votre code Python:",
        value="""# Zone de code libre - Expérimentez !
# Vous pouvez utiliser les variables des cellules précédentes

# Exemple: Filtrer les données
if 'df' in globals():
    print("Personnes de plus de 30 ans:")
    jeunes = df[df['age'] > 30]
    print(jeunes[['nom', 'age', 'ville']])
    
    print("\\nStatistiques par ville:")
    print(df.groupby('ville')['salaire'].mean())
else:
    print("Exécutez d'abord la cellule 2 pour créer le DataFrame")

# Créer une liste de nombres pairs
pairs = [i for i in range(1, 21) if i % 2 == 0]
print(f"\\nNombres pairs de 1 à 20: {pairs}")""",
        height=300,
        key="code_free"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("▶️ Exécuter", key="run_free"):
            success, output, globals_updated = execute_python_code(code_free, st.session_state.code_globals)
            st.session_state.code_globals = globals_updated
            
            if success:
                st.markdown('<div class="success-output">', unsafe_allow_html=True)
                st.code(output, language="")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="error-output">', unsafe_allow_html=True)
                st.code(output, language="")
                st.markdown('</div>', unsafe_allow_html=True)
    
    # Bouton pour réinitialiser les variables
    if st.button("🔄 Réinitialiser les variables", key="reset_vars"):
        st.session_state.code_globals = {
            'pd': pd, 'np': np, 'plt': plt, 'sns': sns, 
            'px': px, 'go': go, 'st': st, 'datetime': datetime,
            'requests': requests, 'json': json
        }
        st.success("Variables réinitialisées!")

# Section 3: Analyse des données
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

# Section 4: Expérimentations
elif selected_section == "🔬 Expérimentations":
    st.markdown('<div class="cell-header">Cellule Expérimentation 1: Génération de données</div>', unsafe_allow_html=True)
    
    st.write("**Créez vos propres données et analysez-les**")
    
    code_exp_1 = st.text_area(
        "Code Python:",
        value="""# Générer des données aléatoires
import numpy as np
import pandas as pd

# Paramètres
n_samples = 1000
np.random.seed(42)

# Générer des données de ventes fictives
dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='D')
ventes = np.random.normal(loc=1000, scale=200, size=n_samples)
ventes = np.maximum(ventes, 0)  # Pas de ventes négatives

# Ajouter une tendance saisonnière
trend = np.sin(np.arange(n_samples) * 2 * np.pi / 365) * 150
ventes = ventes + trend

# Créer le DataFrame
df_exp = pd.DataFrame({
    'date': dates,
    'ventes': ventes,
    'mois': dates.month,
    'jour_semaine': dates.dayofweek
})

print("Données générées:")
print(df_exp.head())
print(f"\\nPériode: {df_exp['date'].min()} à {df_exp['date'].max()}")
print(f"Ventes moyennes: {df_exp['ventes'].mean():.2f}")
print(f"Écart-type: {df_exp['ventes'].std():.2f}")""",
        height=300,
        key="code_exp_1"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("▶️ Exécuter", key="run_exp_1"):
            success, output, globals_updated = execute_python_code(code_exp_1, st.session_state.code_globals)
            st.session_state.code_globals = globals_updated
            
            if success:
                st.markdown('<div class="success-output">', unsafe_allow_html=True)
                st.code(output, language="")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="error-output">', unsafe_allow_html=True)
                st.code(output, language="")
                st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="cell-header">Cellule Expérimentation 2: Analyse statistique</div>', unsafe_allow_html=True)
    
    st.write("**Analysez les données créées**")
    
    code_exp_2 = st.text_area(
        "Code Python:",
        value="""# Analyser les données (utilisez df_exp de la cellule précédente)
if 'df_exp' in globals():
    print("=== ANALYSE STATISTIQUE ===")
    
    # Statistiques descriptives
    print("\\n1. Statistiques descriptives:")
    print(df_exp['ventes'].describe())
    
    # Analyse par mois
    print("\\n2. Ventes moyennes par mois:")
    ventes_mois = df_exp.groupby('mois')['ventes'].mean().sort_values(ascending=False)
    for mois, vente in ventes_mois.head().items():
        noms_mois = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun', 
                     'Jul', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc']
        print(f"  {noms_mois[mois-1]}: {vente:.2f} €")
    
    # Analyse par jour de la semaine
    print("\\n3. Ventes moyennes par jour de la semaine:")
    jours = ['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim']
    ventes_jour = df_exp.groupby('jour_semaine')['ventes'].mean()
    for jour_num, vente in ventes_jour.items():
        print(f"  {jours[jour_num]}: {vente:.2f} €")
    
    # Trouver les meilleurs et pires jours
    print("\\n4. Records:")
    max_vente = df_exp.loc[df_exp['ventes'].idxmax()]
    min_vente = df_exp.loc[df_exp['ventes'].idxmin()]
    print(f"  Meilleur jour: {max_vente['date'].strftime('%Y-%m-%d')} ({max_vente['ventes']:.2f} €)")
    print(f"  Pire jour: {min_vente['date'].strftime('%Y-%m-%d')} ({min_vente['ventes']:.2f} €)")
    
else:
    print("Erreur: Exécutez d'abord la cellule précédente pour créer df_exp")""",
        height=300,
        key="code_exp_2"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("▶️ Exécuter", key="run_exp_2"):
            success, output, globals_updated = execute_python_code(code_exp_2, st.session_state.code_globals)
            st.session_state.code_globals = globals_updated
            
            if success:
                st.markdown('<div class="success-output">', unsafe_allow_html=True)
                st.code(output, language="")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="error-output">', unsafe_allow_html=True)
                st.code(output, language="")
                st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="cell-header">Cellule Expérimentation 3: Playground libre</div>', unsafe_allow_html=True)
    
    st.write("**Zone d'expérimentation libre**")
    st.write("Testez vos idées, créez des fonctions, explorez les données...")
    
    code_playground = st.text_area(
        "Votre code d'expérimentation:",
        value="""# PLAYGROUND - Expérimentez librement !

# Exemple 1: Créer une fonction personnalisée
def analyser_tendance(data, colonne):
    \"\"\"Analyse la tendance d'une série temporelle\"\"\"
    if len(data) < 2:
        return "Pas assez de données"
    
    debut = data[colonne].head(30).mean()
    fin = data[colonne].tail(30).mean()
    
    if fin > debut * 1.05:
        return f"Tendance croissante (+{((fin/debut-1)*100):.1f}%)"
    elif fin < debut * 0.95:
        return f"Tendance décroissante ({((fin/debut-1)*100):.1f}%)"
    else:
        return "Tendance stable"

# Utiliser la fonction
if 'df_exp' in globals():
    tendance = analyser_tendance(df_exp, 'ventes')
    print(f"Tendance des ventes: {tendance}")

# Exemple 2: Calculer des moyennes mobiles
if 'df_exp' in globals():
    df_exp['moyenne_7j'] = df_exp['ventes'].rolling(window=7).mean()
    df_exp['moyenne_30j'] = df_exp['ventes'].rolling(window=30).mean()
    
    print("\\nMoyennes mobiles calculées:")
    print(df_exp[['date', 'ventes', 'moyenne_7j', 'moyenne_30j']].tail())

# Exemple 3: Simuler des prédictions
import random
if 'df_exp' in globals():
    derniere_vente = df_exp['ventes'].iloc[-1]
    predictions = []
    
    for i in range(7):  # Prédire 7 jours
        # Prédiction simple avec un peu de randomness
        pred = derniere_vente * (0.95 + random.random() * 0.1)
        predictions.append(pred)
        derniere_vente = pred
    
    print(f"\\nPrédictions pour les 7 prochains jours:")
    for i, pred in enumerate(predictions, 1):
        print(f"  Jour +{i}: {pred:.2f} €")""",
        height=400,
        key="code_playground"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("▶️ Exécuter", key="run_playground"):
            success, output, globals_updated = execute_python_code(code_playground, st.session_state.code_globals)
            st.session_state.code_globals = globals_updated
            
            if success:
                st.markdown('<div class="success-output">', unsafe_allow_html=True)
                st.code(output, language="")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="error-output">', unsafe_allow_html=True)
                st.code(output, language="")
                st.markdown('</div>', unsafe_allow_html=True)

# Section 5: Visualisations
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

# Section 6: Exploration interactive
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

# Section 7: Calculs et modélisation
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

# Section 8: Conclusions
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
    
    st.markdown('<div class="cell-header">Cellule 13: Code final interactif</div>', unsafe_allow_html=True)
    
    st.write("**Dernière expérimentation - Résumé interactif**")
    
    code_final = st.text_area(
        "Code de synthèse:",
        value="""# Synthèse finale - Créez votre propre rapport
print("=== RAPPORT FINAL D'ANALYSE ===")
print("=" * 50)

# Vérifier les variables disponibles
variables = list(globals().keys())
data_vars = [var for var in variables if 'df' in var]

print(f"Variables de données disponibles: {data_vars}")

# Si nous avons des données, créer un rapport
if any('df' in var for var in globals()):
    print("\\n📊 RÉSUMÉ EXÉCUTIF:")
    print("- Analyse de données réalisée avec succès")
    print("- Visualisations créées")
    print("- Tendances identifiées")
    print("- Modèles statistiques appliqués")
    
    # Exemple de calcul personnalisé
    import datetime
    now = datetime.datetime.now()
    print(f"\\n📅 Rapport généré le: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Calcul créatif
    magic_number = sum(ord(c) for c in "STREAMLIT") % 100
    print(f"\\n🎲 Nombre magique du jour: {magic_number}")
    
    print("\\n✅ Analyse terminée avec succès!")
    print("\\nMerci d'avoir utilisé ce notebook interactif!")
else:
    print("Aucune donnée trouvée. Exécutez d'abord les cellules précédentes.")""",
        height=300,
        key="code_final"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("▶️ Exécuter", key="run_final"):
            success, output, globals_updated = execute_python_code(code_final, st.session_state.code_globals)
            st.session_state.code_globals = globals_updated
            
            if success:
                st.markdown('<div class="success-output">', unsafe_allow_html=True)
                st.code(output, language="")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="error-output">', unsafe_allow_html=True)
                st.code(output, language="")
                st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="cell-header">Cellule 14: Prochaines étapes</div>', unsafe_allow_html=True)
    
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
