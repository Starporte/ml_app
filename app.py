import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from io import StringIO
import traceback

# Configuration de la page
st.set_page_config(
    page_title="Test Code Python",
    page_icon="🐍",
    layout="wide"
)

# Style CSS simple et efficace
st.markdown("""
<style>
    .main-title {
        color: #1f77b4;
        text-align: center;
        padding: 20px 0;
        border-bottom: 2px solid #1f77b4;
        margin-bottom: 30px;
    }
    
    .code-section {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        border-left: 4px solid #28a745;
    }
    
    .output-success {
        background-color: #d4edda;
        color: #155724;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
        margin: 10px 0;
        font-family: 'Courier New', monospace;
    }
    
    .output-error {
        background-color: #f8d7da;
        color: #721c24;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #f5c6cb;
        margin: 10px 0;
        font-family: 'Courier New', monospace;
    }
    
    .stTextArea textarea {
        font-family: 'Courier New', monospace !important;
        font-size: 14px !important;
    }
</style>
""", unsafe_allow_html=True)

# Titre principal
st.markdown('<h1 class="main-title">🐍 Test d\'Exécution Python</h1>', unsafe_allow_html=True)

# Fonction pour exécuter du code Python
def execute_code(code_string):
    """Exécute du code Python et retourne le résultat"""
    # Variables disponibles pour le code
    available_vars = {
        'pd': pd,
        'np': np,
        'plt': plt,
        'st': st
    }
    
    # Rediriger stdout pour capturer les print()
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()
    
    try:
        # Exécuter le code
        exec(code_string, available_vars)
        output = captured_output.getvalue()
        return True, output
    except Exception as e:
        error_msg = f"❌ Erreur: {str(e)}"
        return False, error_msg
    finally:
        sys.stdout = old_stdout

# Interface principale
st.markdown('<div class="code-section">', unsafe_allow_html=True)

st.subheader("📝 Écrivez votre code Python ici:")

# Zone de code
code_input = st.text_area(
    "Code Python",
    value="""# Exemple simple - modifiez ce code !
print("Hello, World!")

# Calculer quelque chose
nombres = [1, 2, 3, 4, 5]
somme = sum(nombres)
print(f"La somme de {nombres} est {somme}")

# Utiliser pandas
import pandas as pd
df = pd.DataFrame({
    'nom': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35]
})
print("\\nDataFrame créé:")
print(df)
print(f"Âge moyen: {df['age'].mean()}")

# Petit calcul mathématique
import numpy as np
arr = np.array([1, 4, 9, 16, 25])
print(f"\\nRacines carrées: {np.sqrt(arr)}")""",
    height=300,
    help="Écrivez votre code Python ici. Utilisez print() pour afficher les résultats."
)

st.markdown('</div>', unsafe_allow_html=True)

# Boutons
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    if st.button("▶️ Exécuter le code", type="primary"):
        if code_input.strip():
            success, result = execute_code(code_input)
            
            if success:
                st.markdown("### ✅ Résultat:")
                st.markdown(f'<div class="output-success">{result}</div>', unsafe_allow_html=True)
            else:
                st.markdown("### ❌ Erreur:")
                st.markdown(f'<div class="output-error">{result}</div>', unsafe_allow_html=True)
        else:
            st.warning("Veuillez écrire du code avant d'exécuter!")

with col2:
    if st.button("🔄 Réinitialiser"):
        st.rerun()

with col3:
    if st.button("💡 Exemple"):
        st.info("Exemple chargé dans la zone de code!")

# Exemples rapides
st.markdown("---")
st.subheader("🚀 Exemples rapides à tester:")

examples = {
    "Calcul simple": """# Calcul d'intérêts composés
capital = 1000
taux = 0.05
annees = 10

for annee in range(1, annees + 1):
    capital = capital * (1 + taux)
    print(f"Année {annee}: {capital:.2f} €")

print(f"\\nCapital final: {capital:.2f} €")""",
    
    "Analyse de données": """# Analyse simple d'un dataset
import pandas as pd
import numpy as np

# Créer des données de vente
np.random.seed(42)
ventes = np.random.normal(1000, 200, 30)  # 30 jours de ventes

df = pd.DataFrame({
    'jour': range(1, 31),
    'ventes': ventes
})

print("Données de ventes:")
print(df.head())
print(f"\\nVentes moyennes: {df['ventes'].mean():.2f} €")
print(f"Meilleur jour: Jour {df.loc[df['ventes'].idxmax(), 'jour']} ({df['ventes'].max():.2f} €)")
print(f"Pire jour: Jour {df.loc[df['ventes'].idxmin(), 'jour']} ({df['ventes'].min():.2f} €)")""",
    
    "Boucles et fonctions": """# Créer une fonction personnalisée
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print("Suite de Fibonacci:")
for i in range(10):
    print(f"F({i}) = {fibonacci(i)}")

# Fonction pour analyser une liste
def analyser_liste(ma_liste):
    print(f"Liste: {ma_liste}")
    print(f"Longueur: {len(ma_liste)}")
    print(f"Somme: {sum(ma_liste)}")
    print(f"Moyenne: {sum(ma_liste)/len(ma_liste):.2f}")
    print(f"Min: {min(ma_liste)}, Max: {max(ma_liste)}")

# Tester la fonction
nombres = [23, 45, 12, 67, 34, 89, 56]
analyser_liste(nombres)"""
}

# Afficher les exemples
for titre, code in examples.items():
    with st.expander(f"📋 {titre}"):
        st.code(code, language="python")
        if st.button(f"Utiliser cet exemple", key=f"use_{titre}"):
            st.session_state.example_code = code
            st.rerun()

# Charger l'exemple sélectionné
if 'example_code' in st.session_state:
    code_input = st.session_state.example_code
    del st.session_state.example_code

# Informations sur les limitations
st.markdown("---")
st.info("""
**🔍 Test de faisabilité - Limitations actuelles:**
- Exécution dans un environnement sécurisé
- Pas de sauvegarde des variables entre exécutions
- Bibliothèques limitées: pandas, numpy, matplotlib
- Pas d'accès aux fichiers système
- Timeout automatique pour éviter les boucles infinies
""")

st.markdown("---")
st.markdown("*Version de test - Développé avec Streamlit*")
