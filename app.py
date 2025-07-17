import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from io import StringIO
import traceback
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import make_regression

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
        'st': st,
        'train_test_split': train_test_split,
        'LinearRegression': LinearRegression,
        'RandomForestRegressor': RandomForestRegressor,
        'mean_squared_error': mean_squared_error,
        'r2_score': r2_score,
        'make_regression': make_regression
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
    value="""# Exemple avec Machine Learning - modifiez ce code !
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

print("🤖 Exemple de Machine Learning avec sklearn")
print("=" * 50)

# Créer un dataset synthétique
X, y = make_regression(n_samples=100, n_features=3, noise=10, random_state=42)
print(f"Dataset créé: {X.shape[0]} échantillons, {X.shape[1]} variables")

# Diviser les données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Données d'entraînement: {X_train.shape[0]} échantillons")
print(f"Données de test: {X_test.shape[0]} échantillons")

# Créer et entraîner le modèle
model = LinearRegression()
model.fit(X_train, y_train)
print("\\n✅ Modèle entraîné!")

# Faire des prédictions
y_pred = model.predict(X_test)

# Évaluer les performances
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\\n📊 Performances du modèle:")
print(f"MSE (Mean Squared Error): {mse:.2f}")
print(f"R² Score: {r2:.4f}")

# Afficher quelques prédictions
print(f"\\n🔮 Exemples de prédictions:")
for i in range(min(5, len(y_test))):
    print(f"Réel: {y_test[i]:.2f}, Prédit: {y_pred[i]:.2f}")""",
    height=300,
    help="Écrivez votre code Python ici. Sklearn est disponible !"
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
    "Régression linéaire": """# Prédiction des prix immobiliers
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

print("🏠 Prédiction des prix immobiliers")
print("=" * 40)

# Simuler des données immobilières
np.random.seed(42)
superficie = np.random.normal(100, 30, 200)  # Surface en m²
chambres = np.random.poisson(3, 200)  # Nombre de chambres
anciennete = np.random.uniform(0, 50, 200)  # Âge du bien

# Prix basé sur ces caractéristiques + bruit
prix = (superficie * 3000 + chambres * 15000 - anciennete * 500 + 
        np.random.normal(0, 20000, 200))

# Créer le dataset
X = np.column_stack([superficie, chambres, anciennete])
y = prix

# Division train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modèle
model = LinearRegression()
model.fit(X_train, y_train)

# Prédictions
y_pred = model.predict(X_test)

# Résultats
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"Erreur moyenne: {mean_squared_error(y_test, y_pred, squared=False):.0f} €")

print("\\n🔮 Exemples de prédictions:")
for i in range(5):
    print(f"Superficie: {X_test[i][0]:.0f}m², Chambres: {X_test[i][1]:.0f}, Âge: {X_test[i][2]:.0f}ans")
    print(f"Prix réel: {y_test[i]:.0f}€, Prix prédit: {y_pred[i]:.0f}€")
    print("-" * 50)""",

    "Classification": """# Classification avec Random Forest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

print("🎯 Classification avec Random Forest")
print("=" * 40)

# Créer un dataset de classification
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, 
                         n_redundant=3, n_clusters_per_class=1, random_state=42)

# Labels pour rendre plus réaliste
class_names = ['Classe A', 'Classe B']
print(f"Dataset: {X.shape[0]} échantillons, {X.shape[1]} caractéristiques")
print(f"Classes: {class_names}")

# Division des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modèle Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Prédictions
y_pred = rf_model.predict(X_test)

# Évaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"\\n📊 Précision du modèle: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Importance des caractéristiques
feature_importance = rf_model.feature_importances_
print("\\n🔍 Importance des caractéristiques:")
for i, importance in enumerate(feature_importance):
    print(f"Caractéristique {i+1}: {importance:.4f}")

# Exemples de prédictions
print("\\n🔮 Exemples de prédictions:")
for i in range(10):
    predicted_class = class_names[y_pred[i]]
    actual_class = class_names[y_test[i]]
    status = "✅" if y_pred[i] == y_test[i] else "❌"
    print(f"{status} Prédit: {predicted_class}, Réel: {actual_class}")""",

    "Clustering": """# Clustering avec K-Means
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import numpy as np

print("🎪 Clustering avec K-Means")
print("=" * 30)

# Créer des données avec clusters naturels
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)

print(f"Dataset: {X.shape[0]} points, {X.shape[1]} dimensions")
print(f"Clusters réels: {len(np.unique(y_true))}")

# Appliquer K-Means
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
y_pred = kmeans.fit_predict(X)

# Centres des clusters
centers = kmeans.cluster_centers_
print(f"\\n🎯 Centres des clusters trouvés:")
for i, center in enumerate(centers):
    print(f"Cluster {i}: [{center[0]:.2f}, {center[1]:.2f}]")

# Inertie (somme des distances au carré)
inertia = kmeans.inertia_
print(f"\\n📊 Inertie du modèle: {inertia:.2f}")

# Analyser les clusters
print("\\n🔍 Analyse des clusters:")
for i in range(4):
    cluster_points = X[y_pred == i]
    print(f"Cluster {i}: {len(cluster_points)} points")
    print(f"  Moyenne X: {cluster_points[:, 0].mean():.2f}")
    print(f"  Moyenne Y: {cluster_points[:, 1].mean():.2f}")

# Comparaison avec les vrais clusters
from sklearn.metrics import adjusted_rand_score
ari = adjusted_rand_score(y_true, y_pred)
print(f"\\n✅ Score ARI (similitude): {ari:.4f}")""",

    "Analyse simple": """# Analyse simple d'un dataset
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
print(f"Pire jour: Jour {df.loc[df['ventes'].idxmin(), 'jour']} ({df['ventes'].min():.2f} €)")"""
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
**🤖 Sklearn disponible - Fonctionnalités ML:**
- **Régression** : LinearRegression, RandomForestRegressor
- **Classification** : RandomForestClassifier, LogisticRegression
- **Clustering** : KMeans, DBSCAN
- **Métriques** : accuracy_score, r2_score, mean_squared_error
- **Datasets** : make_regression, make_classification, make_blobs
- **Preprocessing** : train_test_split, StandardScaler

**📚 Bibliothèques disponibles:** pandas, numpy, matplotlib, sklearn
""")

st.markdown("---")
st.markdown("*Version de test - Développé avec Streamlit*")
