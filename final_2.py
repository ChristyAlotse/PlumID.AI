import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
import base64
from huggingface_hub import hf_hub_download


# Configuration de la page Streamlit
st.set_page_config(
    page_title="Automatic Plum Sorting | Tri Automatique des Prunes",
    page_icon="🍃",
    layout="wide"
)


# -------- Background Image Encoder ----------
def get_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# -------- CSS Advanced Styling --------
def add_custom_css():
    css = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
        html, body, .stApp {{
            font-family: 'Inter', sans-serif;
            background-image: url("data:image/png;base64,{get_base64('prune8.jpg')}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}


        /* Appliquer un fond blanc avec arrondis et ombre douce aux blocs */
        .block-white {{
            background-color: rgba(255, 255, 255, 0.92);
            padding: 2rem;
            border-radius: 1rem;
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
        }}

        /* Pour rendre les radios et inputs plus visibles */
        input[type="radio"], .stRadio > div {{
            background-color: black;
            padding: 0.5rem;
            border-radius: 0.5rem;
    
        }}


        /* Titres */
        h1, h2, h3, h4, li, em{{
            color: #1F2937;
        }}

        /* Texte général */
        p, span, label {{
            color: #374151;
        }}
        
        
         .block{{
            color: #1F2937;
        }}               
        
        .title-text {{
            font-size: 2.5rem;
            font-weight: 800;
            color: #ffffff;
            text-align: center;
            padding: 1rem;
            background: rgba(0,0,0,0.5);
            border-radius: 10px;
        }}
        
        
        /* Boutons */
        .stButton>button {{
            background-color: #00205B;
            color: white;
            font-weight: bold;
            border-radius: 5px;
            padding: 10px 20px;
            width: 100%;
        }}

        /* Personnalisation du label */
        .custom-label {{
            font-size: 20px;
            font-weight: bold;
            color: #003399; /* Bleu foncé */
            margin-bottom: 10px;
        }}

        /* Cible le bouton avec la classe par défaut générée par Streamlit */
        button.css-1n543e5 {{
            background-color: white !important;  
            color: gray !important;
            font-size: 18px !important;
            font-weight: bold !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 12px 24px !important;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
            transition: background-color 0.3s ease !important;
        }}

        /* Effet au survol */
        button.css-1n543e5:hover {{
            background-color: skyblue !important;  /* bleu encore plus foncé */
            cursor: pointer !important;
        }}

    

        .prediction-box {{
            background-color: rgba(255, 255, 255, 0.85);
            border-left: 5px solid #10B981;
            padding: 1.2rem;
            font-size: 1.4rem;
            font-weight: 600;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            margin-top: 1.5rem;
            text-align: center;
            color: #1F2937;
        }}


        .prediction-box {{
            background-color: rgba(255, 255, 255, 0.9);
            border-left: 8px solid #8B5CF6;
            padding: 1.5rem;
            margin-top: 1rem;
            border-radius: 12px;
            font-size: 22px;
            font-weight: bold;
            color: #1F2937;
            text-align: center;
            box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        }}

        div.stAlert {{
                    background-color: skyblue !important;
                    color: white !important;
                    font-weight: bold;
                    border-radius: 10px;
                    padding: 1rem;
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
                }}

        /* Résultats */
        .result-box {{
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            font-weight: bold;
            font-size: 20px;
            margin: 10px 0;
        }}
        
        .block {{
            background-color: rgba(255,255,255,0.85);
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
            margin-bottom: 2rem;
        }}
        .custom-btn {{
            background-color: #10B981;
            color: white;
            padding: 0.6rem 1.2rem;
            border-radius: 8px;
            border: none;
            font-weight: 600;
            cursor: pointer;
        }}
        .custom-btn:hover {{
            background-color: #059669;
        }}
        

    </style>
    """
    st.markdown(css, unsafe_allow_html=True)



# -------- Load Model --------
@st.cache_resource
def load_model():
    model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=6)
    checkpoint = hf_hub_download(repo_id="RyanYvan/prune-model", filename="best_model (3).pth")
    model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
    model.eval()
    return model

model = load_model()
all_class_names = ['Spotted', 'Cracked', 'Bruised', 'Unaffected', 'Unripe', 'Rotten']

# -------- Sidebar Navigation --------
def navigation(lang):
    #st.sidebar.title("📂 Menu")
    return st.sidebar.radio(
        "📂 Menu" if lang == "Anglais" else "📂 Menu",
        ["🏠 Accueil", "📸 Prédiction", "ℹ️ À propos"]
    )

# -------- Language Selector --------
st.sidebar.image("prune4.jpg")
def select_language():
    return st.sidebar.selectbox("🌐 Langue / Language", ["Francais", "Anglais"], index=0)

# -------- Page: Accueil --------
def page_home(t):
    st.markdown("<div class='title-text'>🌿 " + t["title"] + "</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='block'>{t['home_text']}</div>", unsafe_allow_html=True)

# -------- Page: Prédiction --------
def page_prediction(t):
    st.markdown(f"<div class='block'><h2>📤 {t['load_image']}</h2></div>", unsafe_allow_html=True)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    input_method = st.radio(t["input_method"], [t["upload"], t["camera"]])
    image = None

    if input_method == t["upload"]:
        uploaded_file = st.file_uploader(t["choose_file"], type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption=t["preview"], use_column_width=True)

    else:
        img_file_buffer = st.camera_input(t["capture_image"])
        if img_file_buffer:
            image = Image.open(img_file_buffer).convert("RGB")
            st.image(image, caption=t["preview"], use_column_width=True)

    if image is not None:
        if st.button(t["analyze"], key="analyze_button"):
            img_tensor = transform(image).unsqueeze(0)
            with st.spinner(t["predicting"]):
                outputs = model(img_tensor)
                _, pred = torch.max(outputs, 1)
                st.success(f"🎯 {t['result']} : *{all_class_names[pred.item()]}*")
            st.markdown('</div>', unsafe_allow_html=True)

# -------- Page: À propos --------
def page_about(t):
    st.markdown(f"<div class='block'><h2>ℹ️ {t['about_title']}</h2><p>{t['about_text']}</p></div>", unsafe_allow_html=True)

# -------- Multilingual Text --------
texts = {
    "Francais": {
        "title": "PlumID AI: Application de Classification de Prunes",
        "home_text": """Cette application intelligente utilise l'apprentissage profond pour évaluer automatiquement l'état de vos prunes.
            Vous pouvez téléverser une image ou en capturer une avec la caméra de votre appareil. Le système analysera instantanément
            l'image pour déterminer si votre prune est saine, pourrie, trop mûre ou appartient à une autre catégorie.
            <br><br>
            Conçue pour être simple et efficace, cette application est idéale pour les agriculteurs, les distributeurs et les chercheurs
            souhaitant évaluer rapidement la qualité de leurs récoltes. """,
        "load_image": "Chargez une image de prune",
        "input_method": "Méthode d'entrée",
        "upload": "Téléverser une image",
        "camera": "Utiliser la caméra",
        "choose_file": "Choisissez une image...",
        "capture_image": "Capturer une image",
        "preview": "Image chargée",
        "analyze": "Analyser",
        "predicting": "Prédiction en cours...",
        "result": "Classe prédite",
        "about_title": "À propos",
        "about_text": """Cette application a été développée dans le cadre d’un projet visant à détecter automatiquement la qualité des prunes grâce à l’intelligence artificielle.
            Elle utilise un modèle de Deep Learning basé sur <strong>EfficientNet-B0</strong> pour classer les prunes selon leur état :
            saine, pourrie, trop mûre ou autre.
            <br><br>
            L’interface est construite avec <strong>Streamlit</strong> et stylisée avec <strong>Tailwind CSS</strong>.
            Le modèle est intégré localement ou via <strong>Hugging Face</strong>.
            <br><br>
            <strong>Fonctionnalités principales :</strong>
            <ul class="list-disc pl-5 mt-2 text-gray-700">
                <li>📸 Téléversement d’image ou capture par caméra</li>
                <li>⚡ Prédiction instantanée</li>
                <li>🌍 Interface multilingue</li>
                <li>📱 Design responsive</li>
                <li>🧠 Modèle EfficientNet-B0</li>
            </ul>
            <br>
            💡 <em>Des suggestions sont les bienvenues pour améliorer l’application !</em>"""
    },
    "Anglais": {
        "title": "PlumID AI: Plum Classification App",
        "home_text": """This intelligent application uses deep learning to automatically assess the quality of your prunes.
            You can upload an image or take one with your device’s camera. The system will instantly analyze it to determine
            whether your prune is healthy, rotten, overripe, or falls into another category.
            <br><br>
            Designed for simplicity and efficiency, this app is ideal for farmers, distributors, and researchers who want
            a fast and effective way to evaluate harvest quality.""",
        "load_image": "Load a plum image",
        "input_method": "Input method",
        "upload": "Upload an image",
        "camera": "Use camera",
        "choose_file": "Choose an image...",
        "capture_image": "Capture an image",
        "preview": "Loaded image",
        "analyze": "Analyze",
        "predicting": "Running prediction...",
        "result": "Predicted class",
        "about_title": "About",
        "about_text": """This application was developed as part of a project aimed at automatically detecting the quality of prunes using artificial intelligence.
            It uses a Deep Learning model based on <strong>EfficientNet-B0</strong> to classify prunes as healthy, rotten, overripe, or other.
            <br><br>
            The interface is built with <strong>Streamlit</strong> and styled using <strong>Tailwind CSS</strong>.
            The model is integrated locally or via <strong>Hugging Face</strong>.
            <br><br>
            <strong>Main features:</strong>
            <ul class="list-disc pl-5 mt-2 text-gray-700">
                <li>📸 Image upload or camera capture</li>
                <li>⚡ Instant prediction</li>
                <li>🌍 Multilingual interface</li>
                <li>📱 Responsive design</li>
                <li>🧠 EfficientNet-B0 model</li>
            </ul>
            <br>
            💡 <em>Suggestions to improve the app are welcome!</em>"""
    }
}

# -------- App Start --------
def main():
    add_custom_css()
    lang = select_language()
    t = texts[lang]
    page = navigation(lang)

    if page == "🏠 Accueil":
        page_home(t)
    elif page == "📸 Prédiction":
        page_prediction(t)
    elif page == "ℹ️ À propos":
        page_about(t)

if __name__ == "__main__":
    main()
