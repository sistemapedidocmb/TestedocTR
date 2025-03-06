import streamlit as st
import tempfile
import os
import numpy as np
from PIL import Image

# Import docTR
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

# Configuração da página
st.set_page_config(
    page_title="Ferramenta OCR com docTR",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inicialização da sessão
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'selected_tool' not in st.session_state:
    st.session_state.selected_tool = None
if 'config' not in st.session_state:
    st.session_state.config = {}
if 'result' not in st.session_state:
    st.session_state.result = ""
if 'model' not in st.session_state:
    st.session_state.model = None

# Funções de navegação
def go_to_home():
    st.session_state.page = 'home'
    st.session_state.selected_tool = None
    st.session_state.config = {}
    st.session_state.result = ""

def go_to_config():
    st.session_state.page = 'config'

def go_to_extraction():
    st.session_state.page = 'extraction'

def select_tool(tool_name):
    st.session_state.selected_tool = tool_name
    if tool_name == "docTR":
        st.session_state.config = {
            "detector": "db_resnet50",
            "recognizer": "crnn_vgg16_bn",
            "detect_orientation": True,
            "language": "pt"
        }
    go_to_config()

# Inicializar modelo docTR
@st.cache_resource
def load_doctr_model(detector, recognizer, detect_orientation):
    # Carregar o modelo com os componentes especificados
    predictor = ocr_predictor(
        det_arch=detector,
        reco_arch=recognizer,
        pretrained=True,
        assume_straight_pages=not detect_orientation
    )
    
    return predictor

# Função para processar OCR com docTR
def process_doctr(file_bytes, file_type, config):
    # Inicializar ou obter modelo
    if st.session_state.model is None:
        with st.spinner('Carregando modelo docTR...'):
            st.session_state.model = load_doctr_model(
                config['detector'],
                config['recognizer'],
                config['detect_orientation']
            )
    
    model = st.session_state.model
    
    # Salvar arquivo temporário
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_type) as tmp:
        tmp.write(file_bytes)
        tmp_filename = tmp.name
    
    try:
        # Carregar documento
        doc = DocumentFile.from_images(tmp_filename)
        
        # Realizar OCR
        result = model(doc)
        
        # Extrair texto
        extracted_text = result.export()
        
        # Formar o texto completo com base na estrutura de exportação do docTR
        full_text = ""
        for page_idx, page in enumerate(extracted_text["pages"]):
            if page_idx > 0:
                full_text += "\n\n----- Página " + str(page_idx + 1) + " -----\n\n"
            for block in page["blocks"]:
                for line in block["lines"]:
                    line_text = ""
                    for word in line["words"]:
                        line_text += word["value"] + " "
                    full_text += line_text.strip() + "\n"
                full_text += "\n"
        
        return full_text if full_text.strip() else "Nenhum texto foi extraído."
    
    except Exception as e:
        return f"Erro ao processar o documento: {str(e)}"
    
    finally:
        if os.path.exists(tmp_filename):
            os.unlink(tmp_filename)

# Função principal para processamento de OCR
def process_ocr(uploaded_file):
    if not uploaded_file:
        return "Por favor, faça o upload de um arquivo."

    file_bytes = uploaded_file.getvalue()
    file_type = os.path.splitext(uploaded_file.name)[-1].lower()

    if file_type not in [".jpg", ".jpeg", ".png", ".pdf", ".tiff", ".bmp"]:
        return "Erro: Formato de arquivo não suportado. Use JPG, PNG, PDF, TIFF ou BMP."

    if st.session_state.selected_tool == "docTR":
        return process_doctr(file_bytes, file_type, st.session_state.config)
    else:
        return "Ferramenta não selecionada ou não suportada."

# Interface da página inicial
def render_home():
    st.title("🔎 Ferramenta OCR com docTR")
    st.markdown("Extraia texto de imagens e documentos usando a biblioteca docTR.")

    st.subheader("Sobre docTR")
    st.markdown("""
    **Vantagens:**  
    - Processamento local (sem API externa)  
    - Suporte para múltiplos idiomas  
    - Código aberto e gratuito
    - Análise de layout avançada
    
    **Limitações:**  
    - Primeiro carregamento do modelo pode ser lento  
    - Requer recursos de processamento local
    """)
    st.button("Configurar docTR", on_click=lambda: select_tool("docTR"), type="primary")

# Interface da página de configuração
def render_config():
    st.title("⚙️ Configurar docTR")

    if st.session_state.selected_tool == "docTR":
        st.session_state.config['language'] = st.selectbox(
            "Idioma principal do documento",
            ["pt", "en", "es", "fr", "de", "it", "ja", "ko", "zh"],
            index=0,
            format_func=lambda x: {
                "pt": "Português",
                "en": "Inglês",
                "es": "Espanhol",
                "fr": "Francês",
                "de": "Alemão",
                "it": "Italiano",
                "ja": "Japonês",
                "ko": "Coreano",
                "zh": "Chinês"
            }[x]
        )

        st.session_state.config['detector'] = st.selectbox(
            "Detector de Texto",
            ["db_resnet50", "db_resnet34", "linknet_resnet18"],
            index=0,
            format_func=lambda x: {
                "db_resnet50": "DB ResNet-50 (Mais preciso)",
                "db_resnet34": "DB ResNet-34 (Balanceado)",
                "linknet_resnet18": "LinkNet ResNet-18 (Mais rápido)"
            }[x]
        )

        st.session_state.config['recognizer'] = st.selectbox(
            "Reconhecedor de Texto",
            ["crnn_vgg16_bn", "crnn_resnet31", "master"],
            index=0,
            format_func=lambda x: {
                "crnn_vgg16_bn": "CRNN VGG-16 (Balanceado)",
                "crnn_resnet31": "CRNN ResNet-31 (Mais preciso)",
                "master": "MASTER (Avançado para textos complexos)"
            }[x]
        )

        st.session_state.config['detect_orientation'] = st.checkbox(
            "Detectar orientação automaticamente",
            value=True
        )

    col1, col2 = st.columns(2)
    with col1:
        st.button("← Voltar para início", on_click=go_to_home)
    with col2:
        st.button("Prosseguir para extração de texto →", on_click=go_to_extraction, type="primary")

# Interface da página de extração
def render_extraction():
    st.title("📄 Extração de Texto com docTR")
    st.markdown("Faça o upload de um arquivo para extrair o texto.")

    uploaded_file = st.file_uploader("Escolha um arquivo", type=["jpg", "jpeg", "png", "pdf", "tiff", "bmp"])

    # Mostrar prévia da imagem para formatos de imagem
    if uploaded_file is not None and uploaded_file.name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        image = Image.open(uploaded_file)
        st.image(image, caption='Prévia da Imagem', width=400)
        # Retorne ao início do arquivo para processamento posterior
        uploaded_file.seek(0)

    col1, col2 = st.columns(2)
    with col1:
        st.button("← Voltar para configurações", on_click=go_to_config)
    with col2:
        if st.button("Processar", type="primary"):
            if uploaded_file is not None:
                with st.spinner('Processando, por favor aguarde...'):
                    st.session_state.result = process_ocr(uploaded_file)
                st.success('Processamento concluído!')

    if st.session_state.result:
        st.subheader("Resultado da Extração")
        st.text_area("Texto Extraído", st.session_state.result, height=300)

        txt_download = st.session_state.result.encode('utf-8')
        st.download_button("📥 Download do Texto", data=txt_download, file_name="texto_extraido.txt", mime="text/plain")

# Renderizar a página apropriada
if st.session_state.page == 'home':
    render_home()
elif st.session_state.page == 'config':
    render_config()
elif st.session_state.page == 'extraction':
    render_extraction()
