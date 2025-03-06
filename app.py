import streamlit as st
import tempfile
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io

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
    st.session_state.result = None
if 'model' not in st.session_state:
    st.session_state.model = None

# Funções de navegação
def go_to_home():
    st.session_state.page = 'home'
    st.session_state.selected_tool = None
    st.session_state.config = {}
    st.session_state.result = None

def go_to_config():
    st.session_state.page = 'config'

def go_to_extraction():
    st.session_state.page = 'extraction'

def select_tool(tool_name):
    st.session_state.selected_tool = tool_name
    if tool_name == "docTR":
        st.session_state.config = {
            "det_arch": "db_resnet50",
            "reco_arch": "crnn_vgg16_bn",
            "assume_straight_pages": True,
            "export_as_straight_boxes": True
        }
    go_to_config()

# Inicializar modelo docTR
@st.cache_resource
def load_doctr_model(det_arch, reco_arch, assume_straight_pages, export_as_straight_boxes):
    # Carregar o modelo com os componentes especificados
    predictor = ocr_predictor(
        det_arch=det_arch,
        reco_arch=reco_arch,
        pretrained=True,
        assume_straight_pages=assume_straight_pages,
        export_as_straight_boxes=export_as_straight_boxes
    )
    
    return predictor

# Função para processar OCR com docTR
def process_doctr(file_bytes, file_type, config):
    # Inicializar ou obter modelo
    if st.session_state.model is None or st.session_state.config != config:
        with st.spinner('Carregando modelo docTR...'):
            st.session_state.model = load_doctr_model(
                config['det_arch'],
                config['reco_arch'],
                config['assume_straight_pages'],
                config['export_as_straight_boxes']
            )
    
    model = st.session_state.model
    
    # Salvar arquivo temporário
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_type) as tmp:
        tmp.write(file_bytes)
        tmp_filename = tmp.name
    
    try:
        # Determinar o tipo de documento e carregar
        if file_type.lower() == '.pdf':
            doc = DocumentFile.from_pdf(tmp_filename)
        else:
            doc = DocumentFile.from_images(tmp_filename)
        
        # Realizar OCR
        result = model(doc)
        
        return result
    
    except Exception as e:
        st.error(f"Erro ao processar o documento: {str(e)}")
        return None
    
    finally:
        if os.path.exists(tmp_filename):
            os.unlink(tmp_filename)

# Função principal para processamento de OCR
def process_ocr(uploaded_file):
    if not uploaded_file:
        st.warning("Por favor, faça o upload de um arquivo.")
        return None

    file_bytes = uploaded_file.getvalue()
    file_type = os.path.splitext(uploaded_file.name)[-1].lower()

    if file_type not in [".jpg", ".jpeg", ".png", ".pdf", ".tiff", ".bmp"]:
        st.error("Erro: Formato de arquivo não suportado. Use JPG, PNG, PDF, TIFF ou BMP.")
        return None

    if st.session_state.selected_tool == "docTR":
        return process_doctr(file_bytes, file_type, st.session_state.config)
    else:
        st.error("Ferramenta não selecionada ou não suportada.")
        return None

# Função para extrair texto do resultado
def extract_text_from_result(result):
    if result is None:
        return "Nenhum resultado disponível."
    
    full_text = ""
    for page_idx, page in enumerate(result.pages):
        if page_idx > 0:
            full_text += f"\n\n----- Página {page_idx + 1} -----\n\n"
        
        for block in page.blocks:
            for line in block.lines:
                line_text = " ".join(word.value for word in line.words)
                full_text += line_text + "\n"
            full_text += "\n"
    
    return full_text if full_text.strip() else "Nenhum texto foi extraído."

# Função para visualizar resultado
def plot_result(result):
    if result is None:
        return None
    
    synthetic_pages = result.synthesize()
    
    if not synthetic_pages:
        return None
    
    fig, axes = plt.subplots(1, len(synthetic_pages), figsize=(15, 10))
    if len(synthetic_pages) == 1:
        axes = [axes]
    
    for idx, (ax, img) in enumerate(zip(axes, synthetic_pages)):
        ax.imshow(img)
        ax.set_title(f"Página {idx+1}")
        ax.axis('off')
    
    plt.tight_layout()
    
    # Converter figura para imagem
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    
    return buf

# Interface da página inicial
def render_home():
    st.title("🔎 Ferramenta OCR com docTR")
    st.markdown("Extraia texto de imagens e documentos usando a biblioteca docTR.")

    st.subheader("Sobre docTR")
    st.markdown("""
    **Vantagens:**  
    - Processamento local (sem API externa)  
    - Modelos pré-treinados de alta qualidade
    - Suporte para múltiplos formatos (PDF, imagens)
    - Detecção e reconhecimento avançados
    
    **Características:**  
    - Abordagem em duas etapas: detecção + reconhecimento
    - Visualização interativa dos resultados
    - Suporte para documentos rotacionados
    - Estrutura hierárquica do resultado (página, bloco, linha, palavra)
    """)
    st.button("Configurar docTR", on_click=lambda: select_tool("docTR"), type="primary")

# Interface da página de configuração
def render_config():
    st.title("⚙️ Configurar docTR")

    if st.session_state.selected_tool == "docTR":
        col1, col2 = st.columns(2)
        
        with col1:
            st.session_state.config['det_arch'] = st.selectbox(
                "Modelo de Detecção",
                ["db_resnet50", "db_resnet34", "linknet_resnet18"],
                index=0,
                format_func=lambda x: {
                    "db_resnet50": "DB ResNet-50 (Mais preciso)",
                    "db_resnet34": "DB ResNet-34 (Balanceado)",
                    "linknet_resnet18": "LinkNet ResNet-18 (Mais rápido)"
                }[x]
            )
        
        with col2:
            st.session_state.config['reco_arch'] = st.selectbox(
                "Modelo de Reconhecimento",
                ["crnn_vgg16_bn", "master", "vitstr_small"],
                index=0,
                format_func=lambda x: {
                    "crnn_vgg16_bn": "CRNN VGG-16 (Padrão)",
                    "master": "MASTER (Melhor para textos irregulares)",
                    "vitstr_small": "ViTSTR Small (Baseado em Vision Transformer)"
                }[x]
            )
        
        st.session_state.config['assume_straight_pages'] = st.checkbox(
            "Assumir que páginas estão retas",
            value=True,
            help="Ative para processamento mais rápido em documentos padrão. Desative para documentos rotacionados."
        )
        
        st.session_state.config['export_as_straight_boxes'] = st.checkbox(
            "Exportar como caixas retas",
            value=True,
            help="Ative para obter caixas delimitadoras retas, mesmo em documentos rotacionados."
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
                if st.session_state.result:
                    st.success('Processamento concluído!')

    if st.session_state.result:
        # Visualização do resultado
        result_buf = plot_result(st.session_state.result)
        if result_buf:
            st.subheader("Visualização do Resultado")
            st.image(result_buf, caption="Documento com texto detectado")
        
        # Texto extraído
        st.subheader("Texto Extraído")
        extracted_text = extract_text_from_result(st.session_state.result)
        st.text_area("Conteúdo", extracted_text, height=300)
        
        # Botão de download
        txt_download = extracted_text.encode('utf-8')
        st.download_button("📥 Download do Texto", data=txt_download, file_name="texto_extraido.txt", mime="text/plain")
        
        # Opção para exportar JSON
        if st.button("Exportar estrutura como JSON"):
            json_output = st.session_state.result.export()
            st.json(json_output)

# Renderizar a página apropriada
if st.session_state.page == 'home':
    render_home()
elif st.session_state.page == 'config':
    render_config()
elif st.session_state.page == 'extraction':
    render_extraction()
