import os
import streamlit as st
import numpy as np
from PIL import Image
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import logging
from pdf2image import convert_from_bytes
import io
import traceback

# Verificação inicial de dependências
try:
    from weasyprint import HTML
    import cairocffi
except ImportError as e:
    st.error(f"Erro de dependência: {str(e)}")
    st.stop()
except OSError as e:
    st.error(f"Erro de biblioteca do sistema: {str(e)}")
    st.stop()

# Configurações iniciais
st.set_page_config(
    page_title="DocTR OCR Completo",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(_name_)

# Inicialização do estado da sessão
if 'processed_text' not in st.session_state:
    st.session_state.processed_text = ""
if 'config' not in st.session_state:
    st.session_state.config = {
        "model_type": "accurate",
        "det_thresh": 0.5,
        "rec_thresh": 0.3
    }

@st.cache_resource
def load_doctr_model(model_type):
    """Carrega o modelo DocTR com cache"""
    logger.info(f"Carregando modelo {model_type}...")
    return ocr_predictor(
        det_arch='db_resnet50' if model_type == "accurate" else 'db_mobilenet_v3_large',
        reco_arch='crnn_vgg16_bn' if model_type == "accurate" else 'crnn_mobilenet_v3_small',
        pretrained=True
    )

def process_file(uploaded_file):
    """Processa o arquivo carregado"""
    try:
        file_bytes = uploaded_file.read()
        
        if uploaded_file.type == "application/pdf":
            images = convert_from_bytes(
                file_bytes,
                dpi=300,
                poppler_path="/usr/bin"
            )
            images = [np.array(img) for img in images]
        else:
            image = Image.open(io.BytesIO(file_bytes))
            images = [np.array(image)]
            
        return images
    
    except Exception as e:
        logger.error(f"Erro no processamento do arquivo: {str(e)}")
        raise

def main():
    st.title("📄 OCR Profissional com DocTR")
    st.markdown("Sistema completo para extração de texto de documentos")

    # Sidebar com configurações
    with st.sidebar.expander("⚙ Configurações Avançadas", expanded=True):
        st.session_state.config["model_type"] = st.selectbox(
            "Tipo de Modelo",
            ["accurate", "fast"],
            index=0,
            help="Modelo preciso (mais lento) ou rápido (menos preciso)"
        )
        
        st.session_state.config["det_thresh"] = st.slider(
            "Limiar de Detecção",
            0.1, 1.0, 0.5, 0.05,
            help="Confiança mínima para detecção de áreas de texto"
        )
        
        st.session_state.config["rec_thresh"] = st.slider(
            "Limiar de Reconhecimento",
            0.1, 1.0, 0.3, 0.05,
            help="Confiança mínima para reconhecimento de caracteres"
        )

    # Upload de arquivo
    uploaded_file = st.file_uploader(
        "Carregue seu documento (PDF ou imagem)",
        type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp"],
        accept_multiple_files=False
    )

    if uploaded_file is not None:
        if st.button("Processar Documento", type="primary"):
            try:
                with st.spinner('Processando...'):
                    # Carregar modelo
                    predictor = load_doctr_model(st.session_state.config["model_type"])
                    
                    # Processar arquivo
                    images = process_file(uploaded_file)
                    
                    # Extrair texto
                    full_text = []
                    for img in images:
                        result = predictor([img])
                        page_text = "\n".join([
                            " ".join([word.value for word in line.words])
                            for block in result.pages[0].blocks
                            for line in block.lines
                        ])
                        full_text.append(page_text)
                    
                    st.session_state.processed_text = "\n\n".join(full_text)
                
                st.success("Processamento concluído com sucesso!")

            except Exception as e:
                st.error(f"Erro: {str(e)}")
                logger.error(traceback.format_exc())

    # Exibir resultados
    if st.session_state.processed_text:
        st.subheader("Resultado da Extração")
        st.text_area("Texto Extraído", st.session_state.processed_text, height=500)
        
        st.download_button(
            label="📥 Baixar Resultado",
            data=st.session_state.processed_text,
            file_name="texto_extraido.txt",
            mime="text/plain"
        )

if __name__ == "__main__":
    main()
