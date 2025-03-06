import streamlit as st

# Configuração da página deve ser o primeiro comando Streamlit
st.set_page_config(
    page_title="Extrator de Texto com docTR",
    page_icon="📄",
    layout="wide"
)

# Importações após o set_page_config
import numpy as np
import io
import os
import tempfile
import sys
from PIL import Image

# Título e descrição
st.title("Extrator de Texto de Imagens com docTR")
st.markdown("Esta aplicação extrai texto de imagens usando a biblioteca docTR.")

# Configuração do ambiente para usar PyTorch
os.environ["USE_TORCH"] = "1"

# Carregar docTR com tratamento de erros
try:
    from doctr.io import DocumentFile
    from doctr.models import ocr_predictor
except Exception as e:
    st.error(f"""
    Erro ao carregar docTR: {e}
    
    Verifique se você tem o arquivo `packages.txt` com:
    ```
    libpango-1.0-0
    libpangocairo-1.0-0
    libpangoft2-1.0-0
    libharfbuzz0b
    libfribidi0
    librsvg2-2
    ```
    """)
    st.stop()

# Configurações do modelo no sidebar
st.sidebar.header("Configurações do Modelo")
det_arch = st.sidebar.selectbox(
    "Arquitetura de Detecção",
    ["db_resnet50", "db_mobilenet_v3_large"],
    index=0
)

reco_arch = st.sidebar.selectbox(
    "Arquitetura de Reconhecimento",
    ["crnn_vgg16_bn", "crnn_mobilenet_v3_small"],
    index=0
)

# Função para carregar o modelo
@st.cache_resource
def load_model(det_arch, reco_arch):
    try:
        return ocr_predictor(
            det_arch=det_arch,
            reco_arch=reco_arch,
            pretrained=True
        )
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        return None

# Upload de arquivo
st.header("Upload de Imagem")
uploaded_file = st.file_uploader("Escolha uma imagem", type=["jpg", "jpeg", "png"])

# Carregar o modelo apenas quando necessário
if uploaded_file is not None:
    with st.spinner("Carregando modelo..."):
        model = load_model(det_arch, reco_arch)
        
        if model is None:
            st.error("Não foi possível carregar o modelo. Verifique os logs para mais detalhes.")
            st.stop()
    
    # Processar a imagem
    with st.spinner("Processando imagem..."):
        try:
            # Ler a imagem
            image_bytes = uploaded_file.getvalue()
            image = Image.open(io.BytesIO(image_bytes))
            
            # Converter para RGB se necessário
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # Converter para numpy array
            img_np = np.array(image)
            
            # Processar com docTR
            result = model([img_np])
            
            # Mostrar resultados
            col1, col2 = st.columns(2)
            
            with col1:
                st.header("Imagem Original")
                st.image(image, use_column_width=True)
            
            with col2:
                st.header("Texto Extraído")
                # Extrair texto de cada página
                for page_idx, page in enumerate(result.pages):
                    page_text = ""
                    for block in page.blocks:
                        for line in block.lines:
                            for word in line.words:
                                page_text += word.value + " "
                            page_text += "\n"
                        page_text += "\n"
                    
                    # Mostrar texto extraído
                    st.text_area("Texto extraído", page_text, height=300)
            
            # Mostrar visualização (se disponível)
            try:
                st.header("Visualização com Detecções")
                synthetic_pages = result.synthesize()
                st.image(synthetic_pages[0], use_column_width=True)
            except Exception as e:
                st.warning(f"Não foi possível gerar visualização: {e}")
            
            # Mostrar JSON estruturado
            st.header("Dados Estruturados")
            with st.expander("Ver JSON"):
                st.json(result.export())
                
        except Exception as e:
            st.error(f"Erro ao processar imagem: {e}")
            st.error("Detalhes técnicos:", exception=e)
