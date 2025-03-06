import streamlit as st

# Configuração da página deve ser o primeiro comando Streamlit
st.set_page_config(
    page_title="Extrator de Texto de PDFs",
    page_icon="📄",
    layout="wide"
)

import numpy as np
import io
import os
import tempfile
from PIL import Image

# Configuração do ambiente para usar PyTorch
os.environ["USE_TORCH"] = "1"

# Carregamento de bibliotecas com tratamento de erros
try:
    import fitz  # PyMuPDF
    from doctr.models import ocr_predictor
except Exception as e:
    st.error(f"Erro ao carregar bibliotecas: {e}")
    st.info("Certifique-se de instalar PyMuPDF: `pip install pymupdf`")
    st.stop()

st.title("Extrator de Texto de Imagens em PDFs")
st.markdown("Esta aplicação extrai texto de imagens contidas em arquivos PDF usando docTR.")

# Função para carregar o modelo
@st.cache_resource
def load_model():
    try:
        return ocr_predictor(pretrained=True)
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        return None

# Função para extrair imagens de um PDF
def extract_images_from_pdf(pdf_file):
    images = []
    # Salvar o PDF em um arquivo temporário
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        pdf_path = tmp_file.name
    
    try:
        # Abrir o PDF usando PyMuPDF
        doc = fitz.open(pdf_path)
        
        # Para cada página do PDF
        for page_index, page in enumerate(doc):
            # Obter as imagens da página
            image_list = page.get_images(full=True)
            
            # Se não houver imagens na página
            if not image_list:
                st.info(f"Nenhuma imagem encontrada na página {page_index+1}")
                continue
                
            for img_index, img in enumerate(image_list):
                # Obter o número da imagem
                xref = img[0]
                # Extrair a imagem
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Converter para formato PIL
                try:
                    image = Image.open(io.BytesIO(image_bytes))
                    # Converter para RGB se necessário
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    images.append({
                        "page": page_index + 1,
                        "index": img_index + 1,
                        "image": image
                    })
                except Exception as e:
                    st.warning(f"Erro ao processar imagem {img_index+1} da página {page_index+1}: {e}")
        
        # Fechar o documento
        doc.close()
    except Exception as e:
        st.error(f"Erro ao extrair imagens do PDF: {e}")
    finally:
        # Remover o arquivo temporário
        os.unlink(pdf_path)
    
    return images

# Upload de arquivo
st.header("Upload de PDF")
uploaded_file = st.file_uploader("Escolha um arquivo PDF", type=["pdf"])

if uploaded_file is not None:
    # Extrair imagens do PDF
    with st.spinner("Extraindo imagens do PDF..."):
        images = extract_images_from_pdf(uploaded_file)
        
        if not images:
            st.warning("Não foram encontradas imagens no PDF.")
            st.stop()
        else:
            st.success(f"Foram encontradas {len(images)} imagens no PDF.")
    
    # Carregar o modelo
    with st.spinner("Carregando modelo OCR..."):
        model = load_model()
        if model is None:
            st.error("Não foi possível carregar o modelo OCR.")
            st.stop()
    
    # Processar cada imagem
    for img_data in images:
        st.markdown(f"## Imagem {img_data['index']} (Página {img_data['page']})")
        
        # Exibir a imagem e processar OCR
        cols = st.columns(2)
        with cols[0]:
            st.image(img_data["image"], caption=f"Imagem {img_data['index']} da página {img_data['page']}", use_column_width=True)
        
        # Processar OCR
        with st.spinner(f"Processando OCR na imagem {img_data['index']}..."):
            try:
                # Converter para numpy array
                img_np = np.array(img_data["image"])
                
                # Executar OCR
                result = model([img_np])
                
                # Extrair texto
                page_text = ""
                for page in result.pages:
                    for block in page.blocks:
                        for line in block.lines:
                            line_text = ""
                            for word in line.words:
                                line_text += word.value + " "
                            page_text += line_text.strip() + "\n"
                        page_text += "\n"
                
                # Exibir resultado
                with cols[1]:
                    st.text_area(f"Texto extraído", page_text, height=250)
                
                # Visualização com anotações
                try:
                    synthetic_pages = result.synthesize()
                    st.image(synthetic_pages[0], caption="Visualização com detecções", use_column_width=True)
                except Exception as e:
                    st.warning(f"Não foi possível gerar a visualização: {e}")
            
            except Exception as e:
                st.error(f"Erro ao processar OCR na imagem: {e}")
        
        # Separador entre imagens
        st.markdown("---")
