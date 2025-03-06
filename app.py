import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import os
import tempfile
import sys

# Configuração do ambiente para usar PyTorch (padrão) ou TensorFlow
import os
if os.environ.get("USE_TF", "0") == "1":
    st.sidebar.write("Usando TensorFlow")
    os.environ["USE_TF"] = "1"
else:
    st.sidebar.write("Usando PyTorch")
    os.environ["USE_TORCH"] = "1"

# Importar docTR após configurar o ambiente
try:
    from doctr.io import DocumentFile
    from doctr.models import ocr_predictor
except OSError as e:
    st.error(f"""
    Erro ao carregar bibliotecas do sistema: {e}
    
    Se estiver no Streamlit Cloud, verifique se você adicionou um arquivo `packages.txt` com as dependências necessárias:
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

# Configuração da página
st.set_page_config(
    page_title="Extrator de Texto com docTR",
    page_icon="📄",
    layout="wide"
)

# Título e descrição
st.title("Extrator de Texto de Imagens com docTR")
st.markdown("""
Esta aplicação permite extrair texto de imagens usando a biblioteca docTR.
Upload uma imagem e veja o texto extraído!
""")

# Configurações do modelo no sidebar
st.sidebar.header("Configurações do Modelo")
det_arch = st.sidebar.selectbox(
    "Arquitetura de Detecção",
    ["db_resnet50", "db_mobilenet_v3_large", "linknet_resnet18", "linknet_resnet34", "fast_tiny"],
    index=0
)

reco_arch = st.sidebar.selectbox(
    "Arquitetura de Reconhecimento",
    ["crnn_vgg16_bn", "crnn_mobilenet_v3_small", "sar_resnet31", "master", "vit_tiny", "vitstr_small", "parseq"],
    index=0
)

assume_straight = st.sidebar.checkbox("Assumir caixas retas", value=True)
straighten_boxes = st.sidebar.checkbox("Exportar como caixas retas", value=True)

# Função para carregar o modelo
@st.cache_resource
def load_model(det_arch, reco_arch, assume_straight, straighten_boxes):
    return ocr_predictor(
        det_arch=det_arch,
        reco_arch=reco_arch,
        pretrained=True,
        assume_straight_pages=assume_straight,
        export_as_straight_boxes=straighten_boxes
    )

# Carregar o modelo com as configurações selecionadas
with st.spinner("Carregando modelo..."):
    model = load_model(det_arch, reco_arch, assume_straight, straighten_boxes)
    st.sidebar.success("Modelo carregado!")

# Upload de arquivo
st.header("Upload de Imagem")
uploaded_file = st.file_uploader("Escolha uma imagem", type=["jpg", "jpeg", "png", "pdf"])

if uploaded_file is not None:
    # Exibir indicador de progresso
    with st.spinner("Processando..."):
        # Determinar se é PDF ou imagem
        file_extension = uploaded_file.name.split(".")[-1].lower()
        
        if file_extension == "pdf":
            # Salvar o PDF temporariamente para processar
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                pdf_path = tmp_file.name
            
            # Processar o PDF
            doc = DocumentFile.from_pdf(pdf_path)
            # Remover arquivo temporário
            os.unlink(pdf_path)
        else:
            # Processar a imagem
            image_bytes = uploaded_file.getvalue()
            image = Image.open(io.BytesIO(image_bytes))
            # Converter para RGB se necessário
            if image.mode != "RGB":
                image = image.convert("RGB")
            # Salvar temporariamente
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_file:
                image.save(tmp_file.name)
                img_path = tmp_file.name
            
            # Abrir com docTR
            doc = DocumentFile.from_images(img_path)
            # Remover arquivo temporário
            os.unlink(img_path)
        
        # Realizar OCR
        result = model(doc)
        
        # Mostrar resultados em colunas
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("Imagem Original")
            # Mostrar a imagem original
            if file_extension == "pdf":
                st.warning("Visualização da primeira página do PDF")
                # Extrair a primeira página como imagem
                page_img = doc[0]
                st.image(page_img, use_column_width=True)
            else:
                st.image(image, use_column_width=True)
        
        with col2:
            st.header("Texto Extraído")
            # Mostrar texto extraído
            for page_idx, page in enumerate(result.pages):
                st.subheader(f"Página {page_idx+1}")
                
                # Extrair todo o texto
                page_text = ""
                for block in page.blocks:
                    for line in block.lines:
                        for word in line.words:
                            page_text += word.value + " "
                        page_text += "\n"
                    page_text += "\n"
                
                # Mostrar o texto extraído
                st.text_area(f"Texto da página {page_idx+1}", page_text, height=400)
        
        # Mostrar página sintetizada
        st.header("Visualização com Detecções")
        synthetic_pages = result.synthesize()
        
        # Mostrar todas as páginas sintetizadas
        for page_idx, page in enumerate(synthetic_pages):
            st.subheader(f"Página {page_idx+1} com detecções")
            st.image(page, use_column_width=True)
        
        # Exportar como JSON
        st.header("Dados Estruturados (JSON)")
        json_output = result.export()
        st.json(json_output)

# Instruções de uso do Streamlit Cloud
st.header("Implantação no Streamlit Cloud")
st.markdown("""
### Para implantar este aplicativo no Streamlit Cloud:

1. Crie um arquivo `requirements.txt` com as dependências:
```
streamlit
numpy
pillow
python-doctr[torch]
opencv-python-headless
```

2. Se preferir usar TensorFlow em vez de PyTorch, use:
```
streamlit
numpy
pillow
python-doctr[tf]
opencv-python-headless
```

3. Crie um repositório GitHub com este arquivo e o requirements.txt
4. Acesse o [Streamlit Cloud](https://streamlit.io/cloud) e implante usando seu repositório
5. Escolha o arquivo main como `app.py` e configure a variável de ambiente `USE_TORCH=1` ou `USE_TF=1`
""")
