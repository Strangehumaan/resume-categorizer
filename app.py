import streamlit as st
from transformers import BertForSequenceClassification, BertTokenizer
from PyPDF2 import PdfReader
import torch
from pdf2image import convert_from_bytes
from io import BytesIO
from PIL import Image
import base64
# --- App Configuration ---
st.set_page_config(
    page_title="✨ Professional Resume Categorizer",
    page_icon="📄",
    layout="wide"
)

# Session state for popup
if 'show_popup' not in st.session_state:
    st.session_state.show_popup = False

# Function to convert PDF first page to image
def get_pdf_first_page_image(pdf_file):
    try:
        images = convert_from_bytes(pdf_file.read(), first_page=1, last_page=1)
        return images[0]
    except Exception as e:
        st.error(f"Error converting PDF to image: {str(e)}")
        return None

# Preview button and popup code
def show_preview_button(uploaded_file):
    if st.button("Show Preview"):
        st.session_state.show_popup = True

    if st.session_state.show_popup and uploaded_file is not None:
        first_page_image = get_pdf_first_page_image(uploaded_file)
        if first_page_image:
            img_buffer = BytesIO()
            first_page_image.save(img_buffer, format="PNG")
            img_bytes = img_buffer.getvalue()

            col1, col2, col3 = st.columns([1, 3, 1])
            with col2:
                with st.container(border=True):
                    st.image(img_bytes, caption="First Page Preview", use_container_width=True)
                    if st.button("Close", key="close_preview_button"):
                        st.session_state.show_popup = False
                        st.experimental_rerun()

# --- Category Data ---
CATEGORIES = [
    'ACCOUNTANT', 'ADVOCATE', 'AGRICULTURE', 'APPAREL', 'ARTS',
    'AUTOMOBILE', 'AVIATION', 'BANKING', 'BPO', 'BUSINESS-DEVELOPMENT',
    'CHEF', 'CONSTRUCTION', 'CONSULTANT', 'DESIGNER', 'DIGITAL-MEDIA',
    'ENGINEERING', 'FINANCE', 'FITNESS', 'HEALTHCARE', 'HR',
    'INFORMATION-TECHNOLOGY', 'PUBLIC-RELATIONS', 'SALES', 'TEACHER'
]

# --- Model Loading ---
@st.cache_resource
def load_model():
    model_path = 'strangehumaan/resume-categorizer'  
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()

# --- Helper Functions ---
def extract_text(pdf_file):
    reader = PdfReader(pdf_file)
    return " ".join(page.extract_text() or "" for page in reader.pages)

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)[0]
    pred_idx = torch.argmax(probs).item()
    return CATEGORIES[pred_idx], round(probs[pred_idx].item() * 100, 1)

# Improved category display function without icons
def display_category(category, confidence):
    st.markdown(f"""
    <div style='
        background: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        margin: 1rem 0;
    ' onmouseover='this.style.boxShadow="0 6px 12px rgba(0,0,0,0.15)"'
      onmouseout='this.style.boxShadow="0 4px 6px rgba(0,0,0,0.1)"'>
        <h3 style='margin:0; display: flex; align-items: center;'>
            <span style='color: #2c3e50;'>{category}</span>
            <span style='margin-left: auto; font-size: 0.9em; color: #7f8c8d;'>
                {confidence}% Confidence
            </span>
        </h3>
    </div>
    """, unsafe_allow_html=True)
    st.progress(confidence / 100)

# --- Sidebar with Improved Category Browser (No Icons) ---
with st.sidebar:
    st.title("Categories")
    
    with st.expander("🔍 Browse All (24)", expanded=False):
        # Search bar for categories
        search_term = st.text_input("Search categories", "")
        filtered_categories = [cat for cat in CATEGORIES if search_term.lower() in cat.lower()]
        
        # Grid layout for categories
        cols = st.columns(2)
        for i, category in enumerate(sorted(filtered_categories)):
            with cols[i % 2]:
                st.markdown(f"""
                <div style='
                    padding: 8px;
                    border-radius: 5px;
                    background: #f8f9fa;
                    margin: 5px 0;
                    cursor: pointer;
                ' onmouseover='this.style.background="#e9ecef"'>
                    {category}
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    **How to use:**
    1. Upload a PDF resume
    2. Click 'Analyze'
    3. View predicted category
    """)
    st.markdown("---")
    with open("example.zip", "rb") as file:
        b64 = base64.b64encode(file.read()).decode()
        href = f'''
                    <a href="data:application/zip;base64,{b64}" download="example.zip" 
                    style="font-size: 0.9em; color: #6c757d; text-decoration: none;">
                    📦 Download Examples</a>
                '''
        st.markdown(href, unsafe_allow_html=True)

# --- Main Interface ---
st.title("📄 Professional Resume Categorizer")
st.markdown("Upload a resume to automatically classify it into one of 24 professional categories")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file:
    try:
        show_preview_button(uploaded_file)
    except Exception as e:
        st.warning(f"Couldn't generate preview: {str(e)}")

    if st.button("🔍 Analyze Resume", type="primary"):
        with st.spinner("Analyzing resume content..."):
            try:
                text = extract_text(uploaded_file)
                
                if not text.strip():
                    st.error("No text extracted - please upload a searchable PDF")
                else:
                    st.subheader("Analysis Results")
                    category, confidence = predict(text)
                    display_category(category, confidence)
                    
                    with st.expander("📝 View extracted text"):
                        st.text_area("Text", text[:3000], height=200, label_visibility="collapsed")
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")

# --- Footer ---
st.markdown("---")
st.caption("Professional Resume Categorizer - Classifies resumes into 24 professional categories")
st.caption("Made By Mohammad Saad Nathani & Navya Sharma")
