import streamlit as st
import requests
from io import StringIO
from PyPDF2 import PdfReader
import docx
from bs4 import BeautifulSoup

st.set_page_config(page_title="Phân loại văn bản", layout="wide")
st.title("Phân loại chủ đề văn bản")

input_method = st.radio(
    "Chọn phương thức nhập dữ liệu",
    ["Nhập văn bản", "Nhập đường link bài báo", "Tải lên file"]
)

text_input = None
url_input = None
file_input = None
file_content = ""
base_url = "https://de0c-34-59-164-109.ngrok-free.app/predict"

if input_method == "Nhập văn bản":
    text_input = st.text_area("Nhập đoạn văn bản")

elif input_method == "Nhập đường link bài báo":
    url_input = st.text_input("Nhập đường link bài báo")
    
    if url_input:
        try:
            response = requests.get(url_input)
            soup = BeautifulSoup(response.content, "html.parser")

            article = soup.find("article")
            if not article:
                article = soup.find("div", class_="sidebar-1")  # fallback
            if article:
                paragraphs = article.find_all("p")
                content = "\n".join(p.get_text(strip=True) for p in paragraphs)
                url_input = content
                file_content = url_input
            else:
                url_input = "Không thể trích xuất nội dung từ đường link này."
                file_content = url_input

        except Exception as e:
            url_input = f"Lỗi khi tải bài viết: {str(e)}"

elif input_method == "Tải lên file":
    file_input = st.file_uploader("Tải lên file văn bản", type=["txt", "pdf", "docx"])

    if file_input is not None:
        if file_input.type == "text/plain":
            content = file_input.read().decode("utf-8")
            file_content = content

        elif file_input.type == "application/pdf":
            pdf_reader = PdfReader(file_input)
            content = ""
            for page in pdf_reader.pages:
                content += page.extract_text()
            file_content = content

        elif file_input.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(file_input)
            content = ""
            for para in doc.paragraphs:
                content += para.text + "\n"
            file_content = content

        else:
            file_content = "Không thể đọc nội dung từ file này."

if file_content:
    st.subheader("Nội dung của file:")
    st.text_area("Nội dung bài báo", value=file_content, height=300)

model_choice = st.radio(
    "Chọn mô hình",
    ["Tất cả các mô hình", "SVM + W2V", "Naive Bayes + TF-IDF", "phoBERT"]
)

if st.button("Dự đoán"):
    input_text = text_input or url_input or file_content
    if not input_text:
        st.warning("Vui lòng nhập văn bản, URL bài báo hoặc tải lên file.")
    else:
        if model_choice == "Tất cả các mô hình":
            models = ["SVM + W2V", "Naive Bayes + TF-IDF", "phoBERT"]
            for model in models:
                data = {
                    "text": input_text,
                    "model_name": model
                }
                try:
                    response = requests.post(base_url, json=data)
                    result = response.json()
                    if "prediction" in result:
                        st.success(f"**{model}** → Kết quả dự đoán: **{result['prediction']}**")
                    else:
                        st.error(f"{model}: {result.get('error', 'Lỗi không xác định')}")
                except Exception as e:
                    st.error(f"{model}: Lỗi khi gọi API: {e}")
        else:
            data = {
                "text": input_text,
                "model_name": model_choice
            }
            try:
                response = requests.post(base_url, json=data)

                if response.status_code != 200:
                    st.error(f"{model_choice}: Server trả về mã lỗi {response.status_code}")
                    st.text(response.text)
                else:
                    result = response.json()
                    if "prediction" in result:
                        st.success(f"**{model_choice}** → Kết quả dự đoán: **{result['prediction']}**")
                    else:
                        st.error(f"{model_choice}: {result.get('error', 'Lỗi không xác định')}")
            except Exception as e:
                st.error(f"{model_choice}: Lỗi khi gọi API: {e}")

