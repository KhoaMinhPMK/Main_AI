import pdfplumber
import re
import csv

def preprocess_text(text):
    # Chuyển thành chữ thường
    text = text.lower()
    # Xóa kí tự đặc biệt và dấu câu
    text = re.sub(r'[^\w\s]', '', text)
    return text

def extract_problems_from_pdf(file_path):
    problems = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            # Xử lý văn bản trích xuất để đảm bảo nó không bị ngắt dòng không mong muốn
            text = text.replace('\n', ' ')
            # Tìm tất cả các đoạn văn bản bắt đầu bằng "Bài" và kết thúc trước "Lời giải"
            matches = re.findall(r'(Bài \d+.*?)(?=Lời giải)', text, re.DOTALL)
            for match in matches:
                # Loại bỏ từ "Bài ... ." ở đầu và "Lời giải" ở cuối
                match = re.sub(r'Bài \d+\.\s*', '', match)
                match = re.sub(r'Lời giải.*', '', match)
                # Tiền xử lý văn bản
                match = preprocess_text(match)
                problems.append(match)
    return problems

def save_problems_to_csv(problems, file_path):
    with open(file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['label', 'problem'])
        for problem in problems:
            writer.writerow(['bài toán chuyển động', problem])

# Đường dẫn đến file PDF
pdf_file_path = 'a.pdf'
# Đường dẫn đến file CSV
csv_file_path = 'problems.csv'

# Trích xuất các bài toán từ PDF
problems = extract_problems_from_pdf(pdf_file_path)

# Lưu các bài toán vào file CSV
save_problems_to_csv(problems, csv_file_path)
