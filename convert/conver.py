import pandas as pd

# Đường dẫn đến tệp CSV đầu vào
input_file_path = r'C:\Users\fujitsu\Desktop\AI_main\af_data\chuyen_dong_thang.csv'
# Đường dẫn đến tệp CSV đầu ra
output_file_path = r'C:\Users\fujitsu\Desktop\AI_main\af_data\chuyen_dong_thang_2.csv'

# Đọc tệp CSV vào DataFrame
df = pd.read_csv(input_file_path)

# Hàm xử lý từng chuỗi văn bản
def process_text(text):
    if isinstance(text, str):
        # Chuyển đổi về chữ thường
        text = text.lower()
        # Xóa các ký tự không mong muốn
        text = text.replace('/', '').replace('\\', '')
    return text

# Áp dụng hàm xử lý cho từng cột của DataFrame
df['label'] = df['label'].apply(process_text)
df['problem'] = df['problem'].apply(process_text)

# Lưu lại DataFrame vào tệp CSV mới
df.to_csv(output_file_path, index=False)

print(f"Dữ liệu đã được xử lý và lưu vào tệp: {output_file_path}")
