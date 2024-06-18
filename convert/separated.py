import csv

# Đọc file CSV gốc
with open('./af_data/processed_problems.csv', 'r', encoding='utf-8') as infile:
    reader = csv.reader(infile)
    data = list(reader)

# Tách dữ liệu dựa trên nhãn
lap_he_phuong_trinh = []
phuong_trinh = []
ti_le_phan_tram = []

for row in data:
    label, content = row[0], row[1]
    if label == 'Lập hệ phương trình':
        lap_he_phuong_trinh.append([label, content])
    elif label == 'phương trình':
        phuong_trinh.append([label, content])
    elif label == 'tỉ lệ phần trăm':
        ti_le_phan_tram.append([label, content])

# Ghi dữ liệu vào các file riêng biệt
def write_to_csv(filename, data):
    with open(filename, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['label', 'problem'])  # Ghi tiêu đề cột
        writer.writerows(data)

write_to_csv('./af_data/lap_he_phuong_trinh.csv', lap_he_phuong_trinh)
write_to_csv('./af_data/phuong_trinh.csv', phuong_trinh)
write_to_csv('./af_data/ti_le_phan_tram.csv', ti_le_phan_tram)
