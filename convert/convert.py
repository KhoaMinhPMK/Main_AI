import json

# Đọc dữ liệu từ file JSON
with open('./bf_data/problems_and_labels.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

labels = data["labels"]
problems = data["problems"]

# Kiểm tra xem số lượng nhãn có bằng với số lượng bài toán không
if len(labels) != len(problems):
    raise ValueError("Số lượng nhãn và số lượng bài toán không khớp nhau.")

# Ghi nhãn vào file labels.txt
with open('./bf_data/labels.txt', 'w', encoding='utf-8') as file:
    for label in labels:
        file.write(label + '\n')

# Ghi bài toán vào file problems.txt
with open('./bf_data/problems.txt', 'w', encoding='utf-8') as file:
    for problem in problems:
        file.write(problem + '\n')

print("Data has been successfully converted and saved to ./data/labels.txt and ./data/problems.txt")
