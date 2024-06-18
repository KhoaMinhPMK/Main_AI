import re
import csv
from underthesea import word_tokenize

def remove_html(txt):
    return re.sub(r'<[^>]*>', '', txt)

def loaddicchar():
    dic = {}
    char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split('|')
    charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split('|')
    for i in range(len(char1252)):
        dic[char1252[i]] = charutf8[i]
    return dic

dicchar = loaddicchar()

def convert_unicode(txt):
    return re.sub('|'.join(dicchar.keys()), lambda x: dicchar[x.group()], txt)

def chuan_hoa_dau_tu_tieng_viet(word):
    bang_nguyen_am = [
        ["a", "à", "á", "ả", "ã", "ạ"],
        ["ă", "ằ", "ắ", "ẳ", "ẵ", "ặ"],
        ["â", "ầ", "ấ", "ẩ", "ẫ", "ậ"],
        ["e", "è", "é", "ẻ", "ẽ", "ẹ"],
        ["ê", "ề", "ế", "ể", "ễ", "ệ"],
        ["i", "ì", "í", "ỉ", "ĩ", "ị"],
        ["o", "ò", "ó", "ỏ", "õ", "ọ"],
        ["ô", "ồ", "ố", "ổ", "ỗ", "ộ"],
        ["ơ", "ờ", "ớ", "ở", "ỡ", "ợ"],
        ["u", "ù", "ú", "ủ", "ũ", "ụ"],
        ["ư", "ừ", "ứ", "ử", "ữ", "ự"],
        ["y", "ỳ", "ý", "ỷ", "ỹ", "ỵ"]
    ]
    nguyen_am_to_ids = {}
    for i in range(len(bang_nguyen_am)):
        for j in range(len(bang_nguyen_am[i])):
            nguyen_am_to_ids[bang_nguyen_am[i][j]] = (i, j)

    dau_cau = 0
    nguyen_am_index = []
    qu_or_gi = False
    chars = list(word)
    for index, char in enumerate(chars):
        x, y = nguyen_am_to_ids.get(char, (-1, -1))
        if x == -1:
            continue
        if x == 9:
            if index != 0 and chars[index - 1] == 'q':
                chars[index] = 'u'
                qu_or_gi = True
        elif x == 5:
            if index != 0 and chars[index - 1] == 'g':
                chars[index] = 'i'
                qu_or_gi = True
        if y != 0:
            dau_cau = y
            chars[index] = bang_nguyen_am[x][0]
        if not qu_or_gi or index != 1:
            nguyen_am_index.append(index)

    if not nguyen_am_index:
        return word

    if len(nguyen_am_index) < 2:
        if qu_or_gi:
            if len(chars) == 2:
                x, y = nguyen_am_to_ids.get(chars[1])
                chars[1] = bang_nguyen_am[x][dau_cau]
            else:
                x, y = nguyen_am_to_ids.get(chars[2], (-1, -1))
                if x != -1:
                    chars[2] = bang_nguyen_am[x][dau_cau]
                else:
                    chars[1] = bang_nguyen_am[5][dau_cau] if chars[1] == 'i' else bang_nguyen_am[9][dau_cau]
        else:
            x, y = nguyen_am_to_ids.get(chars[nguyen_am_index[0]])
            chars[nguyen_am_index[0]] = bang_nguyen_am[x][dau_cau]
    else:
        if len(nguyen_am_index) == 2 and nguyen_am_index[-1] == len(chars) - 1:
            x, y = nguyen_am_to_ids.get(chars[nguyen_am_index[0]])
            chars[nguyen_am_index[0]] = bang_nguyen_am[x][dau_cau]
        else:
            x, y = nguyen_am_to_ids.get(chars[nguyen_am_index[1]])
            chars[nguyen_am_index[1]] = bang_nguyen_am[x][dau_cau]
    return ''.join(chars)

def chuan_hoa_dau_cau_tieng_viet(sentence):
    words = sentence.split()
    for index, word in enumerate(words):
        prefix = re.match(r'^\W+', word)
        suffix = re.match(r'\W+$', word)
        prefix = prefix.group() if prefix else ''
        suffix = suffix.group() if suffix else ''
        cw = re.sub(r'^\W+|\W+$', '', word)
        if len(cw) == 0:
            continue
        cw = chuan_hoa_dau_tu_tieng_viet(cw)
        words[index] = prefix + cw + suffix
    return ' '.join(words)

def remove_punctuation(txt):
    return re.sub(r'[^\w\s]', '', txt)

def chuan_hoa_khoang_trang(txt):
    return re.sub(r'\s+', ' ', txt).strip()

def tokenize(txt):
    return ' '.join(word_tokenize(txt))

def clean_text(text):
    # print("Original:", text)
    text = remove_html(text)
    # print("After remove_html:", text)
    text = convert_unicode(text)
    # print("After convert_unicode:", text)
    text = chuan_hoa_dau_cau_tieng_viet(text)
    # print("After chuan_hoa_dau_cau_tieng_viet:", text)
    text = text.lower()
    # print("After lower:", text)
    text = remove_punctuation(text)
    # print("After remove_punctuation:", text)
    text = chuan_hoa_khoang_trang(text)
    # print("After chuan_hoa_khoang_trang:", text)
    text = tokenize(text)
    # print("After tokenize:", text)
    return text

text = "<div>Chào mừng đến với <b>OpenAI</b>! Đây là một ví dụ về chuẩn hóa văn bản tiếng Việt.</div>"
cleaned_text = clean_text(text)
print("Final cleaned text:", cleaned_text)

# labels = ["Lập hệ phương trình"]
# problems = [["Hai lớp 9A và 9B có tổng số học sinh là 84. Trong đợt vận động mua bút ủng hộ nạn nhân chất độc màu da cam, mỗi học sinh lớp 9A mua 3 bút, mỗi học sinh lớp 9B mua 2 bút. Tính số học sinh mỗi lớp biết rằng tổng số bút hai lớp mua là 209 chiếc. ĐS: số hs lớp 9A là 41 hs, lớp 9B là 43 hs"]]
# Hàm để đọc dữ liệu từ file
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file.readlines()]

# Đường dẫn đến các file
labels_file_path = './bf_data/labels.txt'
problems_file_path = './bf_data/problems.txt'

# Đọc nhãn từ file labels.txt
labels = read_file(labels_file_path)

# Đọc các bài toán từ file problems.txt
problems = read_file(problems_file_path)

# Kiểm tra xem số lượng nhãn có bằng với số lượng bài toán không
if len(labels) != len(problems):
    raise ValueError("Số lượng nhãn và số lượng bài toán không khớp nhau.")


# Clean each problem
cleaned_problems = [clean_text(problem) for problem in problems]

# Organize data into a list of dictionaries for easy CSV writing
# Organize data into a list of dictionaries for easy CSV writing
data = []
for label, problem in zip(labels, cleaned_problems):
    data.append({"label": label, "problem": problem})

# Save to CSV
csv_file = "./af_data/processed_problems.csv"
with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=["label", "problem"])
    writer.writeheader()
    writer.writerows(data)

print(f"Data has been successfully saved to {csv_file}")
