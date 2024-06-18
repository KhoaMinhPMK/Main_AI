import os
import random
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, TFAutoModel
from tensorflow.keras.layers import Input, Dense, Dropout, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import pandas as pd
import numpy as np

# Chỉ định sử dụng cả 2 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# Chiến lược phân phối
strategy = tf.distribute.MirroredStrategy()

print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

def load_synonyms_from_csv(csv_file_path):
    df = pd.read_csv(csv_file_path, encoding='utf-8')
    synonyms_dict = {}
    for index, row in df.iterrows():
        key = row['Key']
        synonyms = row['Synonyms'].split(', ')
        synonyms_dict[key] = synonyms
    return synonyms_dict

synonyms_dict = load_synonyms_from_csv('/kaggle/input/math-data2/synonyms.csv')

def synonym_replacement(sentence, n, synonyms_dict):
    words = sentence.split()
    new_words = words.copy()
    random_word_list = list(set(words))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = synonyms_dict.get(random_word, [])
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break
    sentence = ' '.join(new_words)
    return sentence

def random_insertion(sentence, n, synonyms_dict):
    words = sentence.split()
    for _ in range(n):
        new_synonym = synonyms_dict.get(random.choice(words), [])
        if len(new_synonym) > 0:
            words.insert(random.randint(0, len(words)), random.choice(new_synonym))
    return ' '.join(words)

def random_swap(sentence, n):
    words = sentence.split()
    if len(words) < 2:
        return sentence
    for _ in range(n):
        idx1, idx2 = random.sample(range(len(words)), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
    return ' '.join(words)

def random_deletion(sentence, p):
    words = sentence.split()
    if len(words) == 1:
        return sentence
    new_words = [word for word in words if random.uniform(0, 1) > p]
    if len(new_words) == 0:
        return random.choice(words)
    return ' '.join(new_words)

def augment_data(sentences, labels, num_augmented=2, synonyms_dict=None):
    augmented_sentences = []
    augmented_labels = []
    for sentence, label in zip(sentences, labels):
        augmented_sentences.append(sentence)
        augmented_labels.append(label)
        for _ in range(num_augmented):
            aug_sentence = sentence
            aug_sentence = synonym_replacement(aug_sentence, n=2, synonyms_dict=synonyms_dict)
            aug_sentence = random_insertion(aug_sentence, n=1, synonyms_dict=synonyms_dict)
            aug_sentence = random_swap(aug_sentence, n=1)
            aug_sentence = random_deletion(aug_sentence, p=0.1)
            augmented_sentences.append(aug_sentence)
            augmented_labels.append(label)
    return augmented_sentences, augmented_labels

# lap_he_phuong_trinh_df = pd.read_csv('/kaggle/input/math-data2/lap_he_phuong_trinh.csv', encoding='utf-8')
# phuong_trinh_df = pd.read_csv('/kaggle/input/math-data2/phuong_trinh.csv', encoding='utf-8')
# ti_le_phan_tram_df = pd.read_csv('/kaggle/input/math-data2/ti_le_phan_tram.csv', encoding='utf-8')
# chuyen_dong_nem_df = pd.read_csv('/kaggle/input/math-data2/chuyen_dong_nem.csv', encoding='utf-8')
# chuyen_dong_thang_df = pd.read_csv('/kaggle/input/math-data2/chuyen_dong_thang.csv', encoding='utf-8')
bien_doi_deu_df = pd.read_csv('/kaggle/input/physics/bien_doi_deu.csv', encoding='utf-8')
chuyen_dong_nem_df = pd.read_csv('/kaggle/input/physics/chuyen_dong_nem.csv', encoding='utf-8')
roi_tu_do_df = pd.read_csv('/kaggle/input/physics/roi_tu_do.csv', encoding='utf-8')
thang_deu_df = pd.read_csv('/kaggle/input/physics/thang_deu.csv', encoding='utf-8')
df = pd.concat([thang_deu_df, roi_tu_do_df, chuyen_dong_nem_df, bien_doi_deu_df], ignore_index=True)

problems = df['problem'].tolist()
labels = df['label'].tolist()

augmented_problems, augmented_labels = augment_data(problems, labels, synonyms_dict=synonyms_dict)

label_dict = {label: i for i, label in enumerate(set(augmented_labels))}
y = to_categorical([label_dict[label] for label in augmented_labels], num_classes=len(set(labels)))

X_train, X_val, y_train, y_val = train_test_split(augmented_problems, y, test_size=0.2, random_state=42)

phobert_model_name = "vinai/phobert-base"
tokenizer = AutoTokenizer.from_pretrained(phobert_model_name)

def encode_texts(texts, tokenizer, max_len):
    encodings = tokenizer(texts, truncation=True, padding='max_length', max_length=max_len, return_tensors='tf')
    return encodings['input_ids'], encodings['attention_mask']

max_len = 128
X_train_ids, X_train_masks = encode_texts(X_train, tokenizer, max_len)
X_val_ids, X_val_masks = encode_texts(X_val, tokenizer, max_len)

phobert_model = TFAutoModel.from_pretrained(phobert_model_name)

class PhoBERTLayer(Layer):
    def __init__(self, phobert_model, **kwargs):
        super(PhoBERTLayer, self).__init__(**kwargs)
        self.phobert_model = phobert_model
    
    def call(self, inputs):
        input_ids, attention_mask = inputs
        outputs = self.phobert_model(input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "phobert_model": self.phobert_model
        })
        return config

with strategy.scope():
    input_ids_layer = Input(shape=(max_len,), dtype=tf.int32, name='input_ids')
    attention_masks_layer = Input(shape=(max_len,), dtype=tf.int32, name='attention_mask')

    phobert_output = PhoBERTLayer(phobert_model)([input_ids_layer, attention_masks_layer])
    dense_layer = Dense(64, activation='relu', kernel_regularizer=l2(1e-5))(phobert_output[:, 0, :])
    dropout_layer = Dropout(0.5)(dense_layer)
    output_layer = Dense(len(label_dict), activation='softmax')(dropout_layer)

    model = Model(inputs=[input_ids_layer, attention_masks_layer], outputs=output_layer)
    optimizer = Adam(learning_rate=1e-5)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('best_phobert_model.keras', save_best_only=True, monitor='val_accuracy', mode='max')

    for layer in phobert_model.layers:
        layer.trainable = False

    history = model.fit(
        {'input_ids': X_train_ids, 'attention_mask': X_train_masks}, y_train,
        validation_data=({'input_ids': X_val_ids, 'attention_mask': X_val_masks}, y_val),
        epochs=100,
        batch_size=35,
        callbacks=[early_stopping, model_checkpoint]
    )

    for layer in phobert_model.layers:
        layer.trainable = True

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    history = model.fit(
        {'input_ids': X_train_ids, 'attention_mask': X_train_masks}, y_train,
        validation_data=({'input_ids': X_val_ids, 'attention_mask': X_val_masks}, y_val),
        epochs=70,
        batch_size=35,
        callbacks=[early_stopping, model_checkpoint]
    )

# Chỉnh sửa phần cuối của script chính

import json

# Lưu mô hình
model.save('physics_model.keras')

# Lưu từ điển nhãn
with open('physics_label_dict.json', 'w') as f:
    json.dump(label_dict, f)

def predict_problem(problem):
    encoding = tokenizer(problem, truncation=True, padding='max_length', max_length=max_len, return_tensors='tf')
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    prediction = model.predict({'input_ids': input_ids, 'attention_mask': attention_mask})
    predicted_label = np.argmax(prediction, axis=1)[0]
    return list(label_dict.keys())[list(label_dict.values()).index(predicted_label)]

while True:
    test_problem = input("Nhập bài toán (hoặc 'stop' để kết thúc): ")

    if test_problem.lower() == 'stop':
        print("Kết thúc chương trình.")
        break

    predicted_label = predict_problem(test_problem)
    print(f"Nhãn dự đoán cho bài toán '{test_problem}' là: {predicted_label}")

