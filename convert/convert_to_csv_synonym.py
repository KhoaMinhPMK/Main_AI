import csv
import synonyms_dict

def convert_dict_to_csv(input_dict, csv_file_path):
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Key", "Synonyms"])
        for key, synonyms in input_dict.items():
            writer.writerow([key, ", ".join(synonyms)])

if __name__ == "__main__":
    convert_dict_to_csv(synonyms_dict.synonyms_dict, './af_data/synonyms.csv')
