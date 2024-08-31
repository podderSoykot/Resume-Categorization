import os
import pandas as pd
import pickle
import fitz 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
import shutil
import sys

def load_model_and_vectorizer(model_path, vectorizer_path):
    print(f"Loading model from {model_path}")
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    print(f"Loading vectorizer from {vectorizer_path}")
    with open(vectorizer_path, 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        pdf_document = fitz.open(pdf_path)
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text += page.get_text()
        pdf_document.close()
    except Exception as e:
        print(f"Error reading PDF file {pdf_path}: {e}")
    return text

def categorize_resumes(resume_directory, model, vectorizer):
    categorized_data = []
    files_in_directory = os.listdir(resume_directory)
    print(f"Files in directory: {files_in_directory}")
    
    for file_name in files_in_directory:
        if file_name.endswith('.pdf'):
            file_path = os.path.join(resume_directory, file_name)
            print(f"Processing file: {file_name}")
            try:
                resume_text = extract_text_from_pdf(file_path)
                if not resume_text.strip():
                    print(f"No text found in file {file_name}")
                    continue
                print(f"Resume text (first 500 chars): {resume_text[:500]}")
                resume_vector = vectorizer.transform([resume_text])
                category = model.predict(resume_vector)[0]
                
                category_folder = os.path.join(resume_directory, str(category))
                os.makedirs(category_folder, exist_ok=True)
                shutil.move(file_path, os.path.join(category_folder, file_name))
                categorized_data.append({'filename': file_name, 'category': category})
                print(f"File {file_name} categorized as {category}")
                
            except Exception as e:
                print(f"Error processing file {file_name}: {e}")

    return categorized_data

def main(resume_directory):
    model, vectorizer = load_model_and_vectorizer(
        r'C:\Users\Windows 10 Pro\Desktop\dataset (2)\rf_classifier_model.pkl', 
        r'C:\Users\Windows 10 Pro\Desktop\dataset (2)\tfidf_vectorizer.pkl'
    )
    categorized_data = categorize_resumes(resume_directory, model, vectorizer)
    
    if not categorized_data:
        print("Already categorized / No resumesfound in the Folder.")
    else:
        categorized_df = pd.DataFrame(categorized_data)
        categorized_df.to_csv('categorized_resume.csv', index=False)
        print(f"Resumes have been categorized and saved to 'categorized_resume.csv'.")

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python script.py <resume_directory>")
    else:
        resume_directory = sys.argv[1]
        main(resume_directory)




