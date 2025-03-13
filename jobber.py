from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import os
import nltk
from nltk.stem import PorterStemmer
import joblib
import numpy as np
from gensim.models import FastText

# Initialize Flask app
app = Flask(__name__)


# Download and initialize necessary NLTK resources
try:
    nltk.download('punkt')
    stemmer = PorterStemmer()
except Exception as e:
    print(f"Failed to initialize NLTK resources: {e}")

# Paths to the pre-trained models (using relative paths)
fasttext_model_path = os.path.join(os.path.dirname(__file__), 'models', 'desc_FT.model')
logistic_model_path = os.path.join(os.path.dirname(__file__), 'models', 'descFT_LR.pkl')

# Load pre-trained models for category recommendation
fasttext_model = FastText.load(fasttext_model_path)
logistic_model = joblib.load(logistic_model_path)

# Path to the data directory
data_path = os.path.join(os.path.dirname(__file__), 'data')

# Function to read job files
def read_job_files():
    jobs = {}
    specific_path = r"C:\Users\user\OneDrive\Desktop\ap2\data\Accounting_Finance"
    
    # Traverse through the specific directory to find job files
    for root, dirs, files in os.walk(specific_path):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    content = f.read()
                    # Extract job title and description from the file
                    title_line = next((line for line in content.split('\n') if line.startswith("Title:")), None)
                    title = title_line.replace("Title:", "").strip() if title_line else file
                    description_line = next((line for line in content.split('\n') if line.startswith("Description:")), None)
                    description = description_line.replace("Description:", "").strip() if description_line else "Description not available"
                    jobs[title] = description
    return jobs

# Function to search jobs with stemming in job titles
def search_jobs(keyword):
    jobs = read_job_files()
    stemmed_keyword = stemmer.stem(keyword.lower())
    # Search only in job titles
    result = {title: description for title, description in jobs.items() if stemmed_keyword in [stemmer.stem(word) for word in title.lower().split()]}
    return result

# Function to create a new job listing and save it as a text file in the appropriate subdirectory
def save_job_to_file(job):
    specific_path = r"C:\Users\user\OneDrive\Desktop\ap2\data\Accounting_Finance"
    
    # Create directory if it does not exist
    if not os.path.exists(specific_path):
        os.makedirs(specific_path)
    
    # Generate a new job file ID
    job_id = len(os.listdir(specific_path)) + 1  # Assuming all files in specific_path are job files
    file_name = f"job_{job_id}.txt"
    file_path = os.path.join(specific_path, file_name)
    
    # Write job details to the file
    with open(file_path, 'w') as file:
        file.write(f"Title: {job['title']}\n")
        file.write(f"Company: {job['company']}\n")
        file.write(f"Description: {job['description']}\n")
        file.write(f"Salary: {job['salary']}\n")
        file.write(f"Category: {job['category']}\n")

# Home route
@app.route('/')
def home():
    return render_template('landing_page.html')

# Why Us route
@app.route('/whyus')
def why_us():
    return render_template('why_us.html')

# Career Advice route
@app.route('/careeradvice')
def career_advice():
    return render_template('career_advice.html')

# Job Search route
@app.route('/jobsearch', methods=['GET', 'POST'])
def job_search():
    if request.method == 'POST':
        keyword = request.form['keyword']
        results = search_jobs(keyword)
        return render_template('search_results.html', keyword=keyword, results=results)
    return render_template('job_search.html')

# Create Job route
@app.route('/postopportunities', methods=['GET', 'POST'])
def create_job_route():
    if request.method == 'POST':
        # Extract data from the form fields
        title = request.form['title']
        company = request.form['company']
        description = request.form['description']
        salary = request.form.get('salary', 'Not Provided')
        category = request.form.get('category', '')  # Get the category if provided
        
        new_job = {
            'title': title,
            'company': company,
            'description': description,
            'salary': salary,
            'category': category
        }
        
        # Predict category using the ML model if not manually specified
        if not category:
            job_text = title + " " + description
            job_text_tokens = nltk.word_tokenize(job_text.lower())  # Tokenize text
            job_vector = np.mean([fasttext_model.wv[word] for word in job_text_tokens if word in fasttext_model.wv], axis=0).reshape(1, -1)
            category = logistic_model.predict(job_vector)[0]
            new_job['category'] = category
        
        # Save the new job to the appropriate subdirectory
        save_job_to_file(new_job)
        flash('Job posted successfully!', 'success')
        return redirect('/')
        
    return render_template('create_job.html')

# Predict Category route
@app.route('/predict_category', methods=['POST'])
def predict_category():
    data = request.json
    title = data.get('title', '')
    description = data.get('description', '')
    job_text = title + " " + description
    job_text_tokens = nltk.word_tokenize(job_text.lower())
    job_vector = np.mean([fasttext_model.wv[word] for word in job_text_tokens if word in fasttext_model.wv], axis=0).reshape(1, -1)
    predicted_category = logistic_model.predict(job_vector)[0]
    return jsonify({'category': predicted_category})

# Main entry point
if __name__ == '__main__':
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    app.run(debug=True, port=5001)
