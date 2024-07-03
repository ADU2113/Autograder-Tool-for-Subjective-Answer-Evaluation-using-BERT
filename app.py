from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_mysqldb import MySQL
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import language_tool_python
from sentence_transformers import SentenceTransformer
from rake_nltk import Rake

app = Flask(__name__)
app.secret_key = 'N9zGom5KCIt3MPvvfg5T6cFIOcU149B4'
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'root'
app.config['MYSQL_DB'] = 'grader'
mysql = MySQL(app)
tool = language_tool_python.LanguageTool('en-US')
rake_nltk_var = Rake()
model = SentenceTransformer('bert-base-nli-mean-tokens')

# Rest of the code remains the same...


def calculate_cosine_similarity(corpus):
    vectorizer = TfidfVectorizer()
    trsfm = vectorizer.fit_transform(corpus)
    score = cosine_similarity(trsfm[0], trsfm)[0][1] * 10
    return round(score, 2)

def stemmer(keywords_list):
    ps = PorterStemmer()
    for i in range(len(keywords_list)):
        keywords_list[i] = ps.stem(keywords_list[i])
    return keywords_list

def lemmatize(keywords_list):
    lemmatizer = WordNetLemmatizer()
    for i in range(len(keywords_list)):
        keywords_list[i] = lemmatizer.lemmatize(keywords_list[i])
    return keywords_list

corpus = []

@app.route('/')
def first():
    return render_template('first.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'username' in session:
        return redirect(url_for('index'))
    if request.method == 'POST':
        name = request.form['name']
        password = request.form['password']
        
        cur = mysql.connection.cursor()
        
        cur.execute("SELECT * FROM users WHERE name = %s", (name,))
        user = cur.fetchone()
        
        print(user)
        
        if user:
            if user[3] == password:
                session['username'] = user[1]
                return redirect(url_for('first'))
            else:
                flash('Incorrect password. Please try again.', 'error')
                return redirect(url_for('login'))
        else:
            flash('User does not exist. Please sign up.', 'error')
            return redirect(url_for('login'))
    
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('username', None)
    # flash('You have been logged out successfully!', 'info')
    return redirect(url_for('first'))


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        
        cur = mysql.connection.cursor()
        
        cur.execute("SELECT * FROM users WHERE name = %s", (name,))
        existing_user = cur.fetchone()
        
        if existing_user:
            return redirect(url_for('signup'))
        else:
            cur.execute("INSERT INTO users (name, email, password) VALUES (%s, %s, %s)", (name, email, password))
            mysql.connection.commit()
            return redirect(url_for('login'))
    
    return render_template('signup.html')

@app.route('/preview', methods=["GET", "POST"])
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        df = pd.read_csv(dataset, encoding='utf-8')
        df.set_index('Id', inplace=True)
        return render_template("preview.html", df_view=df) 
    return redirect(url_for('first'))  # Redirect to the first page if accessed via GET

@app.route('/upload', methods=['GET', 'POST'])  # Allow both GET and POST methods
def upload():
    if request.method == 'POST':
        # Add logic to handle file upload
        return redirect(url_for('success'))  # Redirect to success page after file upload
    return render_template('upload.html')  

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/chart')
def chart():
    return render_template('chart.html')

@app.route('/success', methods=['POST', 'GET'])
def success():
    if request.method=='POST':
        f=None
        f=request.files['file']
        if f == None:
            return render_template('errorredirect.html', message='empty_file')
        fname=f.filename
        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            return render_template('errorredirect.html', message='image_file')
        answer=f.read().decode('utf-8')
        matches = tool.check(answer)
        rake_nltk_var.extract_keywords_from_text(answer)
        keywords_answer_list = rake_nltk_var.get_ranked_phrases()
        f.close()
        with open('reference.txt', encoding='utf-8') as fgt:
            corpus.append(fgt.read())
            correct_answer=corpus[0]
            rake_nltk_var.extract_keywords_from_text(correct_answer)
            keywords_correct_answer_list = rake_nltk_var.get_ranked_phrases()
            fgt.close()
        
        common_keywords = 0
        keywords_answer_list = stemmer(keywords_answer_list)
        keywords_correct_answer_list = stemmer(keywords_correct_answer_list)
        keywords_answer_list = lemmatize(keywords_answer_list)
        keywords_correct_answer_list = lemmatize(keywords_correct_answer_list)
        
        keywords_answer_list_set = set(keywords_answer_list)
        keywords_correct_answer_list_set = set(keywords_correct_answer_list)
        
        for ka in keywords_answer_list_set:
            for kca in keywords_correct_answer_list_set:
                if ka == kca:
                    common_keywords+=1
        
        complete_list = keywords_answer_list + keywords_correct_answer_list
        unique_keywords = len(np.unique(complete_list))
        keywords_match_score = (common_keywords/unique_keywords)*10
        
        corpus.append(answer)
        cosine_sim_score = calculate_cosine_similarity(corpus)
        score=((6/10)*(cosine_sim_score))+((4/10)*(keywords_match_score))
        
        if score >= 10:
            score = 10
        corpus.clear()
        if len(matches)>0:
            score = score - len(matches)
        if score<0:
            score = 0
        print("Errors\t", len(matches))
        print('Cosine_sim_score:\t', cosine_sim_score)
        print('keyword_match_score:\t', keywords_match_score)
        
        return render_template('success.html', name=fname, answer=answer, score=score, correct_answer=correct_answer, matches=len(matches))
    return redirect(url_for('index'))  # Redirect to index page if accessed via GET

if __name__ == '__main__':
    app.run(debug=True)
