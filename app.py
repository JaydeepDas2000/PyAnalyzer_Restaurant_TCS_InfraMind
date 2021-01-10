from flask import Flask, render_template, request, url_for, jsonify
import numpy as np
import pickle
import nltk
nltk.download("vader_lexicon")

app = Flask(__name__)

from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

@app.route('/index.html')
def home():
    return render_template("index.html")

@app.route('/')
def home2():
    return render_template("index.html")

@app.route('/analysis.html')
def analysis():
    return render_template("analysis.html")

@app.route('/analysis.html', methods=['GET', 'POST'])
def predict():
    if request.method == "POST":
        message = request.form['message']
        ps = sia.polarity_scores(message)
        if ps['compound']  > 0:
            result = 'Positive'
        elif ps['compound'] == 0:
            result = 'Neutral'
        else:
            result = 'Negative'
        prob = ps['compound']
        #data = {'Negative' : ps['neg'], 'Positive' : ps['pos'], 'Neutral' : ps['neu']}
        return render_template('analysis.html', predictions=result, proba=prob, mes = message)

if __name__ =="__main__":
    app.run(debug=True)