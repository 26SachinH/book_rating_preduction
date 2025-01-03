from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
with open('book_rating_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the label encoder for categorical columns
with open('label_encoder.pkl', 'rb') as le_file:
    label_encoder = pickle.load(le_file)

def safe_label_encode(encoder, values):
    """Safely encode labels, assigning -1 to unseen labels."""
    unseen_labels = set(values) - set(encoder.classes_)
    if unseen_labels:
        encoder.classes_ = np.append(encoder.classes_, list(unseen_labels))
    return encoder.transform(values)

@app.route('/')
def home():
    # Render the HTML form
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        book_title = request.form['book_title']
        book_price = request.form['book_price']
        author = request.form['author']
        year_of_publication = request.form['year_of_publication']
        genre = request.form['genre']

        # Prepare the input data for prediction
        current_year = pd.Timestamp.now().year
        book_age = current_year - int(year_of_publication)

        # Create DataFrame for prediction
        input_data = pd.DataFrame({
            'book title': [book_title],
            'book price': [float(book_price)],
            'author': [author],
            'year of publication': [int(year_of_publication)],
            'genre': [genre],
            'book_age': [book_age]
        })

        # Encode categorical data safely
        input_data['book title'] = safe_label_encode(label_encoder, input_data['book title'])
        input_data['author'] = safe_label_encode(label_encoder, input_data['author'])
        input_data['genre'] = safe_label_encode(label_encoder, input_data['genre'])

        # Predict the rating
        predicted_rating = model.predict(input_data)[0]

        return render_template(
            'index.html',
            prediction=f"Predicted Rating: {round(predicted_rating, 2)}",
            book_title=book_title,
            book_price=book_price,
            author=author,
            year_of_publication=year_of_publication,
            genre=genre
        )

    except Exception as e:
        return render_template(
            'index.html',
            prediction=f"Error: {str(e)}",
            book_title=request.form.get('book_title', ''),
            book_price=request.form.get('book_price', ''),
            author=request.form.get('author', ''),
            year_of_publication=request.form.get('year_of_publication', ''),
            genre=request.form.get('genre', '')
        )

if __name__ == '__main__':
    app.run(debug=True)
