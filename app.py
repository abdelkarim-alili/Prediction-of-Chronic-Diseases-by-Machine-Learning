from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    name = request.form['fullName']
    age = request.form['age']
    return f"Thank you, {name}! We have received your age: {age}."

if __name__ == '__main__':
    app.run(debug=True)
