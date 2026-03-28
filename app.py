from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/upload',methods=['POST'])
def upload():
	file = request.files['file']

	if not file:
		return "No file uploaded"

	filename = file.filename

	# Handle CSV and Excel
	if filename.endswith('.csv'):
		df = pd.read_csv(file)
	elif filename.endswith('.xlsx'):
		df = pd.read_excel(file)
	else:
		return "Unsupported file format"

	summary = df.describe().to_html()

	return f"""
	<h2>Preview</h2>
	{df.head().to_html()}

	<h2>Summary</h2>
	{summary}
	
	<br><a href="/">Back</a>
	"""

if __name__ == '__main__':
	app.run(debug=True)
