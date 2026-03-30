from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import os

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

	# Auto chart (first numeric Column)
	numeric_cols = df.select_dtypes(include='number').columns

	if len(numeric_cols)>0:
		col = numeric_cols[0]

		plt.figure()
		df[col].plot(kind='line')
		plt.title(f"{col} Trend")

		os.makedirs('static', exist_ok=True)
		chart_path = 'static/chart.png'
		plt.savefig(chart_path)
		plt.close()
	else:
		chart_path = None

	summary = df.describe().to_html()

	return f"""
	<h2>Preview</h2>
	{df.head().to_html()}

	<h2>Summary</h2>
	{summary}

	<h2>Chart</h2>
	{"<img src='/static/chart.png'>" if chart_path else "No numeric data"}
	
	<br><a href="/">Back</a>
	"""

if __name__ == '__main__':
	app.run(debug=True)
