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

	#  Improved chart generation
	
	os.makedirs('static', exist_ok=True)
	
	chart_html =""
	numeric_cols = df.select_dtypes(include='number').columns
	categorical_cols = df.select_dtypes(include='object').columns
	
	# 1. Histogram for numeric Column
	for col in numeric_cols[:2]:
		plt.figure()
		df[col].dropna().hist()
		plt.title(f"Distribution of {col}")

		path = f"static/{col}_hist.png"
		plt.savefig(path)
		plt.close()
		
		chart_html +=f"<h3>{col} Distribution</h3><img src='/{path}' width='500'>"

	# 2. Bar chart for categorial + numeric
	if len(categorical_cols) > 0 and len(numeric_cols) > 0:
		cat = categorical_cols[0]
		num = numeric_cols[0]

		grouped = df.groupby(cat)[num].mean().head(10)

		plt.figure()
		grouped.plot(kind='bar')
		plt.title(f"{num} by {cat}")

		path = f"static/bar_chart.png"
		plt.savefig(path)
		plt.close()

		chart_html += f"<h3>Bar Chart</h3><img src='/{path}' width='500'>"

	# 3. Line chart if time-like column exists
	for col in df.columns:
		if 'date' in col.lower():
			df[col] = pd.to_datetime(df[col, errors='coerce')
			
			num = numeric_cols[0]

			plt.figure()
			df.sort_values(col).plot(x=col, y=num)
			plt.title(f"{num} over time")
			
			path = f"static/line_chart.png"
			plt.savefig(path)
			plt.close()
			
			chart_html += f"<h3>Trend</h3><img src='/{path}' width='500'>"
			break

	summary = df.describe().to_html()

	return f"""
	<h2>Preview</h2>
	{df.head().to_html()}

	<h2>Summary</h2>
	{summary}

	<h2>Chart</h2>
	{chart_html}
	
	<br><a href="/">Back</a>
	"""

if __name__ == '__main__':
	app.run(debug=True)
