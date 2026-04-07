from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# 🔥 Smart cleaning function
def clean_dataframe(df):
    df = df.copy()

    for col in df.columns:
        sample = df[col].dropna().astype(str).head(10)

        # Detect date formats
        try:
            parsed = pd.to_datetime(sample, errors='coerce')
            if parsed.notna().sum() > 0.7 * len(sample):
                df[col] = pd.to_datetime(df[col], errors='coerce')
                continue
        except:
            pass

        # Detect ID-like columns
        if df[col].nunique() / len(df) > 0.9:
            df[col] = df[col].astype(str)
            continue

        # Convert numeric safely
        try:
            converted = pd.to_numeric(df[col], errors='coerce')
			
            # Only replace if most values are numeric 
            if converted.notna().sum() > 0.7 * len(df[col]):
                 df[col] = converted
        except:
            pass
            
    return df


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
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

    df = clean_dataframe(df)

    os.makedirs('static', exist_ok=True)
    chart_html = ""

    numeric_cols = [
        col for col in df.select_dtypes(include='number').columns
        if df[col].nunique() > 5
    ]

    categorical_cols = df.select_dtypes(include='object').columns
    date_cols = df.select_dtypes(include='datetime').columns

    # 🔹 Histogram
    for col in numeric_cols[:2]:
        plt.figure()
        df[col].dropna().hist()
        plt.title(f"Distribution of {col}")

        path = f"static/{col}_hist.png"
        plt.savefig(path)
        plt.close()

        chart_html += f"<h3>{col} Distribution</h3><img src='/{path}' width='500'>"

    # 🔹 Bar chart
    if len(categorical_cols) > 0 and len(numeric_cols) > 0:
        cat = categorical_cols[0]
        num = numeric_cols[0]

        grouped = df.groupby(cat)[num].mean().head(10)

        plt.figure()
        grouped.plot(kind='bar')
        plt.title(f"{num} by {cat}")

        path = "static/bar_chart.png"
        plt.savefig(path)
        plt.close()

        chart_html += f"<h3>Bar Chart</h3><img src='/{path}' width='500'>"

    # 🔹 Line chart (time series)
    if len(date_cols) > 0 and len(numeric_cols) > 0:
        date_col = date_cols[0]
        num_col = numeric_cols[0]

        plt.figure()
        df.sort_values(date_col).plot(x=date_col, y=num_col)
        plt.title(f"{num_col} over time")

        path = "static/line_chart.png"
        plt.savefig(path)
        plt.close()

        chart_html += f"<h3>Trend</h3><img src='/{path}' width='500'>"

    summary = df.describe().to_html()

    return f"""
    <h2>Preview</h2>
    {df.head().to_html()}

    <h2>Summary</h2>
    {summary}

    <h2>Charts</h2>
    {chart_html}

    <br><a href="/">Back</a>
    """


if __name__ == '__main__':
    app.run(debug=True)