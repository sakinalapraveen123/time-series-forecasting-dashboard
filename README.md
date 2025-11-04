📈 Time Series Forecasting & EDA Dashboard
Live Demo: https://tsf-dashboard.streamlit.app/

🚩 Problem
Forecasting and exploring time series data is crucial in business, energy, healthcare, and science—but most people rely on code-heavy tools or paid software, limiting access for analysts, students, and business teams.

💡 Solution
This dashboard lets anyone upload their own time series dataset, automatically analyze it, and generate future forecasts—all without writing code. Just:

Upload your file

Select columns (date & value)

Instantly see data summaries, visualizations, and model predictions

Download results for business or study

✨ Features
Flexible Uploads: Accepts .csv, .xlsx, .parquet files.

Auto Data Profiling: Missing value counts, column types, unique values, summary stats.

Visual EDA: Histograms, autocorrelation, and top unique values.

ARIMA Forecasting: One-click modeling for any future horizon.

Interactive Charts: Actual vs predicted lines, annotated forecast period.

Downloadable Results: Export predictions and analyses.

No Installation Needed: Runs instantly via Streamlit Cloud.

📊 Sample Datasets for Demo
Quickly try the dashboard with these open time series datasets (right-click, "Save link as..." to download):

Airline Passengers (monthly) CSV

Daily Min Temperatures (Melbourne) CSV

Sunspots (monthly) CSV

Monthly Retail Sales (US) CSV

Hourly Energy Consumption (USA, from Kaggle) (Kaggle account required)

You can use these in any format supported by the dashboard: .csv, .xlsx, .parquet.

🚀 Usage
Clone or launch online:

bash
git clone https://github.com/yourusername/yourrepo.git
cd yourrepo
pip install -r requirements.txt
streamlit run app.py
Or use the live app.

Workflow:

Upload your time series file

Choose your date and value columns

Explore auto-generated EDA (summary, stats, histograms, autocorrelation)

Use the ARIMA section to forecast into the future, view visuals/metrics, and download predictions

🏆 Why is this dashboard special?
All-in-one: combines advanced EDA and classic forecasting with zero code.

Democratizes forecasting: anyone can use, no matter their technical skills.

Open source, fully extensible, and ready for any time series use case.

🧰 Technologies Used
Streamlit

pandas, numpy

plotly, matplotlib

statsmodels, scikit-learn

📚 Skills Demonstrated
Full-stack Python & app development

Automated classic time series (ARIMA) modeling

Data visualization and UI for non-coders

End-to-end deployment and documentation

🤝 Contributing
Pull requests welcome! If you have suggestions for new features, file an issue or reach out.

©️ License
MIT — free for personal or commercial use.

Questions / feedback / feature ideas?
Open an issue or connect on LinkedIn!

The app will open automatically at http://localhost:8501 when running locally.
