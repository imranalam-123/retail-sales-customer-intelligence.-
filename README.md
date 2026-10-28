# Retail Sales Analytics & Customer Intelligence

An end-to-end portfolio project for retail transaction analysis. It combines a Streamlit dashboard, data cleaning, business SQL, RFM customer segmentation, and a 30-day repeat-purchase propensity model evaluated with a chronological holdout.

## Features

- **Executive overview:** revenue, orders, known customers, average order value, units, monthly trend and country contribution.
- **Sales analysis:** monthly revenue, order trends, top products and country performance.
- **Customer intelligence:** Recency-Frequency-Monetary (RFM) scores and customer segments with downloadable CSV output.
- **Machine learning:** point-in-time customer snapshots and logistic regression estimating repeat purchase in the next 30 days.
- **SQL explorer:** curated read-only queries over the cleaned transaction data.
- **Data quality report:** raw row counts, cancellations/invalid rows removed and missing customer IDs.
- **Bring your own data:** upload a compatible CSV, XLSX or XLS file.

## Important: demo data is synthetic

The bundled sample consists of synthetic transaction records generated for demonstration. It is **not** real company data and its results must not be presented as real business performance. To demonstrate analysis on real data, download the UCI Online Retail dataset and upload its Excel file.

## Run locally

Python 3.10+ is recommended.

\`\`\`bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
\`\`\`

## Real dataset source

UCI Machine Learning Repository — Online Retail: https://archive.ics.uci.edu/dataset/352/online+retail

Expected fields include \`InvoiceNo\`, \`StockCode\`, \`Description\`, \`Quantity\`, \`InvoiceDate\`, \`UnitPrice\`, \`CustomerID\`, and \`Country\`. Common alternative names for required fields are also normalized.

## SQL files

The \`sql/\` directory contains examples for monthly performance, best-selling products, historical customer value, country performance and monthly average order value. In the app, the SQL explorer uses a temporary SQLite database with a table named \`transactions\`.

## Deployment to Streamlit Community Cloud

This project is stored as a folder in the connected GitHub repository. In [Streamlit Community Cloud](https://share.streamlit.io/), create an app using repository \`imranalam-123/booknest\`, branch \`main\`, and main file path:

\`retail-sales-customer-intelligence/app.py\`

Streamlit will install dependencies from the \`requirements.txt\` in this project folder.

## Model design and limitations

- Features are built from transactions available before each snapshot cutoff.
- Label: whether a customer buys in the following 30-day horizon.
- Evaluation uses a chronological holdout to better reflect forward-looking use than a random split.
- Multiple historical snapshots may belong to the same customer. Treat metrics as an initial portfolio benchmark, not production validation.
- RFM segment names are heuristic and relative to the loaded dataset.
- The app displays GBP (£) to match the UCI Online Retail example; change labels if your data uses another currency.

## Resume guidance

Describe only work you have actually run and verified. Example:

> Built an interactive retail analytics app using Python, SQL and Streamlit; implemented transaction cleaning, monthly sales analysis, RFM customer segmentation and a time-based repeat-purchase classifier with holdout evaluation.

Add measured model metrics or business findings only after running the app on a clearly identified dataset. Do not present synthetic demo results as actual business results.
