from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
from retail_analytics import (clean_transactions, summary_metrics, monthly_sales, top_products,
    country_sales, rfm_analysis, train_repeat_purchase_model, run_sql_query)

BASE = Path(__file__).parent
SAMPLE_PATH = BASE / "data" / "sample_retail_transactions.csv"
st.set_page_config(page_title="Retail Intelligence | Sales & Customers", page_icon="🛍️", layout="wide")

st.markdown("""<style>
.block-container{padding-top:1.6rem;padding-bottom:2rem;max-width:1450px}
[data-testid="stMetric"]{background:#f6f8f5;border:1px solid #e2e8e0;padding:14px 16px;border-radius:12px}
.hero{padding:1.25rem 1.4rem;border-radius:16px;background:linear-gradient(120deg,#102c26,#285843);color:white;margin-bottom:1.1rem}
.hero h1{font-size:2.05rem;margin:0 0 .3rem 0}.hero p{margin:0;color:#e0eadf}
.note{font-size:.88rem;color:#657269}
</style>""", unsafe_allow_html=True)

st.sidebar.markdown("# 🛍️ Retail Intelligence")
st.sidebar.caption("Sales analytics • Customer intelligence • Machine learning")
source = st.sidebar.radio("Data source", ["Bundled demo data", "Upload my dataset"], help="Demo data is synthetic and labelled. Upload the UCI Online Retail Excel file or a compatible CSV/XLSX.")
uploaded = None
if source == "Upload my dataset":
    uploaded = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
    st.sidebar.caption("Expected: invoice ID, date, quantity, unit price. Customer, product and country fields improve the analysis.")

@st.cache_data(show_spinner=False)
def read_and_clean(file_bytes: bytes | None, filename: str | None, demo: bool):
    if demo:
        raw = pd.read_csv(SAMPLE_PATH, dtype={"CustomerID":"string", "InvoiceNo":"string"})
    else:
        import io
        if filename and filename.lower().endswith(".csv"):
            raw = pd.read_csv(io.BytesIO(file_bytes), dtype={"CustomerID":"string", "InvoiceNo":"string"})
        else:
            raw = pd.read_excel(io.BytesIO(file_bytes), dtype={"CustomerID":"string", "InvoiceNo":"string"})
    cleaned, report = clean_transactions(raw)
    return cleaned, report

try:
    if source == "Upload my dataset" and uploaded is None:
        st.markdown('<div class="hero"><h1>Retail Sales Analytics & Customer Intelligence</h1><p>Turn transaction data into business decisions.</p></div>', unsafe_allow_html=True)
        st.info("Upload a CSV or Excel file in the sidebar to begin. To explore the app first, choose **Bundled demo data**.")
        st.stop()
    data_bytes = uploaded.getvalue() if uploaded else None
    df, clean_report = read_and_clean(data_bytes, uploaded.name if uploaded else None, source == "Bundled demo data")
except Exception as exc:
    st.error(f"Could not load this dataset: {exc}")
    st.info("Check the file columns, date formats, quantity and price values. You can use the bundled demo dataset as a reference.")
    st.stop()

metrics = summary_metrics(df)
is_demo = source == "Bundled demo data"
st.markdown('<div class="hero"><h1>Retail Sales Analytics & Customer Intelligence</h1><p>Understand revenue, identify valuable customer groups, and estimate repeat-purchase propensity.</p></div>', unsafe_allow_html=True)
if is_demo:
    st.warning("DEMO MODE: the bundled transactions are synthetic, generated for demonstration. They are not actual company sales and must not be presented as real business results.")
else:
    st.success(f"Loaded your dataset: {uploaded.name}")

st.caption(f"Data window: {metrics['date_min']:%d %b %Y} – {metrics['date_max']:%d %b %Y}  •  {clean_report['clean_rows']:,} valid transaction lines  •  {clean_report['invalid_or_removed_rows']:,} rows removed during cleaning")

with st.sidebar.expander("Data quality report", expanded=False):
    for label, key in [("Raw rows", "raw_rows"), ("Clean rows", "clean_rows"), ("Cancellation / return rows removed", "cancellation_rows"), ("Invalid or removed rows", "invalid_or_removed_rows"), ("Rows without customer ID", "rows_without_customer_id")]:
        st.write(f"**{label}:** {clean_report.get(key, 0):,}")

nav = st.sidebar.radio("Explore", ["Executive overview", "Sales analysis", "Customer intelligence", "Repeat-purchase model", "SQL explorer"])

if nav == "Executive overview":
    st.subheader("Business overview")
    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Net sales revenue", f"£{metrics['revenue']:,.0f}")
    k2.metric("Orders", f"{metrics['orders']:,}")
    k3.metric("Known customers", f"{metrics['customers']:,}")
    k4.metric("Average order value", f"£{metrics['aov']:,.2f}")
    k5.metric("Units sold", f"{metrics['units']:,.0f}")
    month = monthly_sales(df)
    left,right = st.columns([1.45,1])
    with left:
        st.markdown("#### Monthly revenue trend")
        if len(month):
            fig = px.line(month, x="InvoiceMonth", y="revenue", markers=True, labels={"InvoiceMonth":"Month","revenue":"Revenue (£)"})
            fig.update_layout(margin=dict(l=10,r=10,t=20,b=10), height=340)
            st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("#### Revenue by country")
        countries = country_sales(df, 8)
        fig = px.bar(countries.sort_values("revenue"), x="revenue", y="Country", orientation="h", labels={"revenue":"Revenue (£)"})
        fig.update_layout(margin=dict(l=10,r=10,t=20,b=10), height=340, yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("#### Quick readout")
    if len(month) >= 2:
        first, last = float(month.iloc[0]["revenue"]), float(month.iloc[-1]["revenue"])
        change = ((last-first)/abs(first)*100) if first else None
        st.write(f"The first observed month recorded **£{first:,.0f}**, while the latest observed month recorded **£{last:,.0f}**. Monthly change across those endpoints: **{change:+.1f}%**." if change is not None else "Revenue is available, but the first month has zero revenue so endpoint percentage change is undefined.")
    st.write("Use the other sections to inspect product performance, customer segments, and the repeat-purchase model. These summaries are descriptive, not proof of causality.")

elif nav == "Sales analysis":
    st.subheader("Sales analysis")
    month = monthly_sales(df)
    left,right = st.columns(2)
    with left:
        st.markdown("#### Revenue by month")
        fig=px.bar(month,x="InvoiceMonth",y="revenue",labels={"InvoiceMonth":"Month","revenue":"Revenue (£)"})
        fig.update_layout(margin=dict(l=10,r=10,t=15,b=10),height=350)
        st.plotly_chart(fig,use_container_width=True)
    with right:
        st.markdown("#### Orders and units by month")
        fig=px.line(month,x="InvoiceMonth",y="orders",markers=True,labels={"InvoiceMonth":"Month","orders":"Orders"})
        fig.update_layout(margin=dict(l=10,r=10,t=15,b=10),height=350)
        st.plotly_chart(fig,use_container_width=True)
    st.markdown("#### Top products by sales revenue")
    products = top_products(df, 15)
    fig=px.bar(products.sort_values("revenue"),x="revenue",y="Description",orientation="h",hover_data=["StockCode","units","orders"],labels={"revenue":"Revenue (£)","Description":"Product"})
    fig.update_layout(margin=dict(l=10,r=10,t=15,b=10),height=470,yaxis_title="")
    st.plotly_chart(fig,use_container_width=True)
    st.dataframe(products, use_container_width=True, hide_index=True)
    st.download_button("Download top-products CSV", products.to_csv(index=False).encode("utf-8"), "top_products.csv", "text/csv")
    st.markdown("#### Country performance")
    cs=country_sales(df,30)
    st.dataframe(cs, use_container_width=True, hide_index=True)

elif nav == "Customer intelligence":
    st.subheader("RFM customer segmentation")
    st.write("RFM means **Recency** (days since last purchase), **Frequency** (distinct orders), and **Monetary** (total spend). Higher R/F/M scores indicate stronger recent engagement and value, using this dataset's relative quintiles.")
    rfm=rfm_analysis(df)
    if rfm.empty:
        st.warning("No valid customer IDs found. Upload a dataset that includes customer identifiers to use segmentation.")
    else:
        counts=rfm["Segment"].value_counts().rename_axis("Segment").reset_index(name="Customers")
        left,right=st.columns([1,1.25])
        with left:
            fig=px.pie(counts,values="Customers",names="Segment",hole=.5)
            fig.update_layout(margin=dict(l=5,r=5,t=15,b=5),height=350,legend_title_text="Segment")
            st.plotly_chart(fig,use_container_width=True)
        with right:
            byseg=rfm.groupby("Segment",as_index=False).agg(Customers=("CustomerID","count"),Revenue=("Monetary","sum"),AvgOrders=("Frequency","mean"))
            st.dataframe(byseg.sort_values("Revenue",ascending=False).round(2),use_container_width=True,hide_index=True)
        segment=st.selectbox("Filter customer segment",["All segments"]+sorted(rfm["Segment"].unique().tolist()))
        filtered=rfm if segment=="All segments" else rfm[rfm["Segment"]==segment]
        st.dataframe(filtered[["CustomerID","RecencyDays","Frequency","Monetary","R_score","F_score","M_score","RFM_score","Segment"]].round(2),use_container_width=True,hide_index=True)
        st.download_button("Download RFM customer segments",filtered.to_csv(index=False).encode("utf-8"),"customer_rfm_segments.csv","text/csv")
        st.caption("Segments are heuristic labels based on relative RFM scores; they are not externally validated customer categories.")

elif nav == "Repeat-purchase model":
    st.subheader("30-day repeat-purchase propensity")
    st.write("A logistic regression model estimates whether a customer will purchase again in the next 30 days based on historical customer snapshots. Features are built before each prediction window; evaluation uses a chronological holdout.")
    st.caption("This is an analytical estimate, not a guarantee that any individual customer will purchase.")
    with st.spinner("Building historical customer snapshots and evaluating the model…"):
        model_result = train_repeat_purchase_model(df)
    if not model_result.get("available"):
        st.warning(model_result.get("reason", "Model cannot be evaluated for this dataset."))
        st.info("For reliable time-based evaluation, provide enough transaction history, ideally spanning at least six months, with customer IDs.")
    else:
        mm=model_result["metrics"]
        a,b,c,d,e=st.columns(5)
        a.metric("ROC-AUC", f"{mm['roc_auc']:.3f}" if mm["roc_auc"] is not None else "N/A")
        b.metric("Accuracy",f"{mm['accuracy']:.3f}")
        c.metric("Precision",f"{mm['precision']:.3f}")
        d.metric("Recall",f"{mm['recall']:.3f}")
        e.metric("F1 score",f"{mm['f1']:.3f}")
        st.caption(f"Time-based split • {mm['train_rows']:,} training snapshots • {mm['test_rows']:,} holdout snapshots • holdout repeat-purchase rate {mm['test_positive_rate']:.1%}")
        st.markdown("#### Model coefficients")
        coef=model_result["feature_coefficients"].sort_values("Coefficient")
        fig=px.bar(coef,x="Coefficient",y="Feature",orientation="h",labels={"Coefficient":"Standardized logistic-regression coefficient"})
        fig.update_layout(margin=dict(l=10,r=10,t=10,b=10),height=320,yaxis_title="")
        st.plotly_chart(fig,use_container_width=True)
        st.markdown("#### Customers with highest estimated repeat-purchase probability")
        preds=model_result["predictions"].copy()
        preds["repeat_probability"]=(preds["repeat_probability"]*100).round(1)
        preds=preds.rename(columns={"repeat_probability":"Repeat probability (%)"})
        st.dataframe(preds[["CustomerID","Repeat probability (%)","RiskBand","recency_days","orders_120d","spend_120d"]].head(100),use_container_width=True,hide_index=True)
        st.download_button("Download customer propensity scores",preds.to_csv(index=False).encode("utf-8"),"repeat_purchase_propensity.csv","text/csv")
        with st.expander("Confusion matrix and evaluation notes"):
            st.write("Rows are actual outcomes (0 = no repeat purchase, 1 = repeat purchase); columns are predicted outcomes (0/1).")
            st.dataframe(pd.DataFrame(model_result["confusion_matrix"],index=["Actual 0","Actual 1"],columns=["Predicted 0","Predicted 1"]),use_container_width=True)
            st.write("The model uses a single chronological holdout but multiple customer snapshots may belong to the same customer across earlier cutoffs. Treat metrics as an initial portfolio benchmark, not production validation. Before deploying for business decisions, add customer-level validation, calibration, leakage checks and monitoring.")

elif nav == "SQL explorer":
    st.subheader("SQL business questions")
    st.write("Run read-only SQL against the cleaned transaction lines in a temporary SQLite database. Table: `transactions`.")
    queries={
        "Monthly revenue": "SELECT InvoiceMonth, ROUND(SUM(Revenue), 2) AS Revenue, COUNT(DISTINCT InvoiceNo) AS Orders FROM transactions GROUP BY InvoiceMonth ORDER BY InvoiceMonth;",
        "Top 10 products": "SELECT StockCode, Description, ROUND(SUM(Revenue), 2) AS Revenue, SUM(Quantity) AS Units FROM transactions GROUP BY StockCode, Description ORDER BY Revenue DESC LIMIT 10;",
        "Top countries": "SELECT Country, ROUND(SUM(Revenue), 2) AS Revenue, COUNT(DISTINCT InvoiceNo) AS Orders, COUNT(DISTINCT CustomerID) AS Customers FROM transactions GROUP BY Country ORDER BY Revenue DESC LIMIT 10;",
        "Customer lifetime value (historical)": "SELECT CustomerID, ROUND(SUM(Revenue), 2) AS HistoricalSpend, COUNT(DISTINCT InvoiceNo) AS Orders, MAX(InvoiceDate) AS LastPurchase FROM transactions WHERE CustomerID IS NOT NULL GROUP BY CustomerID ORDER BY HistoricalSpend DESC LIMIT 25;",
        "Average order value by month": "WITH orders AS (SELECT InvoiceMonth, InvoiceNo, SUM(Revenue) AS OrderRevenue FROM transactions GROUP BY InvoiceMonth, InvoiceNo) SELECT InvoiceMonth, ROUND(AVG(OrderRevenue), 2) AS AverageOrderValue FROM orders GROUP BY InvoiceMonth ORDER BY InvoiceMonth;",
    }
    choice=st.selectbox("Choose a query",list(queries.keys()))
    query=st.text_area("SQL (read-only)",queries[choice],height=125)
    if st.button("Run SQL query",type="primary"):
        try:
            result=run_sql_query(df,query)
            st.success(f"Query returned {len(result):,} row(s).")
            st.dataframe(result,use_container_width=True,hide_index=True)
            st.download_button("Download query results CSV",result.to_csv(index=False).encode("utf-8"),"sql_query_results.csv","text/csv")
        except Exception as exc:
            st.error(f"SQL error: {exc}")
    st.caption("The SQL explorer accepts SELECT and WITH statements only; the temporary in-memory database is discarded after the query.")

st.divider()
st.caption("Retail Sales Analytics & Customer Intelligence • Python · SQL · Pandas · Scikit-learn · Plotly · Streamlit")
