import unittest
import pandas as pd
from retail_analytics import clean_transactions, summary_metrics, rfm_analysis, monthly_sales, run_sql_query, build_customer_snapshots

class AnalyticsTests(unittest.TestCase):
    def setUp(self):
        self.raw = pd.DataFrame({
            "InvoiceNo": ["1001", "1001", "1002", "C1003", "1004", "1005", "1006", "1007"],
            "StockCode": ["A", "B", "A", "A", "B", "A", "B", "A"],
            "Description": ["Alpha", "Beta", "Alpha", "Alpha", "Beta", "Alpha", "Beta", "Alpha"],
            "Quantity": [2, 1, 3, -1, 1, 2, 1, 1],
            "InvoiceDate": ["2024-01-01", "2024-01-01", "2024-02-15", "2024-03-01", "2024-04-01", "2024-05-01", "2024-06-01", "2024-07-01"],
            "UnitPrice": [10, 5, 10, 10, 5, 10, 5, 10],
            "CustomerID": ["1", "1", "1", "1", "2", "1", "2", "1"],
            "Country": ["UK"]*8,
        })
    def test_cancellations_removed_and_revenue_correct(self):
        df, report = clean_transactions(self.raw)
        self.assertEqual(len(df), 7)
        self.assertEqual(report["cancellation_rows"], 1)
        self.assertAlmostEqual(df["Revenue"].sum(), 95.0)
    def test_metrics_and_monthly(self):
        df, _ = clean_transactions(self.raw)
        m = summary_metrics(df)
        self.assertEqual(m["orders"], 6)
        self.assertEqual(m["customers"], 2)
        self.assertGreaterEqual(len(monthly_sales(df)), 4)
    def test_rfm_segments(self):
        df, _ = clean_transactions(self.raw)
        rfm = rfm_analysis(df)
        self.assertEqual(rfm["CustomerID"].nunique(), 2)
        self.assertIn("Segment", rfm.columns)
    def test_sql_readonly(self):
        df, _ = clean_transactions(self.raw)
        result = run_sql_query(df, "SELECT COUNT(*) AS n FROM transactions")
        self.assertEqual(result.loc[0, "n"], len(df))
        with self.assertRaises(ValueError):
            run_sql_query(df, "DELETE FROM transactions")
    def test_snapshots_have_binary_target(self):
        rows = []
        for c in range(1, 8):
            for month in range(1, 13):
                date = pd.Timestamp(2023, month, min(2+c, 25))
                rows.append({"InvoiceNo":f"{c}-{month}","StockCode":"A","Description":"A","Quantity":1,
                             "InvoiceDate":date,"UnitPrice":10.,"CustomerID":str(c),"Country":"UK","Revenue":10.,"InvoiceMonth":date.to_period("M").strftime("%Y-%m")})
        df = pd.DataFrame(rows)
        snap = build_customer_snapshots(df)
        self.assertFalse(snap.empty)
        self.assertTrue(set(snap["target_repeat_30d"].unique()).issubset({0,1}))

if __name__ == "__main__":
    unittest.main()
