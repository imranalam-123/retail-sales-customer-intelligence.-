-- Average order value by month (aggregate invoices first)
WITH invoice_totals AS (
  SELECT InvoiceMonth, InvoiceNo, SUM(Revenue) AS OrderRevenue
  FROM transactions
  GROUP BY InvoiceMonth, InvoiceNo
)
SELECT InvoiceMonth, ROUND(AVG(OrderRevenue), 2) AS AverageOrderValue
FROM invoice_totals
GROUP BY InvoiceMonth
ORDER BY InvoiceMonth;
