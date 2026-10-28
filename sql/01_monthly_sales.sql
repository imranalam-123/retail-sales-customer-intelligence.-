-- Monthly revenue and order counts
SELECT InvoiceMonth, ROUND(SUM(Revenue), 2) AS Revenue,
       COUNT(DISTINCT InvoiceNo) AS Orders, SUM(Quantity) AS Units
FROM transactions
GROUP BY InvoiceMonth
ORDER BY InvoiceMonth;
