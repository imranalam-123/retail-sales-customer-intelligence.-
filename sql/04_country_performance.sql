-- Country-level performance
SELECT Country, ROUND(SUM(Revenue), 2) AS Revenue,
       COUNT(DISTINCT InvoiceNo) AS Orders, COUNT(DISTINCT CustomerID) AS Customers
FROM transactions
GROUP BY Country
ORDER BY Revenue DESC;
