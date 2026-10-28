-- Top products by revenue
SELECT StockCode, Description, ROUND(SUM(Revenue), 2) AS Revenue,
       SUM(Quantity) AS Units, COUNT(DISTINCT InvoiceNo) AS Orders
FROM transactions
GROUP BY StockCode, Description
ORDER BY Revenue DESC
LIMIT 10;
