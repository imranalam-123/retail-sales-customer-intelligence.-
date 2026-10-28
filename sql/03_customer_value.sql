-- Historical spend and ordering frequency by known customer
SELECT CustomerID, ROUND(SUM(Revenue), 2) AS HistoricalSpend,
       COUNT(DISTINCT InvoiceNo) AS Orders, MAX(InvoiceDate) AS LastPurchase
FROM transactions
WHERE CustomerID IS NOT NULL
GROUP BY CustomerID
ORDER BY HistoricalSpend DESC
LIMIT 100;
