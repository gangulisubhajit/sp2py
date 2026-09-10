CREATE PROCEDURE dbo.CancelStaleOrders
    @stale_days INT = 7
AS
BEGIN
    SET NOCOUNT ON;
    BEGIN TRANSACTION;

    BEGIN TRY
        DECLARE @cancelled_count INT = 0;

        CREATE TABLE #stale_orders (order_id INT);

        INSERT INTO #stale_orders (order_id)
        SELECT order_id
        FROM orders
        WHERE order_status = 'PENDING'
          AND order_date < DATEADD(DAY, -@stale_days, GETDATE());

        UPDATE orders
        SET order_status = 'CANCELLED'
        WHERE order_id IN (SELECT order_id FROM #stale_orders);

        SET @cancelled_count = @@ROWCOUNT;

        INSERT INTO order_audit_log (order_id, event, event_time)
        SELECT order_id, 'AUTO_CANCELLED', GETDATE()
        FROM #stale_orders;

        DROP TABLE #stale_orders;

        COMMIT TRANSACTION;

        SELECT @cancelled_count AS cancelled_count;
    END TRY
    BEGIN CATCH
        ROLLBACK TRANSACTION;
        THROW;
    END CATCH
END;
