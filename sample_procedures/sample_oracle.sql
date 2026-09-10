CREATE OR REPLACE PROCEDURE process_customer_order (
    p_customer_id   IN  NUMBER,
    p_order_id      OUT NUMBER
) IS
    v_total_amount  NUMBER(12,2) := 0;
    v_stock         NUMBER;
    CURSOR c_items IS
        SELECT product_id, quantity, unit_price
        FROM   order_items_staging
        WHERE  customer_id = p_customer_id;
BEGIN
    INSERT INTO orders (order_id, customer_id, order_status, order_date, total_amount)
    VALUES (orders_seq.NEXTVAL, p_customer_id, 'PENDING', SYSDATE, 0)
    RETURNING order_id INTO p_order_id;

    FOR item_rec IN c_items LOOP
        SELECT stock_quantity INTO v_stock
        FROM   products
        WHERE  product_id = item_rec.product_id
        FOR UPDATE;

        IF v_stock < item_rec.quantity THEN
            RAISE_APPLICATION_ERROR(-20001, 'Insufficient stock for product ' || item_rec.product_id);
        END IF;

        UPDATE products
        SET    stock_quantity = stock_quantity - item_rec.quantity
        WHERE  product_id = item_rec.product_id;

        INSERT INTO order_items (order_id, product_id, quantity, unit_price)
        VALUES (p_order_id, item_rec.product_id, item_rec.quantity, item_rec.unit_price);

        v_total_amount := v_total_amount + (item_rec.quantity * item_rec.unit_price);
    END LOOP;

    UPDATE orders
    SET    total_amount = v_total_amount,
           order_status = 'CONFIRMED'
    WHERE  order_id = p_order_id;

    COMMIT;
EXCEPTION
    WHEN OTHERS THEN
        ROLLBACK;
        RAISE;
END process_customer_order;
/
