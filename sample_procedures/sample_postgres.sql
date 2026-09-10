CREATE OR REPLACE FUNCTION recalculate_order_total(p_order_id INTEGER)
RETURNS TABLE(order_id INTEGER, new_total NUMERIC) AS $$
DECLARE
    v_total NUMERIC(12,2) := 0;
    item RECORD;
BEGIN
    FOR item IN
        SELECT quantity, unit_price
        FROM order_items
        WHERE order_items.order_id = p_order_id
    LOOP
        v_total := v_total + (item.quantity * item.unit_price);
    END LOOP;

    IF v_total < 0 THEN
        RAISE EXCEPTION 'Computed negative total for order %', p_order_id;
    END IF;

    UPDATE orders
    SET total_amount = v_total,
        order_status = CASE WHEN v_total = 0 THEN 'EMPTY' ELSE order_status END
    WHERE orders.order_id = p_order_id;

    RETURN QUERY SELECT p_order_id, v_total;
END;
$$ LANGUAGE plpgsql;
