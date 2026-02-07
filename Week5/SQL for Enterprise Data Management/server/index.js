import express from "express";
import cors from "cors";
import { pool } from "./db.js";
import dotenv from "dotenv";
dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());


const queries = {
  lowStock: `
    SELECT p.product_name, p.stock_quantity, p.reorder_level, s.supplier_name,
           (p.reorder_level - p.stock_quantity) AS units_to_order
    FROM products p
    JOIN suppliers s ON p.supplier_id = s.supplier_id
    WHERE p.stock_quantity < p.reorder_level
    ORDER BY units_to_order DESC;`,

  inactiveCustomers: `
    SELECT c.customer_id, c.first_name, c.last_name, c.email, c.loyalty_tier,
           MAX(o.order_date) AS last_order_date,
           CASE WHEN MAX(o.order_date) IS NULL THEN NULL
                ELSE (CURRENT_DATE - MAX(o.order_date)::date) END AS days_since_order
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
    GROUP BY c.customer_id, c.first_name, c.last_name, c.email, c.loyalty_tier
    HAVING MAX(o.order_date) < (CURRENT_DATE - INTERVAL '30 days')
        OR MAX(o.order_date) IS NULL
    ORDER BY days_since_order DESC NULLS LAST;`,

  orderDetails: `
    SELECT o.order_id, (c.first_name || ' ' || c.last_name) AS customer_name,
           c.email, o.order_date, p.product_name, oi.quantity, oi.unit_price,
           oi.subtotal, o.order_status
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    JOIN order_items oi ON o.order_id = oi.order_id
    JOIN products p ON oi.product_id = p.product_id
    WHERE o.order_date >= $1::timestamp
    ORDER BY o.order_date DESC, o.order_id;`,

  revenueByCategory: `
    SELECT cat.category_name,
           COUNT(DISTINCT oi.order_id) AS total_orders,
           SUM(oi.quantity) AS units_sold,
           SUM(oi.subtotal) AS total_revenue,
           AVG(oi.unit_price) AS avg_selling_price,
           SUM(oi.subtotal - (p.cost_price * oi.quantity)) AS profit
    FROM categories cat
    JOIN products p ON cat.category_id = p.category_id
    JOIN order_items oi ON p.product_id = oi.product_id
    JOIN orders o ON oi.order_id = o.order_id
    WHERE o.order_status <> 'Cancelled'
    GROUP BY cat.category_id, cat.category_name
    ORDER BY total_revenue DESC;`,

  productsNoSales: `
    SELECT p.product_id, p.product_name, p.stock_quantity, c.category_name,
           COUNT(oi.order_item_id) AS times_ordered
    FROM products p
    LEFT JOIN order_items oi ON p.product_id = oi.product_id
    LEFT JOIN categories c ON p.category_id = c.category_id
    GROUP BY p.product_id, p.product_name, p.stock_quantity, c.category_name
    HAVING COUNT(oi.order_item_id) = 0
    ORDER BY p.stock_quantity DESC;`,

  customersNoOrders2024: `
    SELECT c.customer_id, c.first_name, c.last_name, c.email,
           c.registration_date, c.loyalty_tier,
           COUNT(o.order_id) AS order_count_2024
    FROM customers c
    LEFT JOIN orders o
      ON c.customer_id = o.customer_id
     AND EXTRACT(YEAR FROM o.order_date) = 2024
    GROUP BY c.customer_id, c.first_name, c.last_name, c.email, c.registration_date, c.loyalty_tier
    HAVING COUNT(o.order_id) = 0;`,

  employeeHierarchy: `
    SELECT e1.employee_id,
           (e1.first_name || ' ' || e1.last_name) AS employee_name,
           e1.department,
           (e2.first_name || ' ' || e2.last_name) AS manager_name,
           e2.department AS manager_department
    FROM employees e1
    LEFT JOIN employees e2 ON e1.manager_id = e2.employee_id
    ORDER BY e2.employee_id NULLS FIRST, e1.employee_id;`,

  topCustomers: `
    WITH avg_orders AS (SELECT AVG(total_amount) AS avg_order_value FROM orders),
         avg_customers AS (SELECT AVG(total_spent) AS avg_customer_spent FROM customers)
    SELECT c.customer_id,
           (c.first_name || ' ' || c.last_name) AS customer_name,
           c.loyalty_tier,
           COUNT(o.order_id) AS total_orders,
           SUM(o.total_amount) AS total_spent,
           ao.avg_order_value,
           SUM(o.total_amount) - (ao.avg_order_value * COUNT(o.order_id)) AS vs_average
    FROM customers c
    JOIN orders o ON c.customer_id = o.customer_id
    CROSS JOIN avg_orders ao
    CROSS JOIN avg_customers ac
    GROUP BY c.customer_id, c.first_name, c.last_name, c.loyalty_tier, ao.avg_order_value, ac.avg_customer_spent
    HAVING SUM(o.total_amount) > ac.avg_customer_spent
    ORDER BY total_spent DESC;`,

  runningRevenue: `
    SELECT o.order_date::date AS order_day,
           COUNT(o.order_id) AS orders_count,
           SUM(o.total_amount) AS daily_revenue,
           SUM(SUM(o.total_amount)) OVER (ORDER BY o.order_date::date) AS running_total,
           AVG(SUM(o.total_amount)) OVER (
             ORDER BY o.order_date::date
             ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
           ) AS moving_avg_3day
    FROM orders o
    WHERE o.order_status <> 'Cancelled'
    GROUP BY o.order_date::date
    ORDER BY order_day;`,

     explainProductsOrderStats: `
    EXPLAIN (ANALYZE, BUFFERS)
    SELECT
      p.product_name,
      COUNT(oi.order_item_id) AS times_ordered,
      SUM(oi.quantity) AS total_quantity
    FROM products p
    LEFT JOIN order_items oi ON p.product_id = oi.product_id
    GROUP BY p.product_id, p.product_name
    ORDER BY times_ordered DESC;
  `,

  // 4.2 Index listing (Postgres)
  listIndexes: `
    SELECT schemaname, tablename, indexname, indexdef
    FROM pg_indexes
    WHERE tablename = ANY($1::text[])
    ORDER BY tablename, indexname;
  `,

  // Phase 6: Monthly sales dashboard (Postgres)
  monthlySalesDashboard: `
    SELECT
      to_char(o.order_date, 'YYYY-MM') AS month,
      COUNT(DISTINCT o.order_id) AS total_orders,
      COUNT(DISTINCT o.customer_id) AS unique_customers,
      SUM(o.total_amount) AS revenue,
      AVG(o.total_amount) AS avg_order_value,
      SUM(CASE WHEN o.order_status = 'Cancelled' THEN 1 ELSE 0 END) AS cancelled_orders,
      ROUND(
        SUM(CASE WHEN o.order_status = 'Cancelled' THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(*), 0),
        2
      ) AS cancellation_rate
    FROM orders o
    GROUP BY to_char(o.order_date, 'YYYY-MM')
    ORDER BY month DESC;`
};

// ===============================
// PHASE 7 QUERIES
const phase7 = {
  // Challenge 1.1: Products that need reordering across all warehouses
  inventoryReorderAllWarehouses: `
    WITH totals AS (
      SELECT i.product_id, COALESCE(SUM(i.quantity), 0) AS total_qty
      FROM inventory i
      GROUP BY i.product_id
    )
    SELECT
      p.product_id,
      p.product_name,
      p.reorder_level,
      t.total_qty AS total_stock_all_warehouses,
      GREATEST(p.reorder_level - t.total_qty, 0) AS units_to_order
    FROM products p
    JOIN totals t ON t.product_id = p.product_id
    WHERE t.total_qty < p.reorder_level
    ORDER BY units_to_order DESC, p.product_name;
  `,

  // Challenge 1.2: Cost of restocking all products below reorder level
  // Uses cost_price * units_to_order (restock cost)
  inventoryRestockCost: `
    WITH totals AS (
      SELECT i.product_id, COALESCE(SUM(i.quantity), 0) AS total_qty
      FROM inventory i
      GROUP BY i.product_id
    ),
    needs AS (
      SELECT
        p.product_id,
        p.product_name,
        p.cost_price,
        p.reorder_level,
        t.total_qty,
        GREATEST(p.reorder_level - t.total_qty, 0) AS units_to_order
      FROM products p
      JOIN totals t ON t.product_id = p.product_id
      WHERE t.total_qty < p.reorder_level
    )
    SELECT
      product_id,
      product_name,
      reorder_level,
      total_qty AS total_stock_all_warehouses,
      units_to_order,
      cost_price,
      ROUND(units_to_order * cost_price, 2) AS restock_cost,
      ROUND(SUM(units_to_order * cost_price) OVER (), 2) AS grand_total_restock_cost
    FROM needs
    ORDER BY restock_cost DESC, product_name;
  `,

  // Challenge 1.3: Warehouse transfers to balance inventory (heuristic)
  // Target = average stock per warehouse for each product.
  // This is a recommendation (not perfect allocation).
  inventoryTransferRecommendations: `
    WITH wh_count AS (
      SELECT COUNT(*)::int AS n FROM warehouses
    ),
    prod_totals AS (
      SELECT
        i.product_id,
        SUM(i.quantity)::int AS total_qty
      FROM inventory i
      GROUP BY i.product_id
    ),
    targets AS (
      SELECT
        pt.product_id,
        pt.total_qty,
        CEIL(pt.total_qty::numeric / NULLIF(wc.n,0))::int AS target_per_wh
      FROM prod_totals pt
      CROSS JOIN wh_count wc
    ),
    inv AS (
      SELECT
        i.product_id,
        i.warehouse_id,
        i.quantity::int AS qty,
        t.target_per_wh
      FROM inventory i
      JOIN targets t ON t.product_id = i.product_id
    ),
    surplus AS (
      SELECT product_id, warehouse_id, (qty - target_per_wh)::int AS surplus_qty
      FROM inv
      WHERE qty > target_per_wh
    ),
    deficit AS (
      SELECT product_id, warehouse_id, (target_per_wh - qty)::int AS deficit_qty
      FROM inv
      WHERE qty < target_per_wh
    )
    SELECT
      p.product_id,
      p.product_name,
      ws.warehouse_name AS from_warehouse,
      wd.warehouse_name AS to_warehouse,
      LEAST(s.surplus_qty, d.deficit_qty) AS suggested_transfer_qty
    FROM surplus s
    JOIN deficit d
      ON d.product_id = s.product_id
     AND d.warehouse_id <> s.warehouse_id
    JOIN products p ON p.product_id = s.product_id
    JOIN warehouses ws ON ws.warehouse_id = s.warehouse_id
    JOIN warehouses wd ON wd.warehouse_id = d.warehouse_id
    WHERE LEAST(s.surplus_qty, d.deficit_qty) > 0
    ORDER BY p.product_name, suggested_transfer_qty DESC;
  `,

  // Challenge 2.1: Customer cohort analysis by registration month
  // Output: cohort_month, order_month, cohort_size, active_customers, retention_rate
  customerCohorts: `
    WITH cohorts AS (
      SELECT
        c.customer_id,
        date_trunc('month', c.registration_date)::date AS cohort_month
      FROM customers c
    ),
    cohort_sizes AS (
      SELECT cohort_month, COUNT(*) AS cohort_size
      FROM cohorts
      GROUP BY cohort_month
    ),
    activity AS (
      SELECT
        co.cohort_month,
        date_trunc('month', o.order_date)::date AS order_month,
        COUNT(DISTINCT o.customer_id) AS active_customers
      FROM cohorts co
      JOIN orders o ON o.customer_id = co.customer_id
      WHERE o.order_status <> 'Cancelled'
      GROUP BY co.cohort_month, date_trunc('month', o.order_date)::date
    )
    SELECT
      a.cohort_month,
      a.order_month,
      cs.cohort_size,
      a.active_customers,
      ROUND(a.active_customers * 100.0 / NULLIF(cs.cohort_size, 0), 2) AS retention_rate_pct
    FROM activity a
    JOIN cohort_sizes cs USING (cohort_month)
    ORDER BY cohort_month, order_month;
  `,

  // Challenge 2.2: Customer churn rate (parameter: days)
  // Churn definition: no order in last X days OR never ordered.
  customerChurnRate: `
    WITH last_orders AS (
      SELECT
        c.customer_id,
        MAX(o.order_date)::date AS last_order_date
      FROM customers c
      LEFT JOIN orders o
        ON o.customer_id = c.customer_id
       AND o.order_status <> 'Cancelled'
      GROUP BY c.customer_id
    )
    SELECT
      $1::int AS churn_days,
      COUNT(*) AS total_customers,
      SUM(CASE
            WHEN last_order_date IS NULL THEN 1
            WHEN last_order_date < (CURRENT_DATE - ($1::int * INTERVAL '1 day')) THEN 1
            ELSE 0
          END) AS churned_customers,
      ROUND(
        SUM(CASE
              WHEN last_order_date IS NULL THEN 1
              WHEN last_order_date < (CURRENT_DATE - ($1::int * INTERVAL '1 day')) THEN 1
              ELSE 0
            END) * 100.0 / NULLIF(COUNT(*), 0),
        2
      ) AS churn_rate_pct
    FROM last_orders;
  `,

  // Challenge 2.3: Customers likely to upgrade loyalty tiers (heuristic)
  // Uses total_spent proximity to next tier threshold.
  // Thresholds can be adjusted easily.
  customersLikelyToUpgrade: `
    WITH thresholds AS (
      SELECT
        c.customer_id,
        c.first_name,
        c.last_name,
        c.email,
        c.loyalty_tier,
        c.total_spent,
        CASE c.loyalty_tier
          WHEN 'Bronze' THEN 1000
          WHEN 'Silver' THEN 5000
          WHEN 'Gold' THEN 10000
          ELSE NULL
        END AS next_threshold,
        CASE c.loyalty_tier
          WHEN 'Bronze' THEN 'Silver'
          WHEN 'Silver' THEN 'Gold'
          WHEN 'Gold' THEN 'Platinum'
          ELSE NULL
        END AS next_tier
      FROM customers c
    ),
    last_order AS (
      SELECT customer_id, MAX(order_date)::date AS last_order_date
      FROM orders
      WHERE order_status <> 'Cancelled'
      GROUP BY customer_id
    )
    SELECT
      t.customer_id,
      (t.first_name || ' ' || t.last_name) AS customer_name,
      t.email,
      t.loyalty_tier,
      t.next_tier,
      t.total_spent,
      t.next_threshold,
      ROUND((t.next_threshold - t.total_spent), 2) AS amount_needed,
      lo.last_order_date
    FROM thresholds t
    LEFT JOIN last_order lo ON lo.customer_id = t.customer_id
    WHERE t.next_threshold IS NOT NULL
      AND (t.next_threshold - t.total_spent) > 0
      AND (t.next_threshold - t.total_spent) <= (t.next_threshold * $1::numeric) -- buffer ratio e.g. 0.10
      AND (lo.last_order_date IS NULL OR lo.last_order_date >= CURRENT_DATE - INTERVAL '90 days')
    ORDER BY amount_needed ASC, t.total_spent DESC;
  `,

  // Challenge 3.1: Most profitable product combinations (pairs in same order)
  productCombosMostProfitable: `
    WITH lines AS (
      SELECT
        oi.order_id,
        oi.product_id,
        oi.quantity,
        oi.subtotal,
        p.cost_price,
        (oi.subtotal - (p.cost_price * oi.quantity))::numeric(12,2) AS line_profit
      FROM order_items oi
      JOIN products p ON p.product_id = oi.product_id
      JOIN orders o ON o.order_id = oi.order_id
      WHERE o.order_status <> 'Cancelled'
    ),
    pairs AS (
      SELECT
        l1.product_id AS product_a_id,
        l2.product_id AS product_b_id,
        COUNT(*) AS times_bought_together,
        SUM(l1.line_profit + l2.line_profit)::numeric(12,2) AS pair_profit
      FROM lines l1
      JOIN lines l2
        ON l2.order_id = l1.order_id
       AND l2.product_id > l1.product_id
      GROUP BY l1.product_id, l2.product_id
    )
    SELECT
      pa.product_name AS product_a,
      pb.product_name AS product_b,
      times_bought_together,
      pair_profit
    FROM pairs
    JOIN products pa ON pa.product_id = pairs.product_a_id
    JOIN products pb ON pb.product_id = pairs.product_b_id
    ORDER BY pair_profit DESC, times_bought_together DESC
    LIMIT $1::int;
  `,

  // Challenge 3.2: Discount effectiveness on revenue
  discountEffectiveness: `
    SELECT
      p.product_id,
      p.product_name,
      SUM(CASE WHEN oi.discount > 0 THEN oi.quantity ELSE 0 END) AS discounted_units,
      SUM(CASE WHEN oi.discount = 0 THEN oi.quantity ELSE 0 END) AS fullprice_units,
      ROUND(SUM(CASE WHEN oi.discount > 0 THEN oi.subtotal ELSE 0 END), 2) AS discounted_revenue,
      ROUND(SUM(CASE WHEN oi.discount = 0 THEN oi.subtotal ELSE 0 END), 2) AS fullprice_revenue,
      ROUND(AVG(CASE WHEN oi.discount > 0 THEN (oi.discount / NULLIF(oi.unit_price * oi.quantity,0)) * 100 END), 2) AS avg_discount_pct
    FROM order_items oi
    JOIN products p ON p.product_id = oi.product_id
    JOIN orders o ON o.order_id = oi.order_id
    WHERE o.order_status <> 'Cancelled'
    GROUP BY p.product_id, p.product_name
    ORDER BY discounted_revenue DESC NULLS LAST, fullprice_revenue DESC;
  `,

  // Challenge 3.3: Revenue per warehouse (via shipments)
  // If multiple shipments per order exist, this attributes revenue by the latest shipment record.
  revenuePerWarehouse: `
    WITH latest_shipment AS (
      SELECT DISTINCT ON (s.order_id)
        s.order_id,
        s.warehouse_id,
        s.shipment_date
      FROM shipments s
      ORDER BY s.order_id, s.shipment_date DESC
    )
    SELECT
      w.warehouse_id,
      w.warehouse_name,
      COUNT(DISTINCT o.order_id) AS orders_count,
      ROUND(SUM(o.total_amount), 2) AS revenue
    FROM latest_shipment ls
    JOIN orders o ON o.order_id = ls.order_id
    JOIN warehouses w ON w.warehouse_id = ls.warehouse_id
    WHERE o.order_status <> 'Cancelled'
    GROUP BY w.warehouse_id, w.warehouse_name
    ORDER BY revenue DESC;
  `,

  // Challenge 4: index recommendations (returns SQL text as rows)
  indexRecommendations: `
    SELECT * FROM (VALUES
      ('CREATE INDEX IF NOT EXISTS idx_inventory_product ON inventory(product_id);'),
      ('CREATE INDEX IF NOT EXISTS idx_inventory_warehouse ON inventory(warehouse_id);'),
      ('CREATE INDEX IF NOT EXISTS idx_inventory_product_warehouse ON inventory(product_id, warehouse_id);'),
      ('CREATE INDEX IF NOT EXISTS idx_customers_registration_date ON customers(registration_date);'),
      ('CREATE INDEX IF NOT EXISTS idx_orders_order_date ON orders(order_date);'),
      ('CREATE INDEX IF NOT EXISTS idx_shipments_order_warehouse ON shipments(order_id, warehouse_id);'),
      ('CREATE INDEX IF NOT EXISTS idx_order_items_order_product ON order_items(order_id, product_id);')
    ) AS t(sql);
  `
};


// ===============================
// PHASE 7 ROUTES
// ===============================

// Challenge 1
app.get("/api/phase7/inventory/reorder", async (_, res) => {
  const { rows } = await pool.query(phase7.inventoryReorderAllWarehouses);
  res.json(rows);
});

app.get("/api/phase7/inventory/restock-cost", async (_, res) => {
  const { rows } = await pool.query(phase7.inventoryRestockCost);
  res.json(rows);
});

app.get("/api/phase7/inventory/transfer-recommendations", async (_, res) => {
  const { rows } = await pool.query(phase7.inventoryTransferRecommendations);
  res.json(rows);
});

// Challenge 2
app.get("/api/phase7/customers/cohorts", async (_, res) => {
  const { rows } = await pool.query(phase7.customerCohorts);
  res.json(rows);
});

// churn?days=60
app.get("/api/phase7/customers/churn", async (req, res) => {
  const days = Number(req.query.days ?? 60);
  const { rows } = await pool.query(phase7.customerChurnRate, [days]);
  res.json(rows[0]); // single row summary
});

// upgrade-candidates?buffer=0.10
app.get("/api/phase7/customers/upgrade-candidates", async (req, res) => {
  const buffer = Number(req.query.buffer ?? 0.10);
  const { rows } = await pool.query(phase7.customersLikelyToUpgrade, [buffer]);
  res.json(rows);
});

// Challenge 3
// product-combos?limit=20
app.get("/api/phase7/revenue/product-combos", async (req, res) => {
  const limit = Number(req.query.limit ?? 20);
  const { rows } = await pool.query(phase7.productCombosMostProfitable, [limit]);
  res.json(rows);
});

app.get("/api/phase7/revenue/discount-effectiveness", async (_, res) => {
  const { rows } = await pool.query(phase7.discountEffectiveness);
  res.json(rows);
});

app.get("/api/phase7/revenue/by-warehouse", async (_, res) => {
  const { rows } = await pool.query(phase7.revenuePerWarehouse);
  res.json(rows);
});

// Challenge 4
app.get("/api/phase7/performance/index-recommendations", async (_, res) => {
  const { rows } = await pool.query(phase7.indexRecommendations);
  res.json(rows);
});


app.get("/api/inventory/low-stock", async (_, res) => {
  const { rows } = await pool.query(queries.lowStock);
  res.json(rows);
});

app.get("/api/customers/inactive", async (_, res) => {
  const { rows } = await pool.query(queries.inactiveCustomers);
  res.json(rows);
});

app.get("/api/orders/details", async (req, res) => {
  const from = req.query.from ?? "2024-01-01";
  const { rows } = await pool.query(queries.orderDetails, [from]);
  res.json(rows);
});

app.get("/api/analytics/revenue-by-category", async (_, res) => {
  const { rows } = await pool.query(queries.revenueByCategory);
  res.json(rows);
});

app.get("/api/products/no-sales", async (_, res) => {
  const { rows } = await pool.query(queries.productsNoSales);
  res.json(rows);
});

app.get("/api/customers/no-orders-2024", async (_, res) => {
  const { rows } = await pool.query(queries.customersNoOrders2024);
  res.json(rows);
});

app.get("/api/employees/hierarchy", async (_, res) => {
  const { rows } = await pool.query(queries.employeeHierarchy);
  res.json(rows);
});

app.get("/api/customers/top", async (_, res) => {
  const { rows } = await pool.query(queries.topCustomers);
  res.json(rows);
});

app.get("/api/analytics/running-revenue", async (_, res) => {
  const { rows } = await pool.query(queries.runningRevenue);
  res.json(rows);
});

app.listen(process.env.PORT || 4000, () => {
  console.log(`API running on http://localhost:${process.env.PORT || 4000}`);
});

// 4.1 EXPLAIN endpoint
app.get("/api/performance/explain/products-order-stats", async (_, res) => {
  try {
    const { rows } = await pool.query(explainSql);
    const planText = rows.map(r => r["QUERY PLAN"]).join("\n");
    res.json({ plan: planText });
  } catch (err) {
    console.error("EXPLAIN error:", err.message);
    res.status(500).json({ error: err.message });
  }
});

const explainSql = `
  EXPLAIN (ANALYZE, BUFFERS)
  SELECT
    p.product_name,
    COUNT(oi.order_item_id) AS times_ordered,
    SUM(oi.quantity) AS total_quantity
  FROM products p
  LEFT JOIN order_items oi ON p.product_id = oi.product_id
  GROUP BY p.product_id, p.product_name
  ORDER BY times_ordered DESC;
`;
``

app.get("/api/views/customer-order-summary", async (req, res) => {
  try {
    const min = Number(req.query.min ?? 1000);
    const { rows } = await pool.query(
      `SELECT * FROM vw_customer_order_summary
       WHERE lifetime_value > $1
       ORDER BY lifetime_value DESC;`,
      [min]
    );
    res.json(rows);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// 4.2 List indexes for tables
app.get("/api/performance/indexes", async (req, res) => {
  try {
    const tables = (req.query.tables ? req.query.tables.split(",") : ["orders", "products"]);
    const { rows } = await pool.query(queries.listIndexes, [tables]);
    res.json(rows);
  } catch (err) {
    console.error("Indexes error:", err.message);
    res.status(500).json({ error: err.message });
  }
});

// Phase 6: Monthly sales dashboard
app.get("/api/bi/monthly-sales", async (_, res) => {
  try {
    const { rows } = await pool.query(queries.monthlySalesDashboard);
    res.json(rows);
  } catch (err) {
    console.error("Monthly sales error:", err.message);
    res.status(500).json({ error: err.message });
  }
});