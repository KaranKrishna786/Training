import { useEffect, useState } from "react";
import { Typography, Paper } from "@mui/material";
import { api } from "../api";
import QueryGrid from "../components/QueryGrid";

export default function TopCustomers() {
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.get("/customers/top").then((res) => setRows(res.data)).finally(() => setLoading(false));
  }, []);

  const columns = [
    { field: "customer_id", headerName: "Customer ID", width: 120 },
    { field: "customer_name", headerName: "Customer", width: 220 },
    { field: "loyalty_tier", headerName: "Tier", width: 120 },
    { field: "total_orders", headerName: "Orders", width: 100 },
    { field: "total_spent", headerName: "Total Spent", width: 130 },
    { field: "avg_order_value", headerName: "Avg Order", width: 120 },
    { field: "vs_average", headerName: "Vs Avg", width: 120 },
  ];

  return (
    <Paper sx={{ p: 2 }}>
      <Typography variant="h5" gutterBottom>Top Customers (Q11)</Typography>
      <QueryGrid rows={rows} columns={columns} loading={loading} />
    </Paper>
  );
}
``