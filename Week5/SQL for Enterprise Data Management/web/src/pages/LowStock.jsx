import { useEffect, useState } from "react";
import { Typography, Paper } from "@mui/material";
import { api } from "../api";
import QueryGrid from "../components/QueryGrid";

export default function LowStock() {
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.get("/inventory/low-stock").then((res) => setRows(res.data)).finally(() => setLoading(false));
  }, []);

  const columns = [
    { field: "product_name", headerName: "Product", flex: 1 },
    { field: "supplier_name", headerName: "Supplier", flex: 1 },
    { field: "stock_quantity", headerName: "Stock", width: 120 },
    { field: "reorder_level", headerName: "Reorder Level", width: 140 },
    { field: "units_to_order", headerName: "Units to Order", width: 140 },
  ];

  return (
    <Paper sx={{ p: 2 }}>
      <Typography variant="h5" gutterBottom>Low Stock Alerts (Q1)</Typography>
      <QueryGrid rows={rows} columns={columns} loading={loading} />
    </Paper>
  );
}