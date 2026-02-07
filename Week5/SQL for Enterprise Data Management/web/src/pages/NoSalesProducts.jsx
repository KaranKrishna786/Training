import { useEffect, useState } from "react";
import { Typography, Paper } from "@mui/material";
import { api } from "../api";
import QueryGrid from "../components/QueryGrid";

export default function NoSalesProducts() {
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.get("/products/no-sales").then((res) => setRows(res.data)).finally(() => setLoading(false));
  }, []);

  const columns = [
    { field: "product_id", headerName: "Product ID", width: 120 },
    { field: "product_name", headerName: "Product", flex: 1 },
    { field: "category_name", headerName: "Category", width: 170 },
    { field: "stock_quantity", headerName: "Stock", width: 110 },
    { field: "times_ordered", headerName: "Times Ordered", width: 140 },
  ];

  return (
    <Paper sx={{ p: 2 }}>
      <Typography variant="h5" gutterBottom>Products with No Sales (Q5)</Typography>
      <QueryGrid rows={rows} columns={columns} loading={loading} />
    </Paper>
  );
}
