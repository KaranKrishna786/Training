import { useEffect, useState } from "react";
import { Typography, Paper, TextField, Button, Stack } from "@mui/material";
import { api } from "../api";
import QueryGrid from "../components/QueryGrid";

export default function OrderDetails() {
  const [from, setFrom] = useState("2024-01-01");
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(false);

  const load = () => {
    setLoading(true);
    api.get("/orders/details", { params: { from } })
      .then((res) => setRows(res.data))
      .finally(() => setLoading(false));
  };

  useEffect(() => { load(); }, []);

  const columns = [
    { field: "order_id", headerName: "Order ID", width: 110 },
    { field: "customer_name", headerName: "Customer", width: 180 },
    { field: "email", headerName: "Email", flex: 1 },
    { field: "order_date", headerName: "Order Date", width: 180 },
    { field: "product_name", headerName: "Product", width: 220 },
    { field: "quantity", headerName: "Qty", width: 80 },
    { field: "unit_price", headerName: "Unit Price", width: 110 },
    { field: "subtotal", headerName: "Subtotal", width: 110 },
    { field: "order_status", headerName: "Status", width: 120 },
  ];

  return (
    <Paper sx={{ p: 2 }}>
      <Typography variant="h5" gutterBottom>Order Details (Q3)</Typography>

      <Stack direction="row" spacing={2} sx={{ mb: 2 }}>
        <TextField
          label="From date"
          type="date"
          value={from}
          onChange={(e) => setFrom(e.target.value)}
          InputLabelProps={{ shrink: true }}
        />
        <Button variant="contained" onClick={load}>Load</Button>
      </Stack>

      <QueryGrid rows={rows} columns={columns} loading={loading} />
    </Paper>
  );
}