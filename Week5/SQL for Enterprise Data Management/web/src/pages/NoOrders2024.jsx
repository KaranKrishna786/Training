import { useEffect, useState } from "react";
import { Typography, Paper } from "@mui/material";
import { api } from "../api";
import QueryGrid from "../components/QueryGrid";

export default function NoOrders2024() {
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.get("/customers/no-orders-2024").then((res) => setRows(res.data)).finally(() => setLoading(false));
  }, []);

  const columns = [
    { field: "customer_id", headerName: "Customer ID", width: 120 },
    { field: "first_name", headerName: "First Name", width: 130 },
    { field: "last_name", headerName: "Last Name", width: 130 },
    { field: "email", headerName: "Email", flex: 1 },
    { field: "registration_date", headerName: "Registered", width: 150 },
    { field: "loyalty_tier", headerName: "Tier", width: 120 },
    { field: "order_count_2024", headerName: "Orders 2024", width: 130 },
  ];

  return (
    <Paper sx={{ p: 2 }}>
      <Typography variant="h5" gutterBottom>Customers with No Orders in 2024 (Q6)</Typography>
      <QueryGrid rows={rows} columns={columns} loading={loading} />
    </Paper>
  );
}