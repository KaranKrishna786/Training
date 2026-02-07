import { useEffect, useState } from "react";
import { Typography, Paper } from "@mui/material";
import { api } from "../api";
import QueryGrid from "../components/QueryGrid";

export default function InactiveCustomers() {
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.get("/customers/inactive").then((res) => setRows(res.data)).finally(() => setLoading(false));
  }, []);

  const columns = [
    { field: "customer_id", headerName: "Customer ID", width: 120 },
    { field: "first_name", headerName: "First Name", width: 130 },
    { field: "last_name", headerName: "Last Name", width: 130 },
    { field: "email", headerName: "Email", flex: 1 },
    { field: "loyalty_tier", headerName: "Tier", width: 120 },
    { field: "last_order_date", headerName: "Last Order", width: 160 },
    { field: "days_since_order", headerName: "Days Since", width: 130 },
  ];

  return (
    <Paper sx={{ p: 2 }}>
      <Typography variant="h5" gutterBottom>Customers inactive 30+ days (Q2)</Typography>
      <QueryGrid rows={rows} columns={columns} loading={loading} />
    </Paper>
  );
}
