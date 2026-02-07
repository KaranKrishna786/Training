import { useEffect, useState } from "react";
import { Typography, Paper, Grid } from "@mui/material";
import { api } from "../api";
import QueryGrid from "../components/QueryGrid";
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip } from "recharts";

export default function RevenueByCategory() {
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.get("/analytics/revenue-by-category").then((res) => setRows(res.data)).finally(() => setLoading(false));
  }, []);

  const columns = [
    { field: "category_name", headerName: "Category", flex: 1 },
    { field: "total_orders", headerName: "Orders", width: 110 },
    { field: "units_sold", headerName: "Units Sold", width: 120 },
    { field: "total_revenue", headerName: "Revenue", width: 140 },
    { field: "avg_selling_price", headerName: "Avg Price", width: 130 },
    { field: "profit", headerName: "Profit", width: 140 },
  ];

  return (
    <Paper sx={{ p: 2 }}>
      <Typography variant="h5" gutterBottom>Revenue by Category (Q4)</Typography>

      <Grid container spacing={2}>
        <Grid item xs={12} md={6} sx={{ height: 320 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={rows}>
              <XAxis dataKey="category_name" hide />
              <YAxis />
              <Tooltip />
              <Bar dataKey="total_revenue" fill="#1976d2" />
            </BarChart>
          </ResponsiveContainer>
          <Typography sx={{ mt: 1, opacity: 0.7 }}>Revenue chart (top categories)</Typography>
        </Grid>

        <Grid item xs={12} md={12}>
          <QueryGrid rows={rows} columns={columns} loading={loading} />
        </Grid>
      </Grid>
    </Paper>
  );
}