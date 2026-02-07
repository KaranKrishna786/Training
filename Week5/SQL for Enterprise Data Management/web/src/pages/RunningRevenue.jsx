import { useEffect, useState } from "react";
import { Typography, Paper, Grid } from "@mui/material";
import { api } from "../api";
import QueryGrid from "../components/QueryGrid";
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip } from "recharts";

export default function RunningRevenue() {
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.get("/analytics/running-revenue").then((res) => setRows(res.data)).finally(() => setLoading(false));
  }, []);

  const columns = [
    { field: "order_day", headerName: "Day", width: 140 },
    { field: "orders_count", headerName: "Orders", width: 100 },
    { field: "daily_revenue", headerName: "Daily Revenue", width: 140 },
    { field: "running_total", headerName: "Running Total", width: 140 },
    { field: "moving_avg_3day", headerName: "3-day Avg", width: 120 },
  ];

  return (
    <Paper sx={{ p: 2 }}>
      <Typography variant="h5" gutterBottom>Running Daily Revenue (Q14)</Typography>

      <Grid container spacing={2}>
        <Grid item xs={12} md={6} sx={{ height: 320 }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={rows}>
              <XAxis dataKey="order_day" />
              <YAxis />
              <Tooltip />
              <Line type="monotone" dataKey="daily_revenue" stroke="#2e7d32" dot={false} />
              <Line type="monotone" dataKey="running_total" stroke="#1976d2" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </Grid>

        <Grid item xs={12}>
          <QueryGrid rows={rows} columns={columns} loading={loading} />
        </Grid>
      </Grid>
    </Paper>
  );
}