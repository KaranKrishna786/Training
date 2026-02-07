import * as React from "react";
import { DataGrid } from "@mui/x-data-grid";
import { Box, Typography, Alert, CircularProgress } from "@mui/material";
import { api } from "../api"; // your axios instance

export default function ReportGrid({ title, endpoint, getRowId }) {
  const [rows, setRows] = React.useState([]);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState("");

  React.useEffect(() => {
    let alive = true;

    (async () => {
      try {
        setLoading(true);
        const { data } = await api.get(endpoint);
        if (!alive) return;
        setRows(Array.isArray(data) ? data : []);
        setError("");
      } catch (e) {
        if (!alive) return;
        setError(e?.response?.data?.error || e.message);
      } finally {
        if (alive) setLoading(false);
      }
    })();

    return () => { alive = false; };
  }, [endpoint]);

  const columns = React.useMemo(() => {
  if (!rows || rows.length === 0) return [];
  return Object.keys(rows[0]).map((key) => ({
    field: key,
    headerName: key.replaceAll("_", " ").toUpperCase(),
    flex: 1,
    minWidth: 160,
  }));
}, [rows]);

  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="h5" sx={{ mb: 2 }}>{title}</Typography>

      {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

      {loading ? (
        <Box sx={{ display: "flex", gap: 2, alignItems: "center" }}>
          <CircularProgress size={22} />
          <Typography>Loading...</Typography>
        </Box>
      ) : (
        <Box sx={{ height: 620, width: "100%" }}>
          <DataGrid
            rows={rows}
            columns={columns}
            getRowId={getRowId || ((row, idx) => row.id ?? row.order_id ?? row.product_id ?? row.customer_id ?? idx)}
            pageSizeOptions={[10, 25, 50, 100]}
            initialState={{ pagination: { paginationModel: { pageSize: 25, page: 0 } } }}
          />
        </Box>
      )}
    </Box>
  );
}