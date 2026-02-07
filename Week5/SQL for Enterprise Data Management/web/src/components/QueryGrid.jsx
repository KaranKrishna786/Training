import { DataGrid } from "@mui/x-data-grid";
import { Box, CircularProgress } from "@mui/material";

export default function QueryGrid({ rows, columns, loading }) {
  return (
    <Box sx={{ height: 560, width: "100%" }}>
      {loading ? (
        <Box sx={{ display: "flex", justifyContent: "center", mt: 8 }}>
          <CircularProgress />
        </Box>
      ) : (
        <DataGrid
          rows={rows}
          columns={columns}
          getRowId={(r) => r.id ?? r.order_id ?? r.customer_id ?? r.product_id ?? r.employee_id ?? r.review_id ?? JSON.stringify(r)}
          pageSizeOptions={[10, 25, 50]}
          initialState={{ pagination: { paginationModel: { pageSize: 10, page: 0 } } }}
        />
      )}
    </Box>
  );
}