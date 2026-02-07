import { useState } from "react";
import { Paper, Typography, Stack, TextField } from "@mui/material";
import ReportGrid from "../ReportGrid";

export default function ProductCombos() {
  const [limit, setLimit] = useState(20);
  const endpoint = `/phase7/revenue/product-combos?limit=${limit}`;

  return (
    <Paper sx={{ p: 2 }}>
      <Typography variant="h5" gutterBottom>
        Phase 7 — Most Profitable Product Combinations
      </Typography>

      <Stack direction="row" spacing={2} sx={{ mb: 2 }}>
        <TextField
          label="Top N"
          type="number"
          value={limit}
          onChange={(e) => setLimit(Number(e.target.value))}
          sx={{ width: 140 }}
        />
      </Stack>

      <ReportGrid
        title="Profitable Product Pairs"
        endpoint={endpoint}
        getRowId={(r, idx) => `${r.product_a}-${r.product_b}-${idx}`}
      />
    </Paper>
  );
}