import { useState } from "react";
import { Paper, Typography, Stack, TextField, Button } from "@mui/material";
import ReportGrid from "../ReportGrid";

export default function UpgradeCandidates() {
  const [buffer, setBuffer] = useState(0.10);
  const endpoint = `/phase7/customers/upgrade-candidates?buffer=${buffer}`;

  return (
    <Paper sx={{ p: 2 }}>
      <Typography variant="h5" gutterBottom>
        Phase 7 — Customers Likely to Upgrade Tier
      </Typography>

      <Stack direction="row" spacing={2} sx={{ mb: 2 }}>
        <TextField
          label="Buffer (e.g. 0.10 = within 10%)"
          type="number"
          value={buffer}
          onChange={(e) => setBuffer(Number(e.target.value))}
          sx={{ width: 280 }}
          inputProps={{ step: "0.01" }}
        />
        <Button variant="contained" onClick={() => { /* rerender triggers new endpoint */ }}>
          Refresh
        </Button>
      </Stack>

      <ReportGrid
        title="Upgrade Candidates"
        endpoint={endpoint}
        getRowId={(r) => r.customer_id}
      />
    </Paper>
  );
}