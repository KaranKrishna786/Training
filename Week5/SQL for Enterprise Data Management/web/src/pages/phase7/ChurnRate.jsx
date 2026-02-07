import { useEffect, useState } from "react";
import { Paper, Typography, Stack, TextField, Button, Alert } from "@mui/material";
import { api } from "../../api";

export default function ChurnRate() {
  const [days, setDays] = useState(60);
  const [data, setData] = useState(null);
  const [error, setError] = useState("");

  const load = async () => {
    try {
      setError("");
      const res = await api.get("/phase7/customers/churn", { params: { days } });
      setData(res.data);
    } catch (e) {
      setError(e?.response?.data?.error || e.message);
    }
  };

  useEffect(() => { load(); }, []);

  return (
    <Paper sx={{ p: 2 }}>
      <Typography variant="h5" gutterBottom>
        Phase 7 — Customer Churn Rate
      </Typography>

      <Stack direction="row" spacing={2} sx={{ mb: 2 }}>
        <TextField
          label="Churn Days"
          type="number"
          value={days}
          onChange={(e) => setDays(Number(e.target.value))}
          sx={{ width: 180 }}
        />
        <Button variant="contained" onClick={load}>Calculate</Button>
      </Stack>

      {error && <Alert severity="error">{error}</Alert>}

      {data && (
        <pre style={{ textAlign: "left" }}>
{JSON.stringify(data, null, 2)}
        </pre>
      )}
    </Paper>
  );
}
