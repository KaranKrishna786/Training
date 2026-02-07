import * as React from "react";
import { Box, Typography, Alert, Button } from "@mui/material";
import { api } from "../api";

export default function ExplainPlan() {
  const [plan, setPlan] = React.useState(null);
  const [error, setError] = React.useState("");

  const load = async () => {
    try {
      setError("");
      const { data } = await api.get("/performance/explain/products-order-stats");
      setPlan(data);
    } catch (e) {
      setError(e?.response?.data?.error || e.message);
    }
  };

  React.useEffect(() => { load(); }, []);

  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="h5" sx={{ mb: 2 }}>
        EXPLAIN Plan: Products Order Stats
      </Typography>

      <Button variant="contained" onClick={load} sx={{ mb: 2 }}>
        Re-run EXPLAIN
      </Button>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Box
        component="pre"
        sx={{
          p: 2,
          borderRadius: 2,
          bgcolor: "#0b1020",
          color: "#d7e1ff",
          overflow: "auto",
          maxHeight: 650,
          whiteSpace: "pre-wrap",   // ✅ wraps long lines
          wordBreak: "break-word",
        }}
      >
        {plan?.plan ? plan.plan : "Loading..."}
      </Box>
    </Box>
  );
}
