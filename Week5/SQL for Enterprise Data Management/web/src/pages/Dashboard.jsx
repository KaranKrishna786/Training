import * as React from "react";
import Grid from "@mui/material/Grid"; // ✅ Grid v2
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Stack,
} from "@mui/material";
import { Link as RouterLink } from "react-router-dom";

export default function Dashboard() {
  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="h4" sx={{ mb: 2, fontWeight: 700 }}>
        Dashboard
      </Typography>

      <Typography variant="body1" sx={{ mb: 3, color: "text.secondary" }}>
        Open reports and performance tools from here.
      </Typography>

      {/* ✅ Grid v2: no `item`, no `xs/md` props. Use `size={{ xs, md }}` */}
      <Grid container spacing={2}>
        {/* Monthly Sales */}
        <Grid size={{ xs: 12, md: 4 }}>
          <Card variant="outlined" sx={{ height: "100%" }}>
            <CardContent>
              <Stack spacing={1.2}>
                <Typography variant="h6" fontWeight={700}>
                  Monthly Sales Dashboard
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Orders, customers, revenue, AOV, cancellations by month.
                </Typography>

                <Button
                  component={RouterLink}
                  to="/bi/monthly-sales"
                  variant="contained"
                >
                  Open Report
                </Button>
              </Stack>
            </CardContent>
          </Card>
        </Grid>

        {/* Indexes */}
        <Grid size={{ xs: 12, md: 4 }}>
          <Card variant="outlined" sx={{ height: "100%" }}>
            <CardContent>
              <Stack spacing={1.2}>
                <Typography variant="h6" fontWeight={700}>
                  Index Viewer
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  View indexes for orders/products (PostgreSQL).
                </Typography>

                <Button
                  component={RouterLink}
                  to="/performance/indexes"
                  variant="contained"
                >
                  Open Indexes
                </Button>
              </Stack>
            </CardContent>
          </Card>
        </Grid>

        {/* EXPLAIN */}
        <Grid size={{ xs: 12, md: 4 }}>
          <Card variant="outlined" sx={{ height: "100%" }}>
            <CardContent>
              <Stack spacing={1.2}>
                <Typography variant="h6" fontWeight={700}>
                  EXPLAIN (Analyze)
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Inspect query plan for performance troubleshooting.
                </Typography>

                <Button
                  component={RouterLink}
                  to="/performance/explain"
                  variant="contained"
                >
                  Open Explain
                </Button>
              </Stack>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}