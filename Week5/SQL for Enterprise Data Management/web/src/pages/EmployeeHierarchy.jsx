import { useEffect, useState } from "react";
import { Typography, Paper } from "@mui/material";
import { api } from "../api";
import QueryGrid from "../components/QueryGrid";

export default function EmployeeHierarchy() {
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.get("/employees/hierarchy").then((res) => setRows(res.data)).finally(() => setLoading(false));
  }, []);

  const columns = [
    { field: "employee_id", headerName: "Employee ID", width: 120 },
    { field: "employee_name", headerName: "Employee", width: 200 },
    { field: "department", headerName: "Department", width: 140 },
    { field: "manager_name", headerName: "Manager", width: 200 },
    { field: "manager_department", headerName: "Mgr Dept", width: 140 },
  ];

  return (
    <Paper sx={{ p: 2 }}>
      <Typography variant="h5" gutterBottom>Employee Hierarchy (Q7)</Typography>
      <QueryGrid rows={rows} columns={columns} loading={loading} />
    </Paper>
  );
}