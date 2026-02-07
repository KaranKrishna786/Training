import { Outlet, Link, useLocation } from "react-router-dom";
import { Box, Drawer, List, ListItemButton, ListItemText, AppBar, Toolbar, Typography } from "@mui/material";

const menu = [
  { label: "Dashboard", to: "/dashboard" },

  { label: "Inventory: Low Stock (Q1)", to: "/inventory/low-stock" },
  { label: "Inventory: No Sales (Q5)", to: "/inventory/no-sales" },

  { label: "Customers: Inactive 30+ days (Q2)", to: "/customers/inactive" },
  { label: "Customers: No Orders 2024 (Q6)", to: "/customers/no-orders-2024" },
  { label: "Customers: Top Customers (Q11)", to: "/customers/top" },

  { label: "Orders: Details (Q3)", to: "/orders/details" },

  { label: "Analytics: Revenue by Category (Q4)", to: "/analytics/revenue-by-category" },
  { label: "Analytics: Running Revenue (Q14)", to: "/analytics/running-revenue" },

  { label: "Employees: Hierarchy (Q7)", to: "/employees/hierarchy" },
];

export default function Layout() {
  const { pathname } = useLocation();

  return (
    <Box sx={{ display: "flex" }}>
      <AppBar position="fixed" sx={{ zIndex: 1201 }}>
        <Toolbar>
          <Typography variant="h6">E-Commerce Admin</Typography>
          <Typography sx={{ ml: 2, opacity: 0.7 }}>{pathname}</Typography>
        </Toolbar>
      </AppBar>

      <Drawer variant="permanent" sx={{ width: 290, [`& .MuiDrawer-paper`]: { width: 290, mt: 8 } }}>
        <List>
          {menu.map((item) => (
            <ListItemButton key={item.to} component={Link} to={item.to} selected={pathname === item.to}>
              <ListItemText primary={item.label} />
            </ListItemButton>
          ))}
        </List>
      </Drawer>

      <Box component="main" sx={{ flexGrow: 1, p: 3, mt: 8 }}>
        <Outlet />
      </Box>
    </Box>
  );
}