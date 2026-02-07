import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import Layout from "./components/Layout";

import Dashboard from "./pages/Dashboard";
import LowStock from "./pages/LowStock";
import InactiveCustomers from "./pages/InactiveCustomers";
import OrderDetails from "./pages/OrderDetails";
import RevenueByCategory from "./pages/RevenueByCategory";
import NoSalesProducts from "./pages/NoSalesProducts";
import NoOrders2024 from "./pages/NoOrders2024";
import EmployeeHierarchy from "./pages/EmployeeHierarchy";
import TopCustomers from "./pages/TopCustomers";
import RunningRevenue from "./pages/RunningRevenue";
import MonthlySales from "./pages/MonthlySales";
import Indexes from "./pages/Indexes";
import ExplainPlan from "./pages/ExplainPlan";
import InventoryReorder from "./pages/phase7/InventoryReorder";
import RestockCost from "./pages/phase7/RestockCost";
import TransferRecommendations from "./pages/phase7/TransferRecommendations";
import CohortAnalysis from "./pages/phase7/CohortAnalysis";
import ChurnRate from "./pages/phase7/ChurnRate";
import UpgradeCandidates from "./pages/phase7/UpgradeCandidates";
import ProductCombos from "./pages/phase7/ProductCombos";
import DiscountEffectiveness from "./pages/phase7/DiscountEffectiveness";
import RevenueByWarehouse from "./pages/phase7/RevenueByWarehouse";
import IndexRecommendations from "./pages/phase7/IndexRecommendations";


export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<Navigate to="/dashboard" replace />} />
          <Route path="/dashboard" element={<Dashboard />} />

          <Route path="/inventory/low-stock" element={<LowStock />} />
          <Route path="/inventory/no-sales" element={<NoSalesProducts />} />

          <Route path="/customers/inactive" element={<InactiveCustomers />} />
          <Route path="/customers/no-orders-2024" element={<NoOrders2024 />} />
          <Route path="/customers/top" element={<TopCustomers />} />

          <Route path="/orders/details" element={<OrderDetails />} />

          <Route path="/analytics/revenue-by-category" element={<RevenueByCategory />} />
          <Route path="/analytics/running-revenue" element={<RunningRevenue />} />

          <Route path="/employees/hierarchy" element={<EmployeeHierarchy />} />
<Route path="/phase7/inventory/reorder" element={<InventoryReorder />} />
<Route path="/phase7/inventory/restock-cost" element={<RestockCost />} />
<Route path="/phase7/inventory/transfers" element={<TransferRecommendations />} />

<Route path="/phase7/customers/cohorts" element={<CohortAnalysis />} />
<Route path="/phase7/customers/churn" element={<ChurnRate />} />
<Route path="/phase7/customers/upgrade" element={<UpgradeCandidates />} />

<Route path="/phase7/revenue/product-combos" element={<ProductCombos />} />
<Route path="/phase7/revenue/discount-effectiveness" element={<DiscountEffectiveness />} />
<Route path="/phase7/revenue/by-warehouse" element={<RevenueByWarehouse />} />

<Route path="/phase7/performance/index-recommendations" element={<IndexRecommendations />} />

          
        <Route path="/bi/monthly-sales" element={<MonthlySales />} />
        <Route path="/performance/indexes" element={<Indexes />} />
        <Route path="/performance/explain" element={<ExplainPlan />} />

        </Route>
      </Routes>
    </BrowserRouter>
  );
}
