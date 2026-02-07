import ReportGrid from "../ReportGrid";

export default function RevenueByWarehouse() {
  return (
    <ReportGrid
      title="Phase 7 — Revenue Per Warehouse"
      endpoint="/phase7/revenue/by-warehouse"
      getRowId={(r) => r.warehouse_id}
    />
  );
}