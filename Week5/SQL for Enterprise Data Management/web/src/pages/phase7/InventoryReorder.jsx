import ReportGrid from "../ReportGrid";

export default function InventoryReorder() {
  return (
    <ReportGrid
      title="Phase 7 — Inventory: Reorder Across All Warehouses"
      endpoint="/phase7/inventory/reorder"
      getRowId={(r) => r.product_id}
    />
  );
}
