import ReportGrid from "../ReportGrid";

export default function RestockCost() {
  return (
    <ReportGrid
      title="Phase 7 — Inventory: Restock Cost (Below Reorder Level)"
      endpoint="/phase7/inventory/restock-cost"
      getRowId={(r) => r.product_id}
    />
  );
}
