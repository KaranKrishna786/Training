import ReportGrid from "../ReportGrid";

export default function TransferRecommendations() {
  return (
    <ReportGrid
      title="Phase 7 — Inventory: Transfer Recommendations (Balance Stock)"
      endpoint="/phase7/inventory/transfer-recommendations"
      getRowId={(r, idx) => `${r.product_id}-${r.from_warehouse}-${r.to_warehouse}-${idx}`}
    />
  );
}