import ReportGrid from "../ReportGrid";

export default function DiscountEffectiveness() {
  return (
    <ReportGrid
      title="Phase 7 — Discount Effectiveness"
      endpoint="/phase7/revenue/discount-effectiveness"
      getRowId={(r) => r.product_id}
    />
  );
}