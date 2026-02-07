import ReportGrid from "../ReportGrid";

export default function CohortAnalysis() {
  return (
    <ReportGrid
      title="Phase 7 — Customer Cohort Analysis (Registration Month)"
      endpoint="/phase7/customers/cohorts"
      getRowId={(r, idx) => `${r.cohort_month}-${r.order_month}-${idx}`}
    />
  );
}