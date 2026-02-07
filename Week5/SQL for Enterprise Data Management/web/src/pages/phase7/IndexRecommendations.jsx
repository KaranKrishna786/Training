import ReportGrid from "../ReportGrid";

export default function IndexRecommendations() {
  return (
    <ReportGrid
      title="Phase 7 — Performance: Index Recommendations"
      endpoint="/phase7/performance/index-recommendations"
      getRowId={(r, idx) => `${idx}-${r.sql}`}
    />
  );
}