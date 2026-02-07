import ReportGrid from "./ReportGrid";

export default function Indexes() {
  return (
    <ReportGrid
      title="Index List (orders, products)"
      endpoint="/performance/indexes?tables=orders,products"
      getRowId={(row) => `${row.tablename}-${row.indexname}`}
    />
  );
}