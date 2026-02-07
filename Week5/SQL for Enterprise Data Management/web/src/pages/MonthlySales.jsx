import ReportGrid from "./ReportGrid";

export default function MonthlySales() {
  return (
    <ReportGrid
      title="Monthly Sales Dashboard"
      endpoint="/bi/monthly-sales"
      getRowId={(row) => row.month}   // ✅ unique
    />
  );
}