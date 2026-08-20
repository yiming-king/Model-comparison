import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const outputPath = "/Users/yimingzang/Documents/Project/benchmark2/outputs/auc_added/FNR-FPR tables_with_AUC.xlsx";
const input = await FileBlob.load(outputPath);
const workbook = await SpreadsheetFile.importXlsx(input);

for (const sheetName of ["Simulated data", "Empirical data"]) {
  const sheet = workbook.worksheets.getItem(sheetName);
  const check = await workbook.inspect({
    kind: "table",
    sheetId: sheetName,
    range: "B19:J19",
    include: "values,formulas",
    tableMaxRows: 1,
    tableMaxCols: 9,
    maxChars: 3000,
  });
  console.log(`AUC_ROW ${sheetName}`);
  console.log(check.ndjson);
  console.log(`AUC_FORMULAS ${sheetName}`);
  console.log(JSON.stringify(sheet.getRange("B19:J19").formulas));
  const checkPosterior = await workbook.inspect({
    kind: "table",
    sheetId: sheetName,
    range: "B83:F83",
    include: "values,formulas",
    tableMaxRows: 1,
    tableMaxCols: 5,
    maxChars: 3000,
  });
  console.log(`AUC_POSTERIOR ${sheetName}`);
  console.log(checkPosterior.ndjson);
  console.log(`AUC_POSTERIOR_FORMULAS ${sheetName}`);
  console.log(JSON.stringify(sheet.getRange("B83:F83").formulas));
}

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 300 },
  summary: "post-export formula error scan",
});
console.log("POST_EXPORT_ERROR_SCAN");
console.log(errors.ndjson);
