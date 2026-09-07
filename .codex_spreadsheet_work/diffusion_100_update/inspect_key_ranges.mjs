import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const filePath = "/Users/yimingzang/Documents/Project/benchmark2/FNR-FPR/FNR-FPR tables without MMD.xlsx";
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(filePath));

for (const [sheetName, ranges] of Object.entries({
  Method: ["A1:F20"],
  "Simulated data": ["A1:F243"],
  "Empirical data": ["A1:F243"],
  "Chart data": ["A1:O30", "A70:O110", "A140:O220"],
})) {
  for (const range of ranges) {
    console.log((await workbook.inspect({
      kind: "table",
      sheetId: sheetName,
      range,
      include: "values,formulas",
      tableMaxRows: sheetName.includes("data") && range.endsWith("243") ? 243 : 90,
      tableMaxCols: 15,
      maxChars: 50000,
    })).ndjson);
  }
}
