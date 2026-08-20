import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const input = await FileBlob.load("/Users/yimingzang/Documents/Project/benchmark2/outputs/auc_aligned/FNR-FPR tables_with_AUC_aligned.xlsx");
const workbook = await SpreadsheetFile.importXlsx(input);
for (const sheetName of ["Simulated data", "Empirical data"]) {
  for (const cell of ["C10", "D10", "E10", "F10", "C14", "D14", "C30", "D30", "C50", "D50"]) {
    const style = await workbook.inspect({ kind: "computedStyle", sheetId: sheetName, range: cell, maxChars: 1500 });
    console.log(sheetName, cell, "directBold", workbook.worksheets.getItem(sheetName).getRange(cell).format.font.bold, style.ndjson);
  }
}
