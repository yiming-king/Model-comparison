import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";
const path = "/Users/yimingzang/Documents/Project/benchmark2/outputs/true_roc_auc/FNR-FPR tables_true_ROC_AUC.xlsx";
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(path));
for (const sheetName of ["Simulated data", "Empirical data"]) {
  const sheet = workbook.worksheets.getItem(sheetName);
  console.log(sheetName);
  for (const start of [4, 28, 52, 76, 100, 124]) {
    console.log(start, JSON.stringify(sheet.getRange(`A${start}:J${start + 3}`).values));
  }
}
