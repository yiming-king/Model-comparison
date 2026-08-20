import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const input = await FileBlob.load("/Users/yimingzang/Documents/Project/benchmark2/FNR-FPR/FNR-FPR tables.xlsx");
const workbook = await SpreadsheetFile.importXlsx(input);
for (const sheetName of ["Simulated data", "Empirical data"]) {
  const sheet = workbook.worksheets.getItem(sheetName);
  console.log(sheetName, {
    row4: sheet.getRange("A4:J4").format.rowHeightPx,
    row5: sheet.getRange("A5:J5").format.rowHeightPx,
    row6: sheet.getRange("A6:J6").format.rowHeightPx,
    row7: sheet.getRange("A7:J7").format.rowHeightPx,
    row9: sheet.getRange("A9:J9").format.rowHeightPx,
    row19: sheet.getRange("A19:J19").format.rowHeightPx,
    colA: sheet.getRange("A:A").format.columnWidthPx,
    colB: sheet.getRange("B:B").format.columnWidthPx,
    colC: sheet.getRange("C:C").format.columnWidthPx,
  });
  const styles = await workbook.inspect({
    kind: "computedStyle",
    sheetId: sheetName,
    range: "A7:J9",
    maxChars: 12000,
  });
  console.log(styles.ndjson);
}
