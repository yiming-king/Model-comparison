import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const path = "/Users/yimingzang/Documents/Project/benchmark2/outputs/auc_aligned/FNR-FPR tables_with_AUC_aligned.xlsx";
const input = await FileBlob.load(path);
const workbook = await SpreadsheetFile.importXlsx(input);
const pairs = [["C", "D"], ["E", "F"], ["G", "H"], ["I", "J"]];
const errors = [];
let checkedPairs = 0;
let trueTies = 0;

for (const sheetName of ["Simulated data", "Empirical data"]) {
  const sheet = workbook.worksheets.getItem(sheetName);
  for (let block = 0; block < 4; block++) {
    const start = 4 + block * 20;
    for (let group = 0; group < 4; group++) {
      const row = start + 6 + group * 4;
      for (const [left, right] of pairs) {
        checkedPairs++;
        const [lv, rv] = sheet.getRange(`${left}${row}:${right}${row}`).values[0];
        const lb = sheet.getRange(`${left}${row}`).format.font.bold === true;
        const rb = sheet.getRange(`${right}${row}`).format.font.bold === true;
        const ln = typeof lv === "number";
        const rn = typeof rv === "number";
        let expectedLeft = false;
        let expectedRight = false;
        if (ln && rn && Math.abs(lv - rv) <= 1e-12) {
          expectedLeft = true;
          expectedRight = true;
          trueTies++;
        } else if (ln && (!rn || lv > rv)) {
          expectedLeft = true;
        } else if (rn && (!ln || rv > lv)) {
          expectedRight = true;
        }
        if (lb !== expectedLeft || rb !== expectedRight) {
          errors.push({ sheetName, row, pair: `${left}:${right}`, lv, rv, lb, rb, expectedLeft, expectedRight });
        }
      }
    }
  }
  for (let block = 4; block < 6; block++) {
    const start = 4 + block * 20;
    for (let group = 0; group < 4; group++) {
      const row = start + 6 + group * 4;
      for (const col of ["C", "D", "E", "F"]) {
        if (sheet.getRange(`${col}${row}`).format.font.bold === true) {
          errors.push({ sheetName, row, cell: `${col}${row}`, issue: "single-threshold AUC should not be bold" });
        }
      }
    }
  }
}

console.log(JSON.stringify({ checkedPairs, trueTies, errors }, null, 2));
if (errors.length) process.exitCode = 1;
