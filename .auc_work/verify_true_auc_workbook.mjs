import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const path = "/Users/yimingzang/Documents/Project/benchmark2/outputs/true_roc_auc/FNR-FPR tables_true_ROC_AUC.xlsx";
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(path));
const errors = [];
const counts = {};

for (const sheetName of ["Simulated data", "Empirical data"]) {
  const sheet = workbook.worksheets.getItem(sheetName);
  let rocRows = 0;
  let rocNumeric = 0;
  let rocNA = 0;
  for (let block = 0; block < 6; block++) {
    const start = 4 + block * 20;
    const lastCol = block < 4 ? "J" : "F";
    for (let group = 0; group < 4; group++) {
      const aucRow = start + 6 + group * 4;
      if (sheet.getRange(`B${aucRow}`).values[0][0] !== "ROC-AUC") {
        errors.push(`${sheetName}!B${aucRow} label`);
      }
      for (let code = "C".charCodeAt(0); code <= lastCol.charCodeAt(0); code++) {
        const value = sheet.getRange(`${String.fromCharCode(code)}${aucRow}`).values[0][0];
        rocRows++;
        if (typeof value === "number") rocNumeric++;
        else if (value === "N/A") rocNA++;
        else errors.push(`${sheetName}!${String.fromCharCode(code)}${aucRow} unexpected ${value}`);
      }
    }
    if (block < 5) {
      const blank = sheet.getRange(`A${start + 19}:J${start + 19}`).values[0];
      if (blank.some((value) => value !== "" && value !== null)) {
        errors.push(`${sheetName}!${start + 19} separator not blank`);
      }
    }
  }
  counts[sheetName] = { rocCells: rocRows, numeric: rocNumeric, na: rocNA };
}

const source = workbook.worksheets.getItem("Source rates");
if (source.getRange("J1").values[0][0] !== "ROC-AUC") {
  errors.push("Source rates metric headers");
}
const sourceAuc = source.getRange("J2:J321").values.flat();
counts.sourceRates = {
  rows: sourceAuc.length,
  numeric: sourceAuc.filter((value) => typeof value === "number").length,
  na: sourceAuc.filter((value) => value === "N/A").length,
};

const formulaErrors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 500 },
  summary: "post-export formula scan",
});
if (!formulaErrors.ndjson.includes("matched 0 entries")) errors.push(formulaErrors.ndjson);

const removedMetric = await workbook.inspect({
  kind: "match",
  searchTerm: "Balanced Accuracy|balanced_accuracy|Balanced AUC",
  options: { useRegex: true, maxResults: 50 },
  summary: "removed metric scan",
});
if (!removedMetric.ndjson.includes("matched 0 entries")) errors.push(removedMetric.ndjson);

console.log(JSON.stringify({ counts, errors }, null, 2));
if (errors.length) process.exitCode = 1;
