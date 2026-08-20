import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const inputPath = "/Users/yimingzang/Documents/Project/benchmark2/FNR-FPR/FNR-FPR tables.xlsx";
const outputDir = "/Users/yimingzang/Documents/Project/benchmark2/outputs/auc_added";
const outputPath = `${outputDir}/FNR-FPR tables_with_AUC.xlsx`;
const previewDir = "/private/tmp/fnr_fpr_auc_final";

await fs.mkdir(outputDir, { recursive: true });
await fs.mkdir(previewDir, { recursive: true });

const input = await FileBlob.load(inputPath);
const workbook = await SpreadsheetFile.importXlsx(input);

const note = "FNR=FN/(FN+TP); FPR=FP/(FP+TN); F1=2TP/(2TP+FP+FN). Posterior MMD actual positive: observed MMD exceeds the matching well-specified NPE–MCMC q95 (normalized MMD > 1). Preferred values are bold: lowest FNR/FPR and highest F1; ties included. N/A=zero denominator. AUC uses available ROC point(s) plus endpoints (0,0)/(1,1) with trapezoidal integration.";

function pairAucFormula(fnrRow, fprRow, leftCol, rightCol) {
  const x1 = `${leftCol}${fprRow}`;
  const x2 = `${rightCol}${fprRow}`;
  const e1 = `${leftCol}${fnrRow}`;
  const e2 = `${rightCol}${fnrRow}`;
  return `=IF(COUNT(${leftCol}${fnrRow}:${rightCol}${fprRow})<4,"N/A",IF(${x1}=${x2},0.5*${x1}*MAX(1-${e1},1-${e2})+0.5*(1-${x1})*(MAX(1-${e1},1-${e2})+1),IF(${x1}<${x2},0.5*${x1}*(1-${e1})+0.5*(${x2}-${x1})*(2-${e1}-${e2})+0.5*(1-${x2})*(2-${e2}),0.5*${x2}*(1-${e2})+0.5*(${x1}-${x2})*(2-${e1}-${e2})+0.5*(1-${x1})*(2-${e1}))))`;
}

function singleAucFormula(fnrRow, fprRow, col) {
  const x = `${col}${fprRow}`;
  const e = `${col}${fnrRow}`;
  return `=IF(COUNT(${col}${fnrRow}:${col}${fprRow})<2,"N/A",0.5*${x}*(1-${e})+0.5*(1-${x})*(2-${e}))`;
}

function addAucRows(sheet) {
  const blocks = [
    { aucRow: 19, fnrRow: 7, fprRow: 8, type: "paired", cols: ["C", "E", "G", "I"], copyFrom: 18 },
    { aucRow: 35, fnrRow: 23, fprRow: 24, type: "paired", cols: ["C", "E", "G", "I"], copyFrom: 34 },
    { aucRow: 51, fnrRow: 39, fprRow: 40, type: "paired", cols: ["C", "E", "G", "I"], copyFrom: 50 },
    { aucRow: 67, fnrRow: 55, fprRow: 56, type: "paired", cols: ["C", "E", "G", "I"], copyFrom: 66 },
    { aucRow: 83, fnrRow: 71, fprRow: 72, type: "single", cols: ["C", "D", "E", "F"], copyFrom: 82 },
    { aucRow: 99, fnrRow: 87, fprRow: 88, type: "single", cols: ["C", "D", "E", "F"], copyFrom: 98 },
  ];

  sheet.getRange("A2").values = [[note]];

  for (const block of blocks) {
    if (block.aucRow <= 98) {
      const sourceRange = block.type === "paired" ? `B${block.copyFrom}:J${block.copyFrom}` : `B${block.copyFrom}:F${block.copyFrom}`;
      const destRange = block.type === "paired" ? `B${block.aucRow}:J${block.aucRow}` : `B${block.aucRow}:F${block.aucRow}`;
      sheet.getRange(destRange).copyFrom(sheet.getRange(sourceRange), "all");
    } else {
      sheet.getRange("B99:F99").copyFrom(sheet.getRange("B98:F98"), "all");
    }

    sheet.getRange(`B${block.aucRow}`).values = [["AUC"]];

    if (block.type === "paired") {
      const formulas = block.cols.map((col) => [pairAucFormula(block.fnrRow, block.fprRow, col, String.fromCharCode(col.charCodeAt(0) + 1))]);
      for (let i = 0; i < block.cols.length; i++) {
        const col = block.cols[i];
        sheet.getRange(`${col}${block.aucRow}`).formulas = formulas[i];
        sheet.getRange(`${col}${block.aucRow}`).setNumberFormat("0.0%");
        sheet.getRange(`${String.fromCharCode(col.charCodeAt(0) + 1)}${block.aucRow}`).clear({ applyTo: "contents" });
      }
    } else {
      const formulas = block.cols.map((col) => [singleAucFormula(block.fnrRow, block.fprRow, col)]);
      for (let i = 0; i < block.cols.length; i++) {
        sheet.getRange(`${block.cols[i]}${block.aucRow}`).formulas = formulas[i];
        sheet.getRange(`${block.cols[i]}${block.aucRow}`).setNumberFormat("0.0%");
      }
    }
  }
}

addAucRows(workbook.worksheets.getItem("Simulated data"));
addAucRows(workbook.worksheets.getItem("Empirical data"));

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);

for (const sheetName of ["Simulated data", "Empirical data"]) {
  const preview = await workbook.render({ sheetName, autoCrop: "all", scale: 1, format: "png" });
  await fs.writeFile(`${previewDir}/${sheetName.replace(/[^A-Za-z0-9_.-]+/g, "_")}.png`, new Uint8Array(await preview.arrayBuffer()));
}

for (const sheetName of ["Simulated data", "Empirical data"]) {
  const sheet = workbook.worksheets.getItem(sheetName);
  const inspect = await workbook.inspect({
    kind: "table",
    sheetId: sheetName,
    range: "A1:J99",
    include: "values,formulas",
    tableMaxRows: 99,
    tableMaxCols: 10,
    tableMaxCellChars: 80,
    maxChars: 22000,
  });
  console.log(`CHECK ${sheetName}`);
  console.log(inspect.ndjson);
}

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});
console.log("ERROR_SCAN");
console.log(errors.ndjson);
console.log(`OUTPUT ${outputPath}`);
