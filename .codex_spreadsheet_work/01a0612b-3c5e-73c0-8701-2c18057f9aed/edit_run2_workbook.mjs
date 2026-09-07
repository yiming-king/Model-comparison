import fs from "node:fs/promises";
import path from "node:path";

import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";


const inputPath = "/Users/yimingzang/Documents/Project/benchmark2/outputs/01a038b9-1f9b-7d01-9240-218e90236b72/FNR-FPR tables without MMD rerun1.xlsx";
const outputPath = "/Users/yimingzang/Documents/Project/benchmark2/outputs/01a0612b-3c5e-73c0-8701-2c18057f9aed/FNR-FPR tables without MMD rerun1.xlsx";
const previewDir = "/private/tmp/run2_all_simulated_after";


function assertCondition(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}


function validateAllSimulatedScope(rows) {
  const headers = rows[0];
  const expectedHeaders = [
    "Case study", "Source group", "Diagnostic", "Assumed model",
    "Summary dimension", "Error metric", "Threshold definition",
    "Lower threshold", "Upper threshold", "TP", "FP", "FN", "TN",
    "N", "FNR", "FPR", "F1", "AUC",
  ];
  assertCondition(
    JSON.stringify(headers) === JSON.stringify(expectedHeaders),
    "Unexpected Source rates header layout",
  );

  const diffusionSimulated = rows.slice(1).filter(
    (row) => row[0] === "Diffusion" && row[1] === "simulated",
  );
  assertCondition(
    diffusionSimulated.length === 192,
    `Expected 192 Diffusion simulated metric rows; found ${diffusionSimulated.length}`,
  );

  const wrongDenominators = diffusionSimulated.filter((row) => Number(row[13]) !== 119);
  assertCondition(
    wrongDenominators.length === 0,
    `Diffusion simulated rows must use all seven simulated sources (N=119); found ${wrongDenominators.length} exceptions`,
  );

  const inconsistentConfusion = diffusionSimulated.filter(
    (row) => [9, 10, 11, 12].reduce((total, index) => total + Number(row[index]), 0) !== Number(row[13]),
  );
  assertCondition(
    inconsistentConfusion.length === 0,
    `TP+FP+FN+TN must equal N; found ${inconsistentConfusion.length} exceptions`,
  );

  const signedLog10Rows = diffusionSimulated.filter(
    (row) => row[5] === "log10_ml_error",
  );
  assertCondition(
    signedLog10Rows.length === 64,
    `Expected 64 Diffusion simulated signed-log10 rows; found ${signedLog10Rows.length}`,
  );

  return {
    diffusionSimulatedRows: diffusionSimulated.length,
    signedLog10Rows: signedLog10Rows.length,
    denominator: 119,
  };
}


await fs.mkdir(path.dirname(outputPath), { recursive: true });
await fs.mkdir(previewDir, { recursive: true });

const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(inputPath));
const sourceRates = workbook.worksheets.getItem("Source rates");
const validation = validateAllSimulatedScope(sourceRates.getRange("A1:R529").values);

const method = workbook.worksheets.getItem("Method");
method.getRange("B4").values = [[
  "Diffusion (without MMD loss, rerun1) and Gaussian case studies. Diffusion simulated results include every non-empirical source: seven source groups x 17 datasets = N=119 per diagnostic x assumed model x summary dimension x error metric. They are not restricted to model-matched/well-specified datasets. Export QC requires N=119 and TP+FP+FN+TN=N for every Diffusion simulated row.",
]];
method.getRange("B20").values = [["2026-09-02"]];
method.getRange("B4").format.wrapText = true;
method.getRange("A4:B4").format.rowHeightPx = 72;

console.log((await workbook.inspect({
  kind: "region",
  sheetId: "Source rates",
  range: "A98:R101",
  maxChars: 12000,
  tableMaxRows: 6,
  tableMaxCols: 18,
})).ndjson);
console.log((await workbook.inspect({
  kind: "region",
  sheetId: "Method",
  range: "A3:B20",
  maxChars: 12000,
  tableMaxRows: 22,
  tableMaxCols: 2,
})).ndjson);
console.log((await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
  maxChars: 12000,
})).ndjson);

for (const sheet of workbook.worksheets.items) {
  const safeName = sheet.name.replace(/[^A-Za-z0-9_-]+/g, "_");
  const preview = await workbook.render({
    sheetName: sheet.name,
    autoCrop: "all",
    scale: 0.65,
    format: "png",
  });
  await fs.writeFile(
    path.join(previewDir, `${safeName}.png`),
    new Uint8Array(await preview.arrayBuffer()),
  );
}

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);

const saved = await SpreadsheetFile.importXlsx(await FileBlob.load(outputPath));
const savedRows = saved.worksheets.getItem("Source rates").getRange("A1:R529").values;
const savedValidation = validateAllSimulatedScope(savedRows);
console.log(JSON.stringify({
  outputPath,
  ...validation,
  savedValidation,
  sheetCount: saved.worksheets.items.length,
}, null, 2));
