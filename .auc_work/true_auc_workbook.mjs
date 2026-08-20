import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const root = "/Users/yimingzang/Documents/Project/benchmark2";
const inputPath = `${root}/FNR-FPR/FNR-FPR tables.xlsx`;
const aucPath = `${root}/outputs/true_roc_auc/data/roc_auc_summary.json`;
const outputPath = `${root}/outputs/true_roc_auc/FNR-FPR tables_true_ROC_AUC.xlsx`;
const previewDir = `${root}/outputs/true_roc_auc/previews`;

await fs.mkdir(previewDir, { recursive: true });
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(inputPath));
const aucRows = JSON.parse(await fs.readFile(aucPath, "utf8"));

const key = (row) => [
  row.source_group,
  row.diagnostic,
  row.assumed_model,
  row.summary_dimension,
  row.error_metric,
  row.threshold_definition,
].join("|");
const aucLookup = new Map(aucRows.map((row) => [key(row), row.roc_auc ?? null]));

const diagnostics = ["l2", "density", "linf", "mmd"];
const summaries = ["S=D", "S=2D", "S=4D", "S=6D"];
const diagnosticColumnPairs = {
  l2: ["C", "D"],
  density: ["E", "F"],
  linf: ["G", "H"],
  mmd: ["I", "J"],
};
const posteriorDiagnosticColumns = { l2: "C", density: "D", linf: "E", mmd: "F" };
const blocks = [
  { errorMetric: "log10_ml_error", model: "m1", lastCol: "J", thresholds: ["q95_normalized", "fixed"] },
  { errorMetric: "log10_ml_error", model: "m3", lastCol: "J", thresholds: ["q95_normalized", "fixed"] },
  { errorMetric: "pmp_error", model: "m1", lastCol: "J", thresholds: ["q95_normalized", "fixed"] },
  { errorMetric: "pmp_error", model: "m3", lastCol: "J", thresholds: ["q95_normalized", "fixed"] },
  { errorMetric: "posterior_mmd", model: "m1", lastCol: "F", thresholds: ["q95_npe_mcmc"] },
  { errorMetric: "posterior_mmd", model: "m3", lastCol: "F", thresholds: ["q95_npe_mcmc"] },
];

const thin = { style: "thin", color: "#D9DEE5" };
const medium = { style: "medium", color: "#A6AFBB" };
const bodyFont = { bold: false, fontSize: 11, typeface: "Carlito", color: "#1F2937" };

function clearConditionalFormatsCellwise(sheet, startRow, endRow, startCol = "C", endCol = "J") {
  for (let row = startRow; row <= endRow; row++) {
    for (let code = startCol.charCodeAt(0); code <= endCol.charCodeAt(0); code++) {
      sheet.getRange(`${String.fromCharCode(code)}${row}`).conditionalFormats.clear();
    }
  }
}

function styleMetricRow(sheet, row, lastCol, isLastGroup) {
  const full = sheet.getRange(`A${row}:${lastCol}${row}`);
  full.format.font = bodyFont;
  full.format.rowHeightPx = 26.6666666667;
  full.format.borders = {
    top: thin,
    bottom: isLastGroup ? medium : thin,
    left: medium,
    right: medium,
    insideVertical: thin,
  };
  sheet.getRange(`A${row}:B${row}`).format.horizontalAlignment = "center";
  sheet.getRange(`C${row}:${lastCol}${row}`).format.horizontalAlignment = "right";
  sheet.getRange(`C${row}:${lastCol}${row}`).format.numberFormat = "0.0%";
}

function setWinnerBold(sheet, left, right, row) {
  const values = sheet.getRange(`${left}${row}:${right}${row}`).values[0];
  const numeric = values.map((value) => typeof value === "number" && Number.isFinite(value));
  sheet.getRange(`${left}${row}:${right}${row}`).format.font = bodyFont;
  if (numeric[0] && numeric[1]) {
    if (Math.abs(values[0] - values[1]) <= 1e-12) {
      sheet.getRange(`${left}${row}:${right}${row}`).format.font = { ...bodyFont, bold: true };
    } else if (values[0] > values[1]) {
      sheet.getRange(`${left}${row}`).format.font = { ...bodyFont, bold: true };
    } else {
      sheet.getRange(`${right}${row}`).format.font = { ...bodyFont, bold: true };
    }
  } else if (numeric[0]) {
    sheet.getRange(`${left}${row}`).format.font = { ...bodyFont, bold: true };
  } else if (numeric[1]) {
    sheet.getRange(`${right}${row}`).format.font = { ...bodyFont, bold: true };
  }
}

function rebuildResultSheet(sheet, sourceGroup) {
  const heights = new Map();
  for (let row = 1; row <= 98; row++) {
    heights.set(row, sheet.getRange(`A${row}:J${row}`).format.rowHeightPx);
  }
  sheet.getRange("A1:J130").unmerge();

  for (let blockIndex = blocks.length - 1; blockIndex >= 0; blockIndex--) {
    const block = blocks[blockIndex];
    const originalStart = 4 + blockIndex * 16;
    const newStart = 4 + blockIndex * 20;
    const rowMap = [
      [originalStart, newStart],
      [originalStart + 1, newStart + 1],
      [originalStart + 2, newStart + 2],
    ];
    for (let group = 0; group < 4; group++) {
      const originalGroup = originalStart + 3 + group * 3;
      const newGroup = newStart + 3 + group * 4;
      rowMap.push([originalGroup, newGroup]);
      rowMap.push([originalGroup + 1, newGroup + 1]);
      rowMap.push([originalGroup + 2, newGroup + 2]);
    }
    if (blockIndex < blocks.length - 1) {
      const blankTarget = newStart + 19;
      sheet.getRange(`A${blankTarget}:J${blankTarget}`).copyFrom(sheet.getRange("A19:J19"), "all");
      sheet.getRange(`A${blankTarget}:J${blankTarget}`).clear({ applyTo: "contents" });
    }
    rowMap.sort((a, b) => b[0] - a[0]);
    for (const [sourceRow, targetRow] of rowMap) {
      if (sourceRow !== targetRow) {
        sheet.getRange(`A${targetRow}:J${targetRow}`).clear({ applyTo: "contents" });
      }
      sheet.getRange(`A${targetRow}:J${targetRow}`).copyFrom(sheet.getRange(`A${sourceRow}:J${sourceRow}`), "all");
      const height = heights.get(sourceRow);
      if (height !== undefined) sheet.getRange(`A${targetRow}:J${targetRow}`).format.rowHeightPx = height;
    }

    for (let group = 0; group < 4; group++) {
      const groupStart = newStart + 3 + group * 4;
      const fnrRow = groupStart;
      const fprRow = groupStart + 1;
      const f1Row = groupStart + 2;
      const aucRow = groupStart + 3;
      sheet.getRange(`A${aucRow}:J${aucRow}`).copyFrom(sheet.getRange(`A${f1Row}:J${f1Row}`), "all");
      sheet.getRange(`A${aucRow}:J${aucRow}`).clear({ applyTo: "contents" });
      clearConditionalFormatsCellwise(sheet, aucRow, aucRow);
      sheet.getRange(`B${aucRow}`).values = [["ROC-AUC"]];

      for (const diagnostic of diagnostics) {
        const pair = block.lastCol === "J"
          ? diagnosticColumnPairs[diagnostic]
          : [posteriorDiagnosticColumns[diagnostic]];
        for (let thresholdIndex = 0; thresholdIndex < block.thresholds.length; thresholdIndex++) {
          const col = pair[thresholdIndex];
          const lookupKey = [
            sourceGroup,
            diagnostic,
            block.model,
            summaries[group],
            block.errorMetric,
            block.thresholds[thresholdIndex],
          ].join("|");
          const auc = aucLookup.get(lookupKey);
          sheet.getRange(`${col}${aucRow}`).values = [[typeof auc === "number" ? auc : "N/A"]];
        }
      }
      sheet.getRange(`A${f1Row}:${block.lastCol}${f1Row}`).format.borders = { bottom: thin };
      styleMetricRow(sheet, aucRow, block.lastCol, group === 3);
    }
  }

  sheet.mergeCells("A1:J1");
  sheet.mergeCells("A2:J2");
  const originalNote = String(sheet.getRange("A2").values?.[0]?.[0] ?? "");
  const cleanNote = originalNote.replace(/\s*AUC=.*$/u, "").trim();
  sheet.getRange("A2").values = [[
    `${cleanNote} ROC-AUC uses continuous rho; N/A means y_true has one class. Higher is better; for paired q95/fixed columns only the larger value is bold (ties both).`,
  ]];

  for (let blockIndex = 0; blockIndex < blocks.length; blockIndex++) {
    const block = blocks[blockIndex];
    const start = 4 + blockIndex * 20;
    sheet.mergeCells(`A${start}:${block.lastCol}${start}`);
    sheet.mergeCells(`A${start + 1}:A${start + 2}`);
    sheet.mergeCells(`B${start + 1}:B${start + 2}`);
    if (block.lastCol === "J") {
      for (const [left, right] of Object.values(diagnosticColumnPairs)) {
        sheet.mergeCells(`${left}${start + 1}:${right}${start + 1}`);
      }
    }
    for (let group = 0; group < 4; group++) {
      const groupStart = start + 3 + group * 4;
      sheet.mergeCells(`A${groupStart}:A${groupStart + 3}`);
    }
  }
}

function styleResultSheet(sheet) {
  const navy = "#203864";
  const sectionFill = "#D9E7F5";
  const headerFill = "#E7E9ED";
  const labelFill = "#FAFBFC";
  const q95Fill = "#F3F7FC";
  const textColor = "#1F2937";
  const blueBorder = { style: "medium", color: "#4472C4" };
  clearConditionalFormatsCellwise(sheet, 1, 130);

  sheet.getRange("A1:J1").format = {
    fill: navy,
    font: { bold: true, fontSize: 16, typeface: "Carlito", color: "#FFFFFF" },
    horizontalAlignment: "center", verticalAlignment: "center",
  };
  sheet.getRange("A1:J1").format.rowHeightPx = 34;
  sheet.getRange("A2:J2").format = {
    fill: "#EAF2F8",
    font: { italic: true, fontSize: 8.5, typeface: "Carlito", color: "#4B5563" },
    horizontalAlignment: "left", verticalAlignment: "center", wrapText: true,
  };
  sheet.getRange("A2:J2").format.rowHeightPx = 54;
  sheet.getRange("A3:J3").format.rowHeightPx = 14;

  for (let blockIndex = 0; blockIndex < blocks.length; blockIndex++) {
    const block = blocks[blockIndex];
    const start = 4 + blockIndex * 20;
    const title = sheet.getRange(`A${start}:${block.lastCol}${start}`);
    title.format = {
      fill: sectionFill,
      font: { bold: true, fontSize: 13, typeface: "Carlito", color: navy },
      horizontalAlignment: "center", verticalAlignment: "center",
      borders: { top: blueBorder, bottom: medium, left: medium, right: medium },
    };
    title.format.rowHeightPx = 34.6666666667;
    const header = sheet.getRange(`A${start + 1}:${block.lastCol}${start + 2}`);
    header.format = {
      fill: headerFill,
      font: { bold: true, fontSize: 10, typeface: "Carlito", color: textColor },
      horizontalAlignment: "center", verticalAlignment: "center", wrapText: true,
      borders: { preset: "all", style: "thin", color: "#A6AFBB" },
    };
    sheet.getRange(`A${start + 1}:${block.lastCol}${start + 2}`).format.rowHeightPx = 29.3333333333;

    for (let group = 0; group < 4; group++) {
      const groupStart = start + 3 + group * 4;
      const groupEnd = groupStart + 3;
      sheet.getRange(`A${groupStart}:B${groupEnd}`).format = {
        fill: labelFill,
        font: bodyFont,
        horizontalAlignment: "center", verticalAlignment: "center",
        borders: { preset: "all", style: "thin", color: "#D9DEE5" },
      };
      if (block.lastCol === "J") {
        for (const col of ["C", "E", "G", "I"]) sheet.getRange(`${col}${groupStart}:${col}${groupEnd}`).format.fill = q95Fill;
        for (const col of ["D", "F", "H", "J"]) sheet.getRange(`${col}${groupStart}:${col}${groupEnd}`).format.fill = "#FFFFFF";
      } else {
        sheet.getRange(`C${groupStart}:F${groupEnd}`).format.fill = q95Fill;
      }
      const numbers = sheet.getRange(`C${groupStart}:${block.lastCol}${groupEnd}`);
      numbers.format.font = bodyFont;
      numbers.format.horizontalAlignment = "right";
      numbers.format.verticalAlignment = "center";
      numbers.format.numberFormat = "0.0%";
      numbers.format.borders = { preset: "all", style: "thin", color: "#D9DEE5" };
      sheet.getRange(`A${groupStart}:A${groupEnd}`).format.borders = { left: medium, right: thin, top: thin, bottom: group === 3 ? medium : thin };
      sheet.getRange(`${block.lastCol}${groupStart}:${block.lastCol}${groupEnd}`).format.borders = { right: medium, left: thin, top: thin, bottom: group === 3 ? medium : thin };
      sheet.getRange(`A${groupEnd}:${block.lastCol}${groupEnd}`).format.borders = { bottom: group === 3 ? medium : thin, left: medium, right: medium, insideVertical: thin, top: thin };
      for (let row = groupStart; row <= groupEnd; row++) sheet.getRange(`A${row}:${block.lastCol}${row}`).format.rowHeightPx = 26.6666666667;

      const winnerRules = [[groupStart, "MIN"], [groupStart + 1, "MIN"], [groupStart + 2, "MAX"]];
      for (const [row, fn] of winnerRules) {
        sheet.getRange(`C${row}:${block.lastCol}${row}`).conditionalFormats.add("expression", {
          formula: `=AND(ISNUMBER(C${row}),C${row}=${fn}($C${row}:$${block.lastCol}${row}))`,
          format: { font: { bold: true } },
        });
      }
      if (block.lastCol === "J") {
        for (const [left, right] of Object.values(diagnosticColumnPairs)) {
          setWinnerBold(sheet, left, right, groupStart + 3);
        }
      }
    }
    if (blockIndex < blocks.length - 1) {
      const blank = start + 19;
      sheet.getRange(`A${blank}:J${blank}`).clear({ applyTo: "all" });
      sheet.getRange(`A${blank}:J${blank}`).clear({ applyTo: "contents" });
      sheet.getRange(`A${blank}:J${blank}`).formulas = [["", "", "", "", "", "", "", "", "", ""]];
      sheet.getRange(`A${blank}:J${blank}`).values = [["", "", "", "", "", "", "", "", "", ""]];
      sheet.getRange(`A${blank}:J${blank}`).format.rowHeightPx = 14;
    }
  }
  sheet.getRange("G84:J123").clear({ applyTo: "all" });
  sheet.showGridLines = false;
}

rebuildResultSheet(workbook.worksheets.getItem("Simulated data"), "simulated");
rebuildResultSheet(workbook.worksheets.getItem("Empirical data"), "empirical");
styleResultSheet(workbook.worksheets.getItem("Simulated data"));
styleResultSheet(workbook.worksheets.getItem("Empirical data"));

const sourceSheet = workbook.worksheets.getItem("Source rates");
const sourceValues = sourceSheet.getRange("A1:I321").values;
sourceSheet.getRange("A1:K321").clear({ applyTo: "contents" });
sourceSheet.getRange("A1:J321").values = sourceValues.map((row, index) => {
  if (index === 0) return [...row, "ROC-AUC"];
  const lookupKey = [row[0], row[1], row[2], row[3], row[4], row[5]].join("|");
  const auc = aucLookup.get(lookupKey);
  return [...row, typeof auc === "number" ? auc : "N/A"];
});
sourceSheet.getRange("J2:J321").format.numberFormat = "0.0%";
sourceSheet.getRange("J2:J321").format.horizontalAlignment = "right";
sourceSheet.getRange("J1").format.font = { bold: true, fontSize: 11, typeface: "Carlito" };
sourceSheet.getRange("J:J").format.columnWidthPx = 130;

for (const sheetName of ["Simulated data", "Empirical data", "Source rates"]) {
  const preview = await workbook.render({ sheetName, autoCrop: "all", scale: sheetName === "Source rates" ? 0.45 : 0.75, format: "png" });
  await fs.writeFile(`${previewDir}/${sheetName.replaceAll(" ", "_")}.png`, new Uint8Array(await preview.arrayBuffer()));
}

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 500 },
  summary: "formula error scan",
});
console.log("ERROR_SCAN");
console.log(errors.ndjson);

for (const sheetName of ["Simulated data", "Empirical data"]) {
  const sample = await workbook.inspect({
    kind: "table", sheetId: sheetName, range: "A4:J27", include: "values,formulas",
    tableMaxRows: 20, tableMaxCols: 10, maxChars: 12000,
  });
  console.log(`SAMPLE ${sheetName}`);
  console.log(sample.ndjson);
}

await (await SpreadsheetFile.exportXlsx(workbook)).save(outputPath);
console.log(outputPath);
