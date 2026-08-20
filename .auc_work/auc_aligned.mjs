import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const inputPath = "/Users/yimingzang/Documents/Project/benchmark2/FNR-FPR/FNR-FPR tables.xlsx";
const outputDir = "/Users/yimingzang/Documents/Project/benchmark2/outputs/auc_aligned";
const outputPath = `${outputDir}/FNR-FPR tables_with_AUC_aligned.xlsx`;
const previewDir = "/private/tmp/fnr_fpr_auc_aligned";

await fs.mkdir(outputDir, { recursive: true });
await fs.mkdir(previewDir, { recursive: true });

const input = await FileBlob.load(inputPath);
const workbook = await SpreadsheetFile.importXlsx(input);

const thin = { style: "thin", color: "#D9DEE5" };
const medium = { style: "medium", color: "#A6AFBB" };
const bodyFont = { bold: false, fontSize: 11, typeface: "Carlito", color: "#1F2937" };

function clearConditionalFormatsCellwise(sheet, startRow, endRow, startCol = "C", endCol = "J") {
  for (let row = startRow; row <= endRow; row++) {
    for (let code = startCol.charCodeAt(0); code <= endCol.charCodeAt(0); code++) {
      const col = String.fromCharCode(code);
      sheet.getRange(`${col}${row}`).conditionalFormats.clear();
    }
  }
}

function styleAucRow(sheet, row, lastCol, isLastGroup) {
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

function rebuildResultSheet(sheet) {
  const heights = new Map();
  for (let row = 1; row <= 98; row++) {
    heights.set(row, sheet.getRange(`A${row}:J${row}`).format.rowHeightPx);
  }

  sheet.getRange("A1:J130").unmerge();

  for (let block = 5; block >= 0; block--) {
    const originalStart = 4 + block * 16;
    const newStart = 4 + block * 20;
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

    if (block < 5) {
      const blankTarget = newStart + 19;
      sheet.getRange(`A${blankTarget}:J${blankTarget}`).copyFrom(sheet.getRange("A19:J19"), "all");
      sheet.getRange(`A${blankTarget}:J${blankTarget}`).clear({ applyTo: "contents" });
    }

    rowMap.sort((a, b) => b[0] - a[0]);
    for (const [sourceRow, targetRow] of rowMap) {
      sheet.getRange(`A${targetRow}:J${targetRow}`).copyFrom(sheet.getRange(`A${sourceRow}:J${sourceRow}`), "all");
      const height = heights.get(sourceRow);
      if (height !== undefined) sheet.getRange(`A${targetRow}:J${targetRow}`).format.rowHeightPx = height;
    }

    const lastCol = block < 4 ? "J" : "F";
    for (let group = 0; group < 4; group++) {
      const groupStart = newStart + 3 + group * 4;
      const fnrRow = groupStart;
      const fprRow = groupStart + 1;
      const f1Row = groupStart + 2;
      const aucRow = groupStart + 3;

      sheet.getRange(`A${aucRow}:J${aucRow}`).copyFrom(sheet.getRange(`A${f1Row}:J${f1Row}`), "all");
      sheet.getRange(`A${aucRow}:J${aucRow}`).clear({ applyTo: "contents" });
      clearConditionalFormatsCellwise(sheet, aucRow, aucRow);
      sheet.getRange(`B${aucRow}`).values = [["AUC"]];

      const startCode = "C".charCodeAt(0);
      const endCode = lastCol.charCodeAt(0);
      for (let code = startCode; code <= endCode; code++) {
        const col = String.fromCharCode(code);
        sheet.getRange(`${col}${aucRow}`).formulas = [[`=IF(COUNT(${col}${fnrRow}:${col}${fprRow})<2,"N/A",1-(${col}${fnrRow}+${col}${fprRow})/2)`]];
      }

      sheet.getRange(`A${f1Row}:${lastCol}${f1Row}`).format.borders = { bottom: thin };
      styleAucRow(sheet, aucRow, lastCol, group === 3);

    }
  }

  sheet.mergeCells("A1:J1");
  sheet.mergeCells("A2:J2");
  const note = String(sheet.getRange("A2").values?.[0]?.[0] ?? "");
  if (!note.includes("AUC=")) sheet.getRange("A2").values = [[`${note} AUC=1-(FNR+FPR)/2; higher is better, and the better q95/fixed AUC is bold within each diagnostic.`]];

  for (let block = 0; block < 6; block++) {
    const start = 4 + block * 20;
    const lastCol = block < 4 ? "J" : "F";
    sheet.mergeCells(`A${start}:${lastCol}${start}`);
    sheet.mergeCells(`A${start + 1}:A${start + 2}`);
    sheet.mergeCells(`B${start + 1}:B${start + 2}`);
    if (block < 4) {
      for (const [left, right] of [["C", "D"], ["E", "F"], ["G", "H"], ["I", "J"]]) {
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

  // Imported rules are scoped per cell, so clear each exact cell before adding
  // the intended row-level rules. This prevents copied F1 rules from overriding
  // the explicit AUC winner formatting when Excel opens the exported workbook.
  clearConditionalFormatsCellwise(sheet, 1, 130);

  sheet.getRange("A1:J1").format = {
    fill: navy,
    font: { bold: true, fontSize: 16, typeface: "Carlito", color: "#FFFFFF" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
  };
  sheet.getRange("A1:J1").format.rowHeightPx = 34;
  sheet.getRange("A2:J2").format = {
    fill: "#EAF2F8",
    font: { italic: true, fontSize: 9, typeface: "Carlito", color: "#4B5563" },
    horizontalAlignment: "left",
    verticalAlignment: "center",
    wrapText: true,
  };
  sheet.getRange("A2:J2").format.rowHeightPx = 42;
  sheet.getRange("A3:J3").format.rowHeightPx = 14;

  for (let block = 0; block < 6; block++) {
    const start = 4 + block * 20;
    const lastCol = block < 4 ? "J" : "F";
    const title = sheet.getRange(`A${start}:${lastCol}${start}`);
    title.format = {
      fill: sectionFill,
      font: { bold: true, fontSize: 13, typeface: "Carlito", color: navy },
      horizontalAlignment: "center",
      verticalAlignment: "center",
      borders: { top: blueBorder, bottom: medium, left: medium, right: medium },
    };
    title.format.rowHeightPx = 34.6666666667;

    const header = sheet.getRange(`A${start + 1}:${lastCol}${start + 2}`);
    header.format = {
      fill: headerFill,
      font: { bold: true, fontSize: 10, typeface: "Carlito", color: textColor },
      horizontalAlignment: "center",
      verticalAlignment: "center",
      wrapText: true,
      borders: { preset: "all", style: "thin", color: "#A6AFBB" },
    };
    sheet.getRange(`A${start + 1}:${lastCol}${start + 1}`).format.rowHeightPx = 29.3333333333;
    sheet.getRange(`A${start + 2}:${lastCol}${start + 2}`).format.rowHeightPx = 29.3333333333;
    sheet.getRange(`A${start + 1}:A${start + 2}`).format.borders = { left: medium, right: thin, top: thin, bottom: thin };
    sheet.getRange(`${lastCol}${start + 1}:${lastCol}${start + 2}`).format.borders = { right: medium, left: thin, top: thin, bottom: thin };

    for (let group = 0; group < 4; group++) {
      const groupStart = start + 3 + group * 4;
      const groupEnd = groupStart + 3;
      const metricRows = [groupStart, groupStart + 1, groupStart + 2, groupStart + 3];

      sheet.getRange(`A${groupStart}:B${groupEnd}`).format = {
        fill: labelFill,
        font: { bold: false, fontSize: 11, typeface: "Carlito", color: textColor },
        horizontalAlignment: "center",
        verticalAlignment: "center",
        borders: { preset: "all", style: "thin", color: "#D9DEE5" },
      };

      if (block < 4) {
        for (const col of ["C", "E", "G", "I"]) {
          sheet.getRange(`${col}${groupStart}:${col}${groupEnd}`).format.fill = q95Fill;
        }
        for (const col of ["D", "F", "H", "J"]) {
          sheet.getRange(`${col}${groupStart}:${col}${groupEnd}`).format.fill = "#FFFFFF";
        }
      } else {
        sheet.getRange(`C${groupStart}:F${groupEnd}`).format.fill = q95Fill;
      }

      const numbers = sheet.getRange(`C${groupStart}:${lastCol}${groupEnd}`);
      numbers.format.font = { bold: false, fontSize: 11, typeface: "Carlito", color: textColor };
      numbers.format.horizontalAlignment = "right";
      numbers.format.verticalAlignment = "center";
      numbers.format.numberFormat = "0.0%";
      numbers.format.borders = { preset: "all", style: "thin", color: "#D9DEE5" };

      sheet.getRange(`A${groupStart}:A${groupEnd}`).format.borders = { left: medium, right: thin, top: thin, bottom: group === 3 ? medium : thin };
      sheet.getRange(`${lastCol}${groupStart}:${lastCol}${groupEnd}`).format.borders = { right: medium, left: thin, top: thin, bottom: group === 3 ? medium : thin };
      sheet.getRange(`A${groupEnd}:${lastCol}${groupEnd}`).format.borders = {
        bottom: group === 3 ? medium : thin,
        left: medium,
        right: medium,
        insideVertical: thin,
        top: thin,
      };

      for (const row of metricRows) sheet.getRange(`A${row}:${lastCol}${row}`).format.rowHeightPx = 26.6666666667;

      const rules = [
        [groupStart, "MIN"],
        [groupStart + 1, "MIN"],
        [groupStart + 2, "MAX"],
      ];
      for (const [row, fn] of rules) {
        sheet.getRange(`C${row}:${lastCol}${row}`).conditionalFormats.add("expression", {
          formula: `=AND(ISNUMBER(C${row}),C${row}=${fn}($C${row}:$${lastCol}${row}))`,
          format: { font: { bold: true } },
        });
      }

      if (block < 4) {
        const aucRow = groupStart + 3;
        for (const [left, right] of [["C", "D"], ["E", "F"], ["G", "H"], ["I", "J"]]) {
          const [leftValue, rightValue] = sheet.getRange(`${left}${aucRow}:${right}${aucRow}`).values[0];
          const leftNumeric = typeof leftValue === "number";
          const rightNumeric = typeof rightValue === "number";
          if (leftNumeric && (!rightNumeric || leftValue > rightValue)) {
            sheet.getRange(`${left}${aucRow}`).format.font = { bold: true, fontSize: 11, typeface: "Carlito", color: textColor };
          } else if (rightNumeric && (!leftNumeric || rightValue > leftValue)) {
            sheet.getRange(`${right}${aucRow}`).format.font = { bold: true, fontSize: 11, typeface: "Carlito", color: textColor };
          } else if (leftNumeric && rightNumeric && Math.abs(leftValue - rightValue) <= 1e-12) {
            sheet.getRange(`${left}${aucRow}:${right}${aucRow}`).format.font = { bold: true, fontSize: 11, typeface: "Carlito", color: textColor };
          }
        }
      }
    }

    if (block < 5) {
      const blank = start + 19;
      sheet.getRange(`A${blank}:J${blank}`).clear({ applyTo: "all" });
      sheet.getRange(`A${blank}:J${blank}`).format.rowHeightPx = 14;
    }
  }

  sheet.getRange("G84:J123").clear({ applyTo: "all" });
  sheet.showGridLines = false;
}

rebuildResultSheet(workbook.worksheets.getItem("Simulated data"));
rebuildResultSheet(workbook.worksheets.getItem("Empirical data"));
styleResultSheet(workbook.worksheets.getItem("Simulated data"));
styleResultSheet(workbook.worksheets.getItem("Empirical data"));

for (const sheetName of ["Simulated data", "Empirical data"]) {
  const preview = await workbook.render({ sheetName, autoCrop: "all", scale: 1, format: "png" });
  await fs.writeFile(`${previewDir}/${sheetName.replace(/[^A-Za-z0-9_.-]+/g, "_")}.png`, new Uint8Array(await preview.arrayBuffer()));
}
const sourcePreview = await workbook.render({ sheetName: "Source rates", autoCrop: "all", scale: 0.5, format: "png" });
await fs.writeFile(`${previewDir}/Source_rates.png`, new Uint8Array(await sourcePreview.arrayBuffer()));

for (const sheetName of ["Simulated data", "Empirical data"]) {
  const check = await workbook.inspect({
    kind: "table",
    sheetId: sheetName,
    range: "A4:J23",
    include: "values,formulas",
    tableMaxRows: 20,
    tableMaxCols: 10,
    maxChars: 10000,
  });
  console.log(`CHECK ${sheetName}`);
  console.log(check.ndjson);
}

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});
console.log("ERROR_SCAN");
console.log(errors.ndjson);

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
const saved = await FileBlob.load(outputPath);
const reopened = await SpreadsheetFile.importXlsx(saved);
for (const sheetName of ["Simulated data", "Empirical data"]) {
  const secondBlock = await reopened.inspect({
    kind: "table",
    sheetId: sheetName,
    range: "A24:J43",
    include: "values,formulas",
    tableMaxRows: 20,
    tableMaxCols: 10,
    maxChars: 5000,
  });
  const finalBlock = await reopened.inspect({
    kind: "table",
    sheetId: sheetName,
    range: "A104:F122",
    include: "values,formulas",
    tableMaxRows: 19,
    tableMaxCols: 6,
    maxChars: 5000,
  });
  console.log(`REOPENED_SECOND ${sheetName}`);
  console.log(secondBlock.ndjson);
  console.log(`REOPENED_FINAL ${sheetName}`);
  console.log(finalBlock.ndjson);
}
const reopenedErrors = await reopened.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 300 },
  summary: "post-export formula error scan",
});
console.log("POST_EXPORT_ERROR_SCAN");
console.log(reopenedErrors.ndjson);
console.log(`OUTPUT ${outputPath}`);
