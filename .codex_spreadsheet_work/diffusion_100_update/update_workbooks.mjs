import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const projectRoot = "/Users/yimingzang/Documents/Project/benchmark2";
const workDir = path.join(projectRoot, ".codex_spreadsheet_work/diffusion_100_update");
const outputDir = path.join(
  projectRoot,
  "outputs/01a038b9-1f9b-7d01-9240-218e90236b72",
);

const workbookConfigs = [
  {
    fileName: "FNR-FPR tables without MMD rerun1.xlsx",
    methodText:
      "benchmark/examples/diffusion/calibration_outputs_100_noMMD_rerun1/thresholds.csv (S=D/2D/4D, n=100); S=6D retains calibration_outputs_rerun1/thresholds.csv (n=30).",
  },
  {
    fileName: "FNR-FPR tables without MMD.xlsx",
    methodText:
      "benchmark/examples/diffusion/calibration_outputs_100_noMMD/thresholds.csv (S=D/2D/4D, n=100); S=6D retains calibration_outputs/thresholds.csv (n=30).",
  },
];

const summaries = ["S=D", "S=2D", "S=4D", "S=6D"];
const diagnostics = ["l2", "linf", "density", "mmd"];
const statistics = ["FNR", "FPR", "F1", "AUC"];
const statisticIndexes = { FNR: 14, FPR: 15, F1: 16, AUC: 17 };
const blockSpecs = [
  { start: 5, metric: "log10_ml_error", model: "m1" },
  { start: 25, metric: "log10_ml_error", model: "m3" },
  { start: 45, metric: "pmp_error", model: "m1" },
  { start: 65, metric: "pmp_error", model: "m3" },
  { start: 85, metric: "posterior_mmd", model: "m1" },
  { start: 105, metric: "posterior_mmd", model: "m3" },
  { start: 125, metric: "log10_ml_error", model: "m0" },
  { start: 145, metric: "log10_ml_error", model: "m2" },
  { start: 165, metric: "pmp_error", model: "m0" },
  { start: 185, metric: "pmp_error", model: "m2" },
  { start: 205, metric: "posterior_mmd", model: "m0" },
  { start: 225, metric: "posterior_mmd", model: "m2" },
];

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function valuesMatch(actual, expected, tolerance = 1e-12) {
  if (typeof expected === "number") {
    return typeof actual === "number"
      && Number.isFinite(actual)
      && Math.abs(actual - expected) <= tolerance;
  }
  if (expected === "N/A") return actual === "N/A" || actual === "#N/A";
  return actual === expected;
}

function matricesMatch(left, right) {
  if (left.length !== right.length) return false;
  for (let rowIndex = 0; rowIndex < left.length; rowIndex += 1) {
    if (left[rowIndex].length !== right[rowIndex].length) return false;
    for (let columnIndex = 0; columnIndex < left[rowIndex].length; columnIndex += 1) {
      if (!valuesMatch(left[rowIndex][columnIndex], right[rowIndex][columnIndex])) return false;
    }
  }
  return true;
}

function sourceKey(row) {
  return [row[1], row[2], row[3], row[4], row[5]].join("|");
}

function columnLetter(columnNumber) {
  let value = columnNumber;
  let output = "";
  while (value > 0) {
    const remainder = (value - 1) % 26;
    output = String.fromCharCode(65 + remainder) + output;
    value = Math.floor((value - 1) / 26);
  }
  return output;
}

function currentChartSnapshot(workbook) {
  const snapshot = {};
  for (const sheetName of ["Plots - Simulated", "Plots - Empirical", "Plots - Gaussian"]) {
    snapshot[sheetName] = workbook.worksheets.getItem(sheetName).charts.items.map((chart) => ({
      title: typeof chart.title === "string" ? chart.title : chart.title?.text ?? null,
      seriesCount: chart.series.items.length,
    }));
  }
  return snapshot;
}

function updateSourceRates(workbook, payload) {
  const sourceSheet = workbook.worksheets.getItem("Source rates");
  const sourceRange = sourceSheet.getRange("A1:R529");
  const originalValues = sourceRange.values;
  const gaussianSnapshot = originalValues.slice(193, 337).map((row) => [...row]);
  const s6dSnapshot = new Map();
  originalValues.forEach((row, index) => {
    if (row?.[0] === "Diffusion" && row?.[4] === "S=6D") {
      s6dSnapshot.set(index + 1, [...row]);
    }
  });

  const valuesHtoR = sourceSheet.getRange("H2:R529").values;
  for (const replacement of payload.replacements) {
    valuesHtoR[replacement.excelRow - 2] = replacement.valuesHtoR;
  }
  sourceSheet.getRange("H2:R529").values = valuesHtoR;

  const updatedValues = sourceSheet.getRange("A1:R529").values;
  assert(
    matricesMatch(updatedValues.slice(193, 337), gaussianSnapshot),
    "Gaussian Source rates changed unexpectedly",
  );
  for (const [excelRow, snapshot] of s6dSnapshot) {
    assert(
      matricesMatch([updatedValues[excelRow - 1]], [snapshot]),
      `S=6D Source rates changed unexpectedly at row ${excelRow}`,
    );
  }
  for (const replacement of payload.replacements) {
    const actual = updatedValues[replacement.excelRow - 1].slice(7, 18);
    assert(
      matricesMatch([actual], [replacement.valuesHtoR]),
      `Replacement mismatch at Source rates row ${replacement.excelRow}`,
    );
  }
  return updatedValues;
}

function updateBoldingAndVerifySummaries(workbook, sourceValues) {
  const sourceLookup = new Map(
    sourceValues
      .filter((row) => row?.[0] === "Diffusion")
      .map((row) => [sourceKey(row), row]),
  );
  let boldChecks = 0;
  let formulaValueChecks = 0;
  for (const [sheetName, sourceGroup] of [
    ["Simulated data", "simulated"],
    ["Empirical data", "empirical"],
  ]) {
    const sheet = workbook.worksheets.getItem(sheetName);
    for (const block of blockSpecs) {
      const dataRange = sheet.getRange(`C${block.start + 3}:F${block.start + 18}`);
      dataRange.format.font.bold = false;
      for (let summaryIndex = 0; summaryIndex < summaries.length; summaryIndex += 1) {
        const summary = summaries[summaryIndex];
        for (let statisticIndex = 0; statisticIndex < statistics.length; statisticIndex += 1) {
          const statistic = statistics[statisticIndex];
          const expectedValues = diagnostics.map((diagnostic) => {
            const row = sourceLookup.get([
              sourceGroup,
              diagnostic,
              block.model,
              summary,
              block.metric,
            ].join("|"));
            assert(
              row,
              `Missing Source rates row for ${sourceGroup}/${diagnostic}/${block.model}/${summary}/${block.metric}`,
            );
            return row[statisticIndexes[statistic]];
          });
          const numericValues = expectedValues.filter(
            (value) => typeof value === "number" && Number.isFinite(value),
          );
          const winner = numericValues.length
            ? ((statistic === "FNR" || statistic === "FPR")
              ? Math.min(...numericValues)
              : Math.max(...numericValues))
            : null;
          const sheetRow = block.start + 3 + summaryIndex * 4 + statisticIndex;
          expectedValues.forEach((expectedValue, diagnosticIndex) => {
            const column = columnLetter(3 + diagnosticIndex);
            const cell = sheet.getRange(`${column}${sheetRow}`);
            assert(
              valuesMatch(cell.values[0][0], expectedValue),
              `${sheetName}!${column}${sheetRow} does not match Source rates`,
            );
            const shouldBeBold = winner !== null
              && typeof expectedValue === "number"
              && Math.abs(expectedValue - winner) <= 1e-12;
            if (shouldBeBold) cell.format.font.bold = true;
            formulaValueChecks += 1;
          });
        }
      }
    }

    for (const block of blockSpecs) {
      for (let summaryIndex = 0; summaryIndex < summaries.length; summaryIndex += 1) {
        for (let statisticIndex = 0; statisticIndex < statistics.length; statisticIndex += 1) {
          const statistic = statistics[statisticIndex];
          const summary = summaries[summaryIndex];
          const expectedValues = diagnostics.map((diagnostic) => sourceLookup.get([
            sourceGroup,
            diagnostic,
            block.model,
            summary,
            block.metric,
          ].join("|"))[statisticIndexes[statistic]]);
          const numericValues = expectedValues.filter(
            (value) => typeof value === "number" && Number.isFinite(value),
          );
          const winner = numericValues.length
            ? ((statistic === "FNR" || statistic === "FPR")
              ? Math.min(...numericValues)
              : Math.max(...numericValues))
            : null;
          const sheetRow = block.start + 3 + summaryIndex * 4 + statisticIndex;
          expectedValues.forEach((expectedValue, diagnosticIndex) => {
            const column = columnLetter(3 + diagnosticIndex);
            const shouldBeBold = winner !== null
              && typeof expectedValue === "number"
              && Math.abs(expectedValue - winner) <= 1e-12;
            const actualBold = Boolean(sheet.getRange(`${column}${sheetRow}`).format.font.bold);
            assert(
              actualBold === shouldBeBold,
              `${sheetName}!${column}${sheetRow} bold-rule mismatch`,
            );
            boldChecks += 1;
          });
        }
      }
    }
  }
  return { boldChecks, formulaValueChecks };
}

async function renderAllSheets(workbook, fileName) {
  const previewDir = path.join(
    workDir,
    "previews_after",
    fileName.replace(/\.xlsx$/i, "").replaceAll(" ", "_"),
  );
  await fs.mkdir(previewDir, { recursive: true });
  const rendered = [];
  for (let index = 0; index < workbook.worksheets.items.length; index += 1) {
    const sheet = workbook.worksheets.items[index];
    const preview = await workbook.render({
      sheetName: sheet.name,
      autoCrop: "all",
      scale: sheet.name === "Source rates" ? 0.4 : 0.7,
      format: "png",
    });
    const safeName = sheet.name.replace(/[^a-z0-9._-]+/gi, "_");
    const previewPath = path.join(
      previewDir,
      `${String(index).padStart(2, "0")}_${safeName}.png`,
    );
    await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));
    rendered.push(previewPath);
  }
  return rendered;
}

async function updateWorkbook(config) {
  const inputPath = path.join(projectRoot, "FNR-FPR", config.fileName);
  const payloadPath = path.join(workDir, `${config.fileName}.payload.json`);
  const payload = JSON.parse(await fs.readFile(payloadPath, "utf8"));
  const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(inputPath));
  const chartSnapshotBefore = currentChartSnapshot(workbook);
  const gaussianDataSnapshot = workbook.worksheets.getItem("Gaussian data").getRange("A1:F196").formulas;

  const sourceValues = updateSourceRates(workbook, payload);
  const summaryChecks = updateBoldingAndVerifySummaries(workbook, sourceValues);

  const methodSheet = workbook.worksheets.getItem("Method");
  methodSheet.getRange("B17").values = [[config.methodText]];
  methodSheet.getRange("B17").format.wrapText = true;
  methodSheet.getRange("A17:F17").format.rowHeight = 32;
  methodSheet.getRange("B20").values = [["2026-08-31"]];

  const chartSnapshotAfter = currentChartSnapshot(workbook);
  assert(
    JSON.stringify(chartSnapshotAfter) === JSON.stringify(chartSnapshotBefore),
    `${config.fileName}: chart titles or series counts changed unexpectedly`,
  );
  assert(
    JSON.stringify(workbook.worksheets.getItem("Gaussian data").getRange("A1:F196").formulas)
      === JSON.stringify(gaussianDataSnapshot),
    `${config.fileName}: Gaussian formulas changed unexpectedly`,
  );

  const formulaErrors = await workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?",
    options: { useRegex: true, maxResults: 300 },
    summary: "final formula error scan excluding intentional N/A",
    maxChars: 10000,
  });
  assert(
    formulaErrors.ndjson.includes("matched 0 entries"),
    `${config.fileName}: formula error scan failed: ${formulaErrors.ndjson}`,
  );

  const keyInspection = await workbook.inspect({
    kind: "table",
    sheetId: "Source rates",
    range: "A1:R12",
    include: "values,formulas",
    tableMaxRows: 12,
    tableMaxCols: 18,
    maxChars: 12000,
  });
  const rendered = await renderAllSheets(workbook, config.fileName);

  await fs.mkdir(outputDir, { recursive: true });
  const outputPath = path.join(outputDir, config.fileName);
  const output = await SpreadsheetFile.exportXlsx(workbook);
  await output.save(outputPath);

  const report = {
    fileName: config.fileName,
    outputPath,
    updatedRows: payload.replacement_count,
    preservedS6DRows: payload.preserved_s6d_rows,
    changedRows: payload.changed_rows,
    changedCells: payload.changed_cells,
    summaryChecks,
    renderedSheets: rendered.length,
    formulaErrors: 0,
    chartCounts: Object.fromEntries(
      Object.entries(chartSnapshotAfter).map(([sheetName, charts]) => [sheetName, charts.length]),
    ),
    keyInspection: keyInspection.ndjson,
  };
  await fs.writeFile(
    path.join(workDir, `${config.fileName}.update-report.json`),
    JSON.stringify(report, null, 2),
  );
  console.log(JSON.stringify(report));
}

for (const config of workbookConfigs) {
  await updateWorkbook(config);
}
