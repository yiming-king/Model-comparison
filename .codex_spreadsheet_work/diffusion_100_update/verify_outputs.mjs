import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const projectRoot = "/Users/yimingzang/Documents/Project/benchmark2";
const workDir = path.join(projectRoot, ".codex_spreadsheet_work/diffusion_100_update");
const outputDir = path.join(projectRoot, "outputs/01a038b9-1f9b-7d01-9240-218e90236b72");
const fileNames = [
  "FNR-FPR tables without MMD rerun1.xlsx",
  "FNR-FPR tables without MMD.xlsx",
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

function rowMatch(left, right) {
  return left.length === right.length && left.every((value, index) => valuesMatch(value, right[index]));
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

function chartSnapshot(workbook) {
  const snapshot = {};
  for (const sheetName of ["Plots - Simulated", "Plots - Empirical", "Plots - Gaussian"]) {
    snapshot[sheetName] = workbook.worksheets.getItem(sheetName).charts.items.map((chart) => ({
      title: typeof chart.title === "string" ? chart.title : chart.title?.text ?? null,
      seriesCount: chart.series.items.length,
    }));
  }
  return snapshot;
}

for (const fileName of fileNames) {
  const payload = JSON.parse(await fs.readFile(path.join(workDir, `${fileName}.payload.json`), "utf8"));
  const current = JSON.parse(await fs.readFile(path.join(workDir, `${fileName}.current.json`), "utf8"));
  const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(path.join(outputDir, fileName)));
  const sourceValues = workbook.worksheets.getItem("Source rates").getRange("A1:R529").values;
  let replacementChecks = 0;
  for (const replacement of payload.replacements) {
    assert(
      rowMatch(sourceValues[replacement.excelRow - 1].slice(7, 18), replacement.valuesHtoR),
      `${fileName}: Source rates mismatch at row ${replacement.excelRow}`,
    );
    replacementChecks += 1;
  }
  assert(
    sourceValues.slice(193, 337).every((row, index) => rowMatch(row, current.sourceValues[index + 193])),
    `${fileName}: Gaussian Source rates changed`,
  );
  assert(
    sourceValues.every((row, index) => (
      row?.[0] !== "Diffusion"
      || row?.[4] !== "S=6D"
      || rowMatch(row, current.sourceValues[index])
    )),
    `${fileName}: S=6D rows changed`,
  );

  const sourceLookup = new Map(
    sourceValues.filter((row) => row?.[0] === "Diffusion").map((row) => [sourceKey(row), row]),
  );
  let summaryValueChecks = 0;
  let boldChecks = 0;
  for (const [sheetName, sourceGroup] of [["Simulated data", "simulated"], ["Empirical data", "empirical"]]) {
    const sheet = workbook.worksheets.getItem(sheetName);
    for (const block of blockSpecs) {
      for (let summaryIndex = 0; summaryIndex < summaries.length; summaryIndex += 1) {
        const summary = summaries[summaryIndex];
        for (let statisticIndex = 0; statisticIndex < statistics.length; statisticIndex += 1) {
          const statistic = statistics[statisticIndex];
          const expectedValues = diagnostics.map((diagnostic) => sourceLookup.get([
            sourceGroup, diagnostic, block.model, summary, block.metric,
          ].join("|"))[statisticIndexes[statistic]]);
          const numericValues = expectedValues.filter((value) => typeof value === "number" && Number.isFinite(value));
          const winner = numericValues.length
            ? ((statistic === "FNR" || statistic === "FPR") ? Math.min(...numericValues) : Math.max(...numericValues))
            : null;
          const rowNumber = block.start + 3 + summaryIndex * 4 + statisticIndex;
          expectedValues.forEach((expectedValue, diagnosticIndex) => {
            const column = columnLetter(3 + diagnosticIndex);
            const cell = sheet.getRange(`${column}${rowNumber}`);
            assert(valuesMatch(cell.values[0][0], expectedValue), `${fileName}: ${sheetName}!${column}${rowNumber} value mismatch`);
            const shouldBeBold = winner !== null
              && typeof expectedValue === "number"
              && Math.abs(expectedValue - winner) <= 1e-12;
            assert(Boolean(cell.format.font.bold) === shouldBeBold, `${fileName}: ${sheetName}!${column}${rowNumber} bold mismatch`);
            summaryValueChecks += 1;
            boldChecks += 1;
          });
        }
      }
    }
  }

  assert(
    JSON.stringify(chartSnapshot(workbook)) === JSON.stringify(current.chartTitles),
    `${fileName}: chart titles or series counts changed`,
  );
  const methodValues = workbook.worksheets.getItem("Method").getRange("A1:B20").values;
  assert(String(methodValues[16][1]).includes("100_noMMD"), `${fileName}: Method threshold path not updated`);
  assert(methodValues[19][1] === "2026-08-31", `${fileName}: Method update date not updated`);
  const errors = await workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?",
    options: { useRegex: true, maxResults: 300 },
    maxChars: 10000,
  });
  assert(errors.ndjson.includes("matched 0 entries"), `${fileName}: formula error scan failed`);

  const result = {
    fileName,
    replacementChecks,
    gaussianRowsUnchanged: 144,
    s6dRowsUnchanged: payload.preserved_s6d_rows,
    summaryValueChecks,
    boldChecks,
    formulaErrors: 0,
    chartCounts: Object.fromEntries(Object.entries(chartSnapshot(workbook)).map(([name, charts]) => [name, charts.length])),
    methodThreshold: methodValues[16][1],
  };
  await fs.writeFile(path.join(workDir, `${fileName}.final-verification.json`), JSON.stringify(result, null, 2));
  console.log(JSON.stringify(result));
}
