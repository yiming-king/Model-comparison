import fs from "node:fs/promises";
import path from "node:path";

import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";


const root = "/Users/yimingzang/Documents/Project/benchmark2";
const workDir = path.join(root, ".codex_spreadsheet_work", "gaussian_interval_20260831");
const inputDir = path.join(root, "outputs", "20260827_reordered_diffusion_plots");
const outputDir = path.join(root, "outputs", "20260831_gaussian_90_interval");
const files = [
  "FNR-FPR tables without MMD rerun1.xlsx",
  "FNR-FPR tables without MMD.xlsx",
];
const computed = JSON.parse(
  await fs.readFile(path.join(workDir, "computed_gaussian_90.json"), "utf8"),
);

const DIAGNOSTICS = ["l2", "linf", "density", "mmd"];
const SUMMARIES = ["S=D", "S=2D", "S=4D"];
const STATISTICS = ["FNR", "FPR", "F1", "AUC"];
const STAT_INDEX = { FNR: 14, FPR: 15, F1: 16, AUC: 17 };
const BLOCKS = [
  { metric: "signed_logml_error", model: "m1", start: 5 },
  { metric: "signed_logml_error", model: "m2", start: 21 },
  { metric: "signed_logml_error", model: "m3", start: 37 },
  { metric: "signed_logml_error", model: "m4", start: 53 },
  { metric: "pmp_error", model: "m1", start: 69 },
  { metric: "pmp_error", model: "m2", start: 85 },
  { metric: "pmp_error", model: "m3", start: 101 },
  { metric: "pmp_error", model: "m4", start: 117 },
];
const SUBTITLE = "Raw, network-specific NPE–analytical central 90% interval [q5, q95]";


function key(row) {
  return row.slice(0, 6).join("|");
}


function assert(condition, message) {
  if (!condition) throw new Error(message);
}


function equalJson(actual, expected, message) {
  if (JSON.stringify(actual) !== JSON.stringify(expected)) throw new Error(message);
}


function valuesMatch(actual, expected, tolerance = 1e-10) {
  if (actual === null || expected === null) return actual === expected;
  if (typeof actual === "number" && typeof expected === "number") {
    return Math.abs(actual - expected) <= tolerance;
  }
  return actual === expected;
}


function chartSnapshot(workbook) {
  const result = {};
  for (const sheet of workbook.worksheets.items) {
    if (!sheet.charts.items.length) continue;
    result[sheet.name] = sheet.charts.items.map((chart) => ({
      title: chart.title?.text ?? String(chart.title ?? ""),
      type: chart.type,
      width: chart.width,
      height: chart.height,
      hasLegend: chart.hasLegend,
      series: chart.series.items.map((series) => ({
        name: series.name,
        formula: series.formula,
        categoryFormula: series.categoryFormula,
        lineStyle: series.line.style,
        lineWidth: series.line.width,
        markerSymbol: series.marker.symbol,
        markerSize: series.marker.size,
      })),
    }));
  }
  return result;
}


async function formulaErrorSignature(workbook) {
  const result = await workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
    options: { useRegex: true, maxResults: 500 },
    maxChars: 50000,
  });
  return result.ndjson
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line))
    .filter((row) => row.kind === "match")
    .map((row) => `${row.sheet}!${row.address}|${row.value}|${row.formula ?? ""}`)
    .sort();
}


function updateSourceRates(workbook, computedMap) {
  const sheet = workbook.worksheets.getItem("Source rates");
  const identifiers = sheet.getRange("A194:F289").values;
  const replacement = [];
  for (const row of identifiers) {
    assert(row[0] === "Gaussian", `Unexpected case study in Gaussian source block: ${row}`);
    assert(
      row[5] === "signed_logml_error" || row[5] === "pmp_error",
      `Unexpected metric in Gaussian signed-error block: ${row[5]}`,
    );
    const source = computedMap.get(key(row));
    assert(source, `Missing recomputed Gaussian row for ${key(row)}`);
    replacement.push(source.slice(6, 18));
  }
  assert(replacement.length === 96, `Expected 96 Gaussian signed-error rows; found ${replacement.length}`);
  sheet.getRange("G194:R289").values = replacement;
  return replacement.length;
}


function updateMethodAndLabels(workbook) {
  const method = workbook.worksheets.getItem("Method");
  method.getRange("B7").values = [[
    "e=estimated logML - reference logML. Diffusion reports log10 units and uses ESS-filtered L=q5(eref), U=q95(eref); Gaussian keeps natural-log units and uses L=q5(eref), U=q95(eref).",
  ]];
  method.getRange("B8").values = [[
    "e=estimated PMP - reference PMP. Diffusion uses ESS-filtered L=q5(eref), U=q95(eref); Gaussian uses L=q5(eref), U=q95(eref). No absolute value or normalization is used.",
  ]];

  const gaussian = workbook.worksheets.getItem("Gaussian data");
  for (const block of BLOCKS) {
    gaussian.getRange(`A${block.start + 1}`).values = [[SUBTITLE]];
  }
}


function updateGaussianBolding(workbook, computedMap) {
  const gaussian = workbook.worksheets.getItem("Gaussian data");
  let boldChecks = 0;
  for (const block of BLOCKS) {
    const dataRange = gaussian.getRange(`C${block.start + 3}:F${block.start + 14}`);
    dataRange.format.font.bold = false;
    dataRange.format.font.color = "#263442";
    for (let summaryIndex = 0; summaryIndex < SUMMARIES.length; summaryIndex += 1) {
      const summary = SUMMARIES[summaryIndex];
      for (let statisticIndex = 0; statisticIndex < STATISTICS.length; statisticIndex += 1) {
        const statistic = STATISTICS[statisticIndex];
        const values = DIAGNOSTICS.map((diagnostic) => {
          const row = computedMap.get([
            "Gaussian", "simulated", diagnostic, block.model, summary, block.metric,
          ].join("|"));
          assert(row, `Missing bolding row for ${block.metric}/${block.model}/${summary}/${diagnostic}`);
          return row[STAT_INDEX[statistic]];
        });
        const numeric = values.filter((value) => typeof value === "number" && Number.isFinite(value));
        if (!numeric.length) continue;
        const winner = statistic === "FNR" || statistic === "FPR"
          ? Math.min(...numeric)
          : Math.max(...numeric);
        const sheetRow = block.start + 3 + summaryIndex * 4 + statisticIndex;
        values.forEach((value, diagnosticIndex) => {
          const cell = gaussian.getRangeByIndexes(sheetRow - 1, 2 + diagnosticIndex, 1, 1);
          const isWinner = typeof value === "number"
            && Number.isFinite(value)
            && Math.abs(value - winner) <= 1e-12;
          if (isWinner) {
            cell.format.font.bold = true;
            cell.format.font.color = "#203864";
          }
          boldChecks += 1;
        });
      }
    }
  }
  return boldChecks;
}


async function renderAll(workbook, fileName) {
  const previewDir = path.join(workDir, "after", fileName.replace(/\.xlsx$/i, ""));
  await fs.mkdir(previewDir, { recursive: true });
  for (let index = 0; index < workbook.worksheets.items.length; index += 1) {
    const sheet = workbook.worksheets.items[index];
    const preview = await workbook.render({
      sheetName: sheet.name,
      autoCrop: "all",
      scale: 0.65,
      format: "png",
    });
    const safeName = sheet.name.replace(/[^a-z0-9._-]+/gi, "_");
    await fs.writeFile(
      path.join(previewDir, `${String(index).padStart(2, "0")}_${safeName}.png`),
      new Uint8Array(await preview.arrayBuffer()),
    );
  }
}


async function verifyWorkbook(
  workbook,
  computedMap,
  beforeCharts,
  beforeFormulas,
  beforeFormulaErrors,
  fileName,
) {
  const source = workbook.worksheets.getItem("Source rates");
  const sourceRows = source.getRange("A194:R289").values;
  let sourceChecks = 0;
  for (const row of sourceRows) {
    const expected = computedMap.get(key(row));
    assert(expected, `${fileName}: missing verification row for ${key(row)}`);
    for (let column = 0; column < 18; column += 1) {
      assert(
        valuesMatch(row[column], expected[column]),
        `${fileName}: source mismatch for ${key(row)} at column ${column + 1}`,
      );
      sourceChecks += 1;
    }
  }

  const gaussian = workbook.worksheets.getItem("Gaussian data");
  let valueChecks = 0;
  let boldChecks = 0;
  for (const block of BLOCKS) {
    assert(
      gaussian.getRange(`A${block.start + 1}`).values[0][0] === SUBTITLE,
      `${fileName}: subtitle mismatch at Gaussian data!A${block.start + 1}`,
    );
    for (let summaryIndex = 0; summaryIndex < SUMMARIES.length; summaryIndex += 1) {
      const summary = SUMMARIES[summaryIndex];
      for (let statisticIndex = 0; statisticIndex < STATISTICS.length; statisticIndex += 1) {
        const statistic = STATISTICS[statisticIndex];
        const expectedValues = DIAGNOSTICS.map((diagnostic) => computedMap.get([
          "Gaussian", "simulated", diagnostic, block.model, summary, block.metric,
        ].join("|"))[STAT_INDEX[statistic]]);
        const numeric = expectedValues.filter((value) => typeof value === "number" && Number.isFinite(value));
        const winner = statistic === "FNR" || statistic === "FPR"
          ? Math.min(...numeric)
          : Math.max(...numeric);
        const sheetRow = block.start + 3 + summaryIndex * 4 + statisticIndex;
        for (let diagnosticIndex = 0; diagnosticIndex < DIAGNOSTICS.length; diagnosticIndex += 1) {
          const cell = gaussian.getRangeByIndexes(sheetRow - 1, 2 + diagnosticIndex, 1, 1);
          const actual = cell.values[0][0];
          const expected = expectedValues[diagnosticIndex];
          assert(valuesMatch(actual, expected), `${fileName}: Gaussian data value mismatch at row ${sheetRow}`);
          const shouldBeBold = typeof expected === "number"
            && Number.isFinite(expected)
            && Math.abs(expected - winner) <= 1e-12;
          assert(
            Boolean(cell.format.font.bold) === shouldBeBold,
            `${fileName}: Gaussian data bold-rule mismatch at row ${sheetRow}, diagnostic ${DIAGNOSTICS[diagnosticIndex]}`,
          );
          valueChecks += 1;
          boldChecks += 1;
        }
      }
    }
  }

  equalJson(chartSnapshot(workbook), beforeCharts, `${fileName}: chart structure or order changed`);
  equalJson(
    workbook.worksheets.getItem("Gaussian data").getRange("C8:F195").formulas,
    beforeFormulas.gaussian,
    `${fileName}: Gaussian table formulas changed`,
  );
  equalJson(
    workbook.worksheets.getItem("Chart data").getRange("A1:O178").formulas,
    beforeFormulas.chartData,
    `${fileName}: Gaussian chart-data formulas changed`,
  );

  const stale = await workbook.inspect({
    kind: "match",
    searchTerm: "central 95%|q2\\.5|q97\\.5",
    options: { useRegex: true, maxResults: 100 },
    maxChars: 12000,
  });
  const staleMatches = stale.ndjson
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line))
    .filter((row) => row.kind === "match");
  assert(staleMatches.length === 0, `${fileName}: stale Gaussian 95% interval text remains`);

  const afterFormulaErrors = await formulaErrorSignature(workbook);
  equalJson(
    afterFormulaErrors,
    beforeFormulaErrors,
    `${fileName}: formula-error set changed during Gaussian interval update`,
  );

  return {
    sourceChecks,
    valueChecks,
    boldChecks,
    chartCounts: Object.fromEntries(
      Object.entries(beforeCharts).map(([sheet, charts]) => [sheet, charts.length]),
    ),
    preexistingFormulaPlaceholders: afterFormulaErrors.length,
    newFormulaErrors: 0,
    staleIntervalLabels: staleMatches.length,
  };
}


const computedMap = new Map(computed.rows.map((row) => [key(row), row]));
assert(computedMap.size === 144, `Expected 144 recomputed Gaussian rows; found ${computedMap.size}`);
await fs.mkdir(outputDir, { recursive: true });

for (const fileName of files) {
  const inputPath = path.join(inputDir, fileName);
  const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(inputPath));
  const beforeCharts = chartSnapshot(workbook);
  const beforeFormulaErrors = await formulaErrorSignature(workbook);
  const beforeFormulas = {
    gaussian: workbook.worksheets.getItem("Gaussian data").getRange("C8:F195").formulas,
    chartData: workbook.worksheets.getItem("Chart data").getRange("A1:O178").formulas,
  };

  const sourceRowsUpdated = updateSourceRates(workbook, computedMap);
  updateMethodAndLabels(workbook);
  const boldCellsUpdated = updateGaussianBolding(workbook, computedMap);

  const outputPath = path.join(outputDir, fileName);
  const output = await SpreadsheetFile.exportXlsx(workbook);
  await output.save(outputPath);

  const reloaded = await SpreadsheetFile.importXlsx(await FileBlob.load(outputPath));
  const verification = await verifyWorkbook(
    reloaded,
    computedMap,
    beforeCharts,
    beforeFormulas,
    beforeFormulaErrors,
    fileName,
  );
  await renderAll(reloaded, fileName);
  const record = {
    inputPath,
    outputPath,
    sourceRowsUpdated,
    boldCellsUpdated,
    ...verification,
  };
  await fs.writeFile(
    path.join(workDir, `${fileName}.verification.json`),
    JSON.stringify(record, null, 2),
    "utf8",
  );
  process.stdout.write(`${JSON.stringify(record)}\n`);
}
