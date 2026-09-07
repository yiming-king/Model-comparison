import fs from "node:fs/promises";
import path from "node:path";
import crypto from "node:crypto";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const root = "/Users/yimingzang/Documents/Project/benchmark2";
const workDir = path.join(root, ".codex_spreadsheet_work", "reorder_diffusion_plots_20260827");
const inputDir = path.join(root, "FNR-FPR");
const outputDir = path.join(root, "outputs", "20260827_reordered_diffusion_plots");
const files = [
  "FNR-FPR tables without MMD rerun1.xlsx",
  "FNR-FPR tables without MMD.xlsx",
];
const plotSheets = ["Plots - Simulated", "Plots - Empirical"];
const desiredGroups = [
  ["Posterior MMD", "M0"],
  ["Posterior MMD", "M1"],
  ["Posterior MMD", "M2"],
  ["Posterior MMD", "M3"],
  ["Signed log10 ML error", "M0"],
  ["Signed log10 ML error", "M1"],
  ["Signed log10 ML error", "M2"],
  ["Signed log10 ML error", "M3"],
  ["Signed PMP error", "M0"],
  ["Signed PMP error", "M1"],
  ["Signed PMP error", "M2"],
  ["Signed PMP error", "M3"],
];

async function loadWorkbook(filePath) {
  return SpreadsheetFile.importXlsx(await FileBlob.load(filePath));
}

async function inspectAndRender(fileName) {
  const workbook = await loadWorkbook(path.join(inputDir, fileName));
  const inspection = await workbook.inspect({
    kind: "drawing",
    include: "id,type,title,position,series",
    maxChars: 40000,
  });
  await fs.writeFile(path.join(workDir, `${fileName}.before-drawings.ndjson`), inspection.ndjson, "utf8");
  const previewDir = path.join(workDir, "before", fileName.replace(/\.xlsx$/i, ""));
  await fs.mkdir(previewDir, { recursive: true });
  for (const sheetName of plotSheets) {
    const preview = await workbook.render({ sheetName, autoCrop: "all", scale: 0.8, format: "png" });
    await fs.writeFile(
      path.join(previewDir, `${sheetName.replace(/[^a-z0-9]+/gi, "_")}.png`),
      new Uint8Array(await preview.arrayBuffer()),
    );
  }
  console.log(JSON.stringify({ fileName, drawingInspection: inspection.ndjson.slice(0, 12000) }));
}

function chartTitle(chart) {
  return chart.title?.text ?? String(chart.title ?? "");
}

function chartSnapshot(chart) {
  return {
    title: chartTitle(chart),
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
  };
}

function plotChartSnapshots(workbook) {
  return Object.fromEntries(plotSheets.map((sheetName) => [
    sheetName,
    workbook.worksheets.getItem(sheetName).charts.items.map(chartSnapshot),
  ]));
}

function workbookCellDigests(workbook) {
  const result = {};
  for (const sheet of workbook.worksheets.items) {
    const used = sheet.getUsedRange();
    const payload = JSON.stringify({ address: used.address, values: used.values, formulas: used.formulas });
    result[sheet.name] = crypto.createHash("sha256").update(payload).digest("hex");
  }
  return result;
}

function assertEqual(actual, expected, message) {
  if (JSON.stringify(actual) !== JSON.stringify(expected)) throw new Error(message);
}

function reorderPlotSheet(sheet) {
  const charts = sheet.charts.items;
  if (charts.length !== 24) throw new Error(`${sheet.name}: expected 24 charts, found ${charts.length}`);
  const byTitle = new Map(charts.map((chart) => [chartTitle(chart), chart]));
  for (let groupIndex = 0; groupIndex < desiredGroups.length; groupIndex += 1) {
    const [metric, model] = desiredGroups[groupIndex];
    const topRow = 3 + groupIndex * 20;
    const placements = [
      { suffix: "FNR/FPR", anchor: `A${topRow}` },
      { suffix: "AUC", anchor: `L${topRow}` },
    ];
    for (const placement of placements) {
      const title = `${metric} — Assumed ${model} — ${placement.suffix}`;
      const chart = byTitle.get(title);
      if (!chart) throw new Error(`${sheet.name}: missing chart '${title}'`);
      chart.setPosition(sheet.getRange(placement.anchor));
      chart.width = 660;
      chart.height = 280;
    }
  }
}

async function visualOrder(workbook, sheetName) {
  const sheet = workbook.worksheets.getItem(sheetName);
  const drawingInspection = await workbook.inspect({
    kind: "drawing",
    sheetId: sheetName,
    include: "id,position",
    maxChars: 30000,
  });
  const drawings = drawingInspection.ndjson
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line))
    .filter((item) => item.kind === "drawing" && item.drawingType === "chart" && item.sheet === sheetName);
  if (drawings.length !== sheet.charts.items.length) {
    throw new Error(`${sheetName}: drawing/chart count mismatch`);
  }
  return drawings
    .map((drawing) => ({
      title: chartTitle(workbook.resolve(drawing.id)),
      row: drawing.anchor.from.row,
      col: drawing.anchor.from.col,
      widthPx: drawing.anchor.extent?.widthPx,
      heightPx: drawing.anchor.extent?.heightPx,
    }))
    .sort((left, right) => left.row - right.row || left.col - right.col);
}

function expectedVisualOrder() {
  return desiredGroups.flatMap(([metric, model], groupIndex) => [
    { title: `${metric} — Assumed ${model} — FNR/FPR`, row: 2 + groupIndex * 20, col: 0, widthPx: 660, heightPx: 280 },
    { title: `${metric} — Assumed ${model} — AUC`, row: 2 + groupIndex * 20, col: 11, widthPx: 660, heightPx: 280 },
  ]);
}

async function renderAllSheets(workbook, fileName) {
  const previewDir = path.join(workDir, "after", fileName.replace(/\.xlsx$/i, ""));
  await fs.mkdir(previewDir, { recursive: true });
  for (const sheet of workbook.worksheets.items) {
    const preview = await workbook.render({ sheetName: sheet.name, autoCrop: "all", scale: 0.8, format: "png" });
    await fs.writeFile(
      path.join(previewDir, `${String(sheet.index).padStart(2, "0")}_${sheet.name.replace(/[^a-z0-9]+/gi, "_")}.png`),
      new Uint8Array(await preview.arrayBuffer()),
    );
  }
}

async function editWorkbook(fileName) {
  const inputPath = path.join(inputDir, fileName);
  const workbook = await loadWorkbook(inputPath);
  const beforeCells = workbookCellDigests(workbook);
  const beforeCharts = plotChartSnapshots(workbook);

  for (const sheetName of plotSheets) reorderPlotSheet(workbook.worksheets.getItem(sheetName));

  assertEqual(workbookCellDigests(workbook), beforeCells, `${fileName}: cell values/formulas changed during reordering`);
  assertEqual(plotChartSnapshots(workbook), beforeCharts, `${fileName}: chart properties changed beyond placement`);
  for (const sheetName of plotSheets) {
    assertEqual(await visualOrder(workbook, sheetName), expectedVisualOrder(), `${fileName}: ${sheetName} visual order mismatch`);
  }
  const errors = await workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?",
    options: { useRegex: true, maxResults: 300 },
    maxChars: 10000,
  });
  if (!errors.ndjson.includes("matched 0 entries")) throw new Error(`${fileName}: formula error scan failed`);

  await renderAllSheets(workbook, fileName);
  await fs.mkdir(outputDir, { recursive: true });
  const outputPath = path.join(outputDir, fileName);
  const output = await SpreadsheetFile.exportXlsx(workbook);
  await output.save(outputPath);

  const exported = await loadWorkbook(outputPath);
  assertEqual(workbookCellDigests(exported), beforeCells, `${fileName}: exported cells/formulas changed`);
  assertEqual(plotChartSnapshots(exported), beforeCharts, `${fileName}: exported chart properties changed beyond placement`);
  const orders = {};
  for (const sheetName of plotSheets) {
    orders[sheetName] = await visualOrder(exported, sheetName);
    assertEqual(orders[sheetName], expectedVisualOrder(), `${fileName}: exported ${sheetName} visual order mismatch`);
  }
  const verification = {
    fileName,
    outputPath,
    sheetsRendered: exported.worksheets.items.length,
    formulaErrors: 0,
    cellsAndFormulasUnchanged: true,
    chartPropertiesUnchangedBeyondPlacement: true,
    order: desiredGroups.map(([metric, model]) => `${metric} — ${model}`),
    plotSheets: Object.fromEntries(Object.entries(orders).map(([name, entries]) => [name, entries.length])),
  };
  await fs.writeFile(path.join(workDir, `${fileName}.verification.json`), JSON.stringify(verification, null, 2), "utf8");
  console.log(JSON.stringify(verification));
}

const mode = process.argv[2] ?? "inspect";
if (mode === "inspect") {
  for (const fileName of files) await inspectAndRender(fileName);
  process.exit(0);
}

if (mode === "edit") {
  try {
    for (const fileName of files) await editWorkbook(fileName);
  } catch (error) {
    console.error(JSON.stringify({ name: error?.name, message: error?.message, stack: String(error?.stack ?? "").split("\n").slice(0, 8) }, null, 2));
    process.exit(1);
  }
  process.exit(0);
}

if (mode === "focus") {
  const targets = [
    { label: "before", filePath: path.join(inputDir, files[0]) },
    { label: "after", filePath: path.join(outputDir, files[0]) },
    { label: "after_nonrerun", filePath: path.join(outputDir, files[1]) },
  ];
  for (const target of targets) {
    const workbook = await loadWorkbook(target.filePath);
    const preview = await workbook.render({ sheetName: "Plots - Empirical", range: "A20:V82", scale: 2, format: "png" });
    await fs.writeFile(path.join(workDir, `${target.label}_empirical_focus.png`), new Uint8Array(await preview.arrayBuffer()));
  }
  process.exit(0);
}

throw new Error(`Unsupported mode: ${mode}`);
