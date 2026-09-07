import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const root = "/Users/yimingzang/Documents/Project/benchmark2";
const inputPaths = [
  path.join(root, "FNR-FPR", "FNR-FPR tables without MMD rerun1.xlsx"),
  path.join(root, "FNR-FPR", "FNR-FPR tables without MMD.xlsx"),
];
const workDir = path.join(root, ".codex_spreadsheet_work", "diffusion_m0_m2");

async function loadWorkbook(filePath) {
  const input = await FileBlob.load(filePath);
  return SpreadsheetFile.importXlsx(input);
}

async function inspectWorkbook(filePath) {
  const workbook = await loadWorkbook(filePath);
  const overview = await workbook.inspect({
    kind: "workbook,sheet,drawing",
    include: "id,name,address,type,title,position,series",
    maxChars: 50000,
  });
  const matches = await workbook.inspect({
    kind: "match",
    searchTerm: "diffusion|m0|m1|m2|m3",
    options: { useRegex: true, maxResults: 500 },
    maxChars: 30000,
    summary: "locate diffusion model blocks",
  });
  const label = path.basename(filePath);
  await fs.writeFile(path.join(workDir, `${label}.overview.ndjson`), overview.ndjson, "utf8");
  await fs.writeFile(path.join(workDir, `${label}.matches.ndjson`), matches.ndjson, "utf8");
  const sheetRows = overview.ndjson
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line))
    .filter((row) => row.kind === "sheet");
  const previewDir = path.join(workDir, "before", label.replace(/\.xlsx$/i, ""));
  await fs.mkdir(previewDir, { recursive: true });
  for (const row of sheetRows) {
    const preview = await workbook.render({
      sheetName: row.name,
      autoCrop: "all",
      scale: 0.8,
      format: "png",
    });
    const safeName = row.name.replace(/[^a-z0-9._-]+/gi, "_");
    await fs.writeFile(
      path.join(previewDir, `${String(row.index).padStart(2, "0")}_${safeName}.png`),
      new Uint8Array(await preview.arrayBuffer()),
    );
  }
  console.log(JSON.stringify({ file: label, overview: overview.ndjson, sheetCount: sheetRows.length }));
}

async function dumpWorkbook(filePath) {
  const workbook = await loadWorkbook(filePath);
  const label = path.basename(filePath);
  const ranges = {
    "Source rates": "A1:R337",
    "Simulated data": "A1:F123",
    "Empirical data": "A1:F123",
    "Method": "A1:B20",
    "Chart data": "A1:O147",
    "Diagnostic chart data": "A1:AB55",
    "Plots - Simulated": "A1:U119",
    "Plots - Empirical": "A1:U119",
    "Diagnostic - Simulated": "A1:X113",
    "Diagnostic - Empirical": "A1:X113",
  };
  const dump = { file: label, sheets: {} };
  for (const [sheetName, address] of Object.entries(ranges)) {
    const sheet = workbook.worksheets.getItem(sheetName);
    const range = sheet.getRange(address);
    dump.sheets[sheetName] = {
      address,
      values: range.values,
      formulas: range.formulas,
      displayFormulas: range.displayFormulas,
    };
  }
  for (const sheetName of ["Plots - Simulated", "Plots - Empirical", "Diagnostic - Simulated", "Diagnostic - Empirical"]) {
    const sheet = workbook.worksheets.getItem(sheetName);
    dump.sheets[sheetName].charts = sheet.charts.items.map((chart, chartIndex) => ({
        chartIndex,
        type: chart.type,
        title: chart.title,
        titleText: chart.title?.text,
        titleTextStyle: chart.titleTextStyle,
        hasLegend: chart.hasLegend,
        legend: chart.legend,
        xAxis: chart.xAxis,
        yAxis: chart.yAxis,
        categories: chart.categories,
        series: chart.series.items.map((series, seriesIndex) => ({
          seriesIndex,
          name: series.name,
          formula: series.formula,
          categoryFormula: series.categoryFormula,
          fill: series.fill,
          line: series.line,
          marker: series.marker,
        })),
      }));
  }
  const styleChecks = {};
  for (const sheetName of ["Simulated data", "Empirical data", "Chart data", "Diagnostic chart data"]) {
    const result = await workbook.inspect({
      kind: "computedStyle",
      sheetId: sheetName,
      range: ranges[sheetName],
      maxChars: 50000,
    });
    styleChecks[sheetName] = result.ndjson;
  }
  dump.computedStyles = styleChecks;
  await fs.writeFile(path.join(workDir, `${label}.dump.json`), JSON.stringify(dump, null, 2), "utf8");
  console.log(JSON.stringify({ file: label, chartCounts: {
    plotsSimulated: dump.sheets["Plots - Simulated"].charts.length,
    plotsEmpirical: dump.sheets["Plots - Empirical"].charts.length,
    diagnosticSimulated: dump.sheets["Diagnostic - Simulated"].charts.length,
    diagnosticEmpirical: dump.sheets["Diagnostic - Empirical"].charts.length,
  }}));
}

const SUMMARY_ORDER = ["S=D", "S=2D", "S=4D", "S=6D"];
const CHART_SUMMARIES = ["S=D", "S=2D", "S=4D"];
const DIAGNOSTICS = ["l2", "linf", "density", "mmd"];
const DIAGNOSTIC_LABELS = ["L2", "L∞", "Density", "MMD"];
const DIAGNOSTIC_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"];
const MODEL_COLORS = {
  m0: "#009E73",
  m1: "#0072B2",
  m2: "#CC79A7",
  m3: "#D55E00",
};
const MODEL_LABELS = {
  m0: "M0 (4 parameters)",
  m1: "M1 (5 parameters)",
  m2: "M2 (5 parameters)",
  m3: "M3 (7 parameters)",
};
const STAT_SOURCE_COLUMNS = {
  FNR: "O",
  FPR: "P",
  F1: "Q",
  AUC: "R",
};
const STAT_SOURCE_INDEXES = {
  FNR: 14,
  FPR: 15,
  F1: 16,
  AUC: 17,
};
const METRIC_SPECS = {
  log10_ml_error: { title: "Signed log10 ML error", templateStart: 5 },
  pmp_error: { title: "Signed PMP error", templateStart: 45 },
  posterior_mmd: { title: "Posterior MMD", templateStart: 85 },
};
const NEW_DATA_BLOCKS = [
  { metric: "log10_ml_error", model: "m0", start: 125 },
  { metric: "log10_ml_error", model: "m2", start: 145 },
  { metric: "pmp_error", model: "m0", start: 165 },
  { metric: "pmp_error", model: "m2", start: 185 },
  { metric: "posterior_mmd", model: "m0", start: 205 },
  { metric: "posterior_mmd", model: "m2", start: 225 },
];

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

function rowKey(row) {
  return [row[1], row[2], row[3], row[4], row[5]].join("|");
}

function sourceRateFormula({ sourceRow, statistic }) {
  const valueColumn = STAT_SOURCE_COLUMNS[statistic];
  return `=IFERROR('Source rates'!$${valueColumn}$${sourceRow},NA())`;
}

function prepareBlockMerges(sheet, startRow) {
  sheet.getRange(`A${startRow}:F${startRow}`).merge();
  sheet.getRange(`A${startRow + 1}:F${startRow + 1}`).merge();
  for (let summaryIndex = 0; summaryIndex < SUMMARY_ORDER.length; summaryIndex += 1) {
    const groupStart = startRow + 3 + summaryIndex * 4;
    sheet.getRange(`A${groupStart}:A${groupStart + 3}`).merge();
    sheet.getRange(`A${groupStart}`).values = [[SUMMARY_ORDER[summaryIndex]]];
  }
}

function styleSummaryBlock(sheet, startRow) {
  const border = { preset: "all", style: "thin", color: "#CFD8E3" };
  sheet.getRange(`A${startRow}:F${startRow}`).format = {
    fill: "#D9E7F5",
    font: { bold: true, fontSize: 11, color: "#203864" },
    borders: {
      top: { style: "medium", color: "#4472C4" },
      bottom: { style: "thin", color: "#CFD8E3" },
    },
    horizontalAlignment: "center",
    rowHeight: 22,
  };
  sheet.getRange(`A${startRow + 1}:F${startRow + 1}`).format = {
    fill: "#F4F7FA",
    font: { italic: true, fontSize: 9, color: "#5D6B78" },
    horizontalAlignment: "center",
    rowHeight: 22,
  };
  sheet.getRange(`A${startRow + 2}:F${startRow + 2}`).format = {
    fill: "#E7E9ED",
    font: { bold: true, fontSize: 10, color: "#263442" },
    borders: border,
    horizontalAlignment: "center",
    rowHeight: 20,
  };
  for (let summaryIndex = 0; summaryIndex < SUMMARY_ORDER.length; summaryIndex += 1) {
    const groupStart = startRow + 3 + summaryIndex * 4;
    const fill = summaryIndex % 2 === 0 ? "#FFFFFF" : "#F8FAFC";
    sheet.getRange(`A${groupStart}:F${groupStart + 3}`).format = {
      fill,
      font: { bold: false, fontSize: 10, color: "#263442" },
      borders: border,
      horizontalAlignment: "center",
      rowHeight: 20,
    };
    sheet.getRange(`C${groupStart}:F${groupStart + 3}`).format.numberFormat = "0.0%";
  }
}

function appendSummaryBlocks(sheet, sourceGroup, newRows, endRow) {
  const lookup = new Map(newRows.map((row, index) => [rowKey(row), { row, sourceRow: 338 + index }]));
  for (const block of NEW_DATA_BLOCKS) {
    const metricSpec = METRIC_SPECS[block.metric];
    const template = sheet.getRange(`A${metricSpec.templateStart}:F${metricSpec.templateStart + 18}`);
    const destination = sheet.getRange(`A${block.start}:F${block.start + 18}`);
    prepareBlockMerges(sheet, block.start);
    destination.copyFrom(template, "all");
    styleSummaryBlock(sheet, block.start);
    sheet.getRange(`A${block.start}`).values = [[`${metricSpec.title} — Assumed ${block.model.toUpperCase()}`]];

    const formulaMatrix = [];
    for (const summary of SUMMARY_ORDER) {
      for (const statistic of ["FNR", "FPR", "F1", "AUC"]) {
        formulaMatrix.push(DIAGNOSTICS.map((diagnostic) => {
          const entry = lookup.get([sourceGroup, diagnostic, block.model, summary, block.metric].join("|"));
          if (!entry) throw new Error(`Missing source rate for ${sourceGroup}, ${diagnostic}, ${block.model}, ${summary}, ${block.metric}`);
          return sourceRateFormula({ sourceRow: entry.sourceRow, statistic });
        }));
      }
    }
    const dataRange = sheet.getRange(`C${block.start + 3}:F${block.start + 18}`);
    dataRange.formulas = formulaMatrix;
    dataRange.format.font.bold = false;

    for (let summaryIndex = 0; summaryIndex < SUMMARY_ORDER.length; summaryIndex += 1) {
      const summary = SUMMARY_ORDER[summaryIndex];
      for (let statisticIndex = 0; statisticIndex < 4; statisticIndex += 1) {
        const statistic = ["FNR", "FPR", "F1", "AUC"][statisticIndex];
        const values = DIAGNOSTICS.map((diagnostic) => {
          const entry = lookup.get([sourceGroup, diagnostic, block.model, summary, block.metric].join("|"));
          if (!entry) throw new Error(`Missing source rate for ${sourceGroup}, ${diagnostic}, ${block.model}, ${summary}, ${block.metric}`);
          return entry.row[STAT_SOURCE_INDEXES[statistic]];
        });
        const numericValues = values.filter((value) => typeof value === "number" && Number.isFinite(value));
        if (!numericValues.length) continue;
        const winner = statistic === "FNR" || statistic === "FPR"
          ? Math.min(...numericValues)
          : Math.max(...numericValues);
        const sheetRow = block.start + 3 + summaryIndex * 4 + statisticIndex;
        values.forEach((value, diagnosticIndex) => {
          if (typeof value === "number" && Number.isFinite(value) && Math.abs(value - winner) <= 1e-12) {
            const sheetColumn = columnLetter(3 + diagnosticIndex);
            sheet.getRange(`${sheetColumn}${sheetRow}`).format.font.bold = true;
            sheet.getRange(`${sheetColumn}${sheetRow}`).format.font.color = "#203864";
          }
        });
      }
    }
  }
}

function chartDataBlockFormulas(dataSheetName, dataBlockStart) {
  const fnrFpr = [];
  const auc = [];
  for (let summaryIndex = 0; summaryIndex < CHART_SUMMARIES.length; summaryIndex += 1) {
    const summaryBase = dataBlockStart + 3 + summaryIndex * 4;
    const fnrRow = summaryBase;
    const fprRow = summaryBase + 1;
    const aucRow = summaryBase + 3;
    const fnrFprRow = [];
    const aucRowFormulas = [];
    for (let diagnosticIndex = 0; diagnosticIndex < DIAGNOSTICS.length; diagnosticIndex += 1) {
      const dataColumn = columnLetter(3 + diagnosticIndex);
      fnrFprRow.push(`=IFERROR('${dataSheetName}'!${dataColumn}${fnrRow}*100,NA())`);
      fnrFprRow.push(`=IFERROR('${dataSheetName}'!${dataColumn}${fprRow}*100,NA())`);
      aucRowFormulas.push(`=IFERROR('${dataSheetName}'!${dataColumn}${aucRow}*100,NA())`);
    }
    fnrFpr.push(fnrFprRow);
    auc.push(aucRowFormulas);
  }
  return { fnrFpr, auc };
}

function appendChartDataBlocks(workbook) {
  const chartData = workbook.worksheets.getItem("Chart data");
  const specs = [];
  let start = 149;
  for (const source of [
    { label: "Simulated data", sheet: "Simulated data", sourceGroup: "simulated" },
    { label: "Empirical data", sheet: "Empirical data", sourceGroup: "empirical" },
  ]) {
    for (const block of NEW_DATA_BLOCKS) {
      if (source.sourceGroup === "empirical" && start === 185) start = 186;
      const metricSpec = METRIC_SPECS[block.metric];
      const templateStart = source.sourceGroup === "simulated"
        ? ({ log10_ml_error: 3, pmp_error: 15, posterior_mmd: 27 })[block.metric]
        : ({ log10_ml_error: 40, pmp_error: 52, posterior_mmd: 64 })[block.metric];
      chartData.getRange(`A${start}:O${start + 5}`).copyFrom(
        chartData.getRange(`A${templateStart}:O${templateStart + 5}`),
        "all",
      );
      const title = `${source.label} — ${metricSpec.title} — Assumed ${block.model.toUpperCase()}`;
      chartData.getRange(`A${start}:O${start}`).values = [Array(15).fill(title)];
      chartData.getRange(`A${start + 2}:A${start + 4}`).values = CHART_SUMMARIES.map((value) => [value]);
      chartData.getRange(`K${start + 2}:K${start + 4}`).values = CHART_SUMMARIES.map((value) => [value]);
      const formulas = chartDataBlockFormulas(source.sheet, block.start);
      chartData.getRange(`B${start + 2}:I${start + 4}`).formulas = formulas.fnrFpr;
      chartData.getRange(`L${start + 2}:O${start + 4}`).formulas = formulas.auc;
      specs.push({
        sourceGroup: source.sourceGroup,
        sheetName: source.sheet,
        metric: block.metric,
        metricTitle: metricSpec.title,
        model: block.model,
        helperStart: start,
      });
      start += 6;
    }
  }
  return specs;
}

function styleLineSeries(series, color, style) {
  series.line.fill = color;
  series.line.style = style;
  series.line.width = style === "solid" ? 3.75 : 2.5;
  series.marker.symbol = "circle";
  series.marker.size = 8;
  series.marker.fill = color;
}

function styleTrendLegendBlock(sheet, excelBlockStart) {
  const legendRow = excelBlockStart + 15;
  const lineRow = excelBlockStart + 16;
  sheet.getRange(`A${legendRow}:U${lineRow}`).unmerge();
  sheet.getRange(`A${legendRow}:U${lineRow}`).clear({ applyTo: "contents" });
  const legendItems = [
    { range: `A${legendRow}:B${legendRow}`, text: "●  L2", color: DIAGNOSTIC_COLORS[0] },
    { range: `C${legendRow}:D${legendRow}`, text: "●  L∞", color: DIAGNOSTIC_COLORS[1] },
    { range: `E${legendRow}:G${legendRow}`, text: "●  Density", color: DIAGNOSTIC_COLORS[2] },
    { range: `H${legendRow}:J${legendRow}`, text: "●  MMD", color: DIAGNOSTIC_COLORS[3] },
    { range: `L${legendRow}:M${legendRow}`, text: "●  L2", color: DIAGNOSTIC_COLORS[0] },
    { range: `N${legendRow}:O${legendRow}`, text: "●  L∞", color: DIAGNOSTIC_COLORS[1] },
    { range: `P${legendRow}:R${legendRow}`, text: "●  Density", color: DIAGNOSTIC_COLORS[2] },
    { range: `S${legendRow}:U${legendRow}`, text: "●  MMD", color: DIAGNOSTIC_COLORS[3] },
  ];
  for (const item of legendItems) {
    const range = sheet.getRange(item.range);
    range.merge();
    range.values = [[item.text]];
    range.format = {
      font: { bold: true, fontSize: 11, color: item.color },
      horizontalAlignment: "center",
      rowHeight: 18,
    };
  }
  const lineItems = [
    { range: `C${lineRow}:E${lineRow}`, text: "━━━━  FNR (solid)", bold: true },
    { range: `F${lineRow}:H${lineRow}`, text: "┄┄┄┄  FPR (dashed)", bold: false },
  ];
  for (const item of lineItems) {
    const range = sheet.getRange(item.range);
    range.merge();
    range.values = [[item.text]];
    range.format = {
      font: { bold: item.bold, fontSize: 11, color: "#334155" },
      horizontalAlignment: "center",
      rowHeight: 18,
    };
  }
}

function createTrendChart(sheet, { title, helperStart, anchorRow, anchorCol, aucOnly }) {
  const chart = sheet.charts.add("line", {
    title,
    titleTextStyle: { fontSize: 12 },
    hasLegend: false,
    from: { row: anchorRow, col: anchorCol },
    extent: { widthPx: 660, heightPx: 280 },
    xAxis: { axisType: "textAxis", textStyle: { fontSize: 9 } },
    yAxis: { min: 0, max: 100, majorUnit: 20, numberFormatCode: "0", textStyle: { fontSize: 9 } },
  });
  chart.hasLegend = false;
  const dataStart = helperStart + 2;
  const dataEnd = helperStart + 4;
  if (aucOnly) {
    for (let diagnosticIndex = 0; diagnosticIndex < DIAGNOSTICS.length; diagnosticIndex += 1) {
      const valueColumn = columnLetter(12 + diagnosticIndex);
      const series = chart.series.add(DIAGNOSTIC_LABELS[diagnosticIndex]);
      series.categoryFormula = `'Chart data'!$K$${dataStart}:$K$${dataEnd}`;
      series.formula = `'Chart data'!$${valueColumn}$${dataStart}:$${valueColumn}$${dataEnd}`;
      styleLineSeries(series, DIAGNOSTIC_COLORS[diagnosticIndex], "solid");
    }
  } else {
    for (let diagnosticIndex = 0; diagnosticIndex < DIAGNOSTICS.length; diagnosticIndex += 1) {
      const fnrColumn = columnLetter(2 + diagnosticIndex * 2);
      const fprColumn = columnLetter(3 + diagnosticIndex * 2);
      const fnr = chart.series.add(`${DIAGNOSTIC_LABELS[diagnosticIndex]} — FNR`);
      fnr.categoryFormula = `'Chart data'!$A$${dataStart}:$A$${dataEnd}`;
      fnr.formula = `'Chart data'!$${fnrColumn}$${dataStart}:$${fnrColumn}$${dataEnd}`;
      styleLineSeries(fnr, DIAGNOSTIC_COLORS[diagnosticIndex], "solid");
      const fpr = chart.series.add(`${DIAGNOSTIC_LABELS[diagnosticIndex]} — FPR`);
      fpr.categoryFormula = `'Chart data'!$A$${dataStart}:$A$${dataEnd}`;
      fpr.formula = `'Chart data'!$${fprColumn}$${dataStart}:$${fprColumn}$${dataEnd}`;
      styleLineSeries(fpr, DIAGNOSTIC_COLORS[diagnosticIndex], "dashed");
    }
  }
  return chart;
}

function appendTrendCharts(workbook, chartSpecs) {
  for (const sourceGroup of ["simulated", "empirical"]) {
    const plotSheetName = sourceGroup === "simulated" ? "Plots - Simulated" : "Plots - Empirical";
    const sheet = workbook.worksheets.getItem(plotSheetName);
    const sourceSpecs = chartSpecs.filter((spec) => spec.sourceGroup === sourceGroup);
    sourceSpecs.forEach((spec, index) => {
      const excelBlockStart = 123 + index * 20;
      sheet.getRange(`A${excelBlockStart}:U${excelBlockStart + 19}`).copyFrom(
        sheet.getRange("A3:U22"),
        "all",
      );
      styleTrendLegendBlock(sheet, excelBlockStart);
      const anchorRow = 122 + index * 20;
      createTrendChart(sheet, {
        title: `${spec.metricTitle} — Assumed ${spec.model.toUpperCase()} — FNR/FPR`,
        helperStart: spec.helperStart,
        anchorRow,
        anchorCol: 0,
        aucOnly: false,
      });
      createTrendChart(sheet, {
        title: `${spec.metricTitle} — Assumed ${spec.model.toUpperCase()} — AUC`,
        helperStart: spec.helperStart,
        anchorRow,
        anchorCol: 11,
        aucOnly: true,
      });
    });
  }
}

function updateLegendRow(sheet, rowNumber, includeLineStyles) {
  sheet.getRange(`A${rowNumber}:X${rowNumber}`).unmerge();
  if (includeLineStyles) {
    const ranges = ["A:D", "E:H", "I:L", "M:P", "Q:T", "U:X"];
    for (const range of ranges) sheet.getRange(`${range.split(":")[0]}${rowNumber}:${range.split(":")[1]}${rowNumber}`).merge();
    const items = [
      { cell: `A${rowNumber}`, text: `●  ${MODEL_LABELS.m0}`, color: MODEL_COLORS.m0 },
      { cell: `E${rowNumber}`, text: `●  ${MODEL_LABELS.m1}`, color: MODEL_COLORS.m1 },
      { cell: `I${rowNumber}`, text: `●  ${MODEL_LABELS.m2}`, color: MODEL_COLORS.m2 },
      { cell: `M${rowNumber}`, text: `●  ${MODEL_LABELS.m3}`, color: MODEL_COLORS.m3 },
      { cell: `Q${rowNumber}`, text: "━━  FNR (solid)", color: "#203864" },
      { cell: `U${rowNumber}`, text: "┄┄  FPR (dashed)", color: "#44546A" },
    ];
    for (const item of items) {
      const cell = sheet.getRange(item.cell);
      cell.values = [[item.text]];
      cell.format = { font: { bold: true, color: item.color, fontSize: 9 }, horizontalAlignment: "center" };
    }
  } else {
    const ranges = ["A:F", "G:L", "M:R", "S:X"];
    for (const range of ranges) sheet.getRange(`${range.split(":")[0]}${rowNumber}:${range.split(":")[1]}${rowNumber}`).merge();
    const items = [
      { cell: `A${rowNumber}`, model: "m0" },
      { cell: `G${rowNumber}`, model: "m1" },
      { cell: `M${rowNumber}`, model: "m2" },
      { cell: `S${rowNumber}`, model: "m3" },
    ];
    for (const item of items) {
      const cell = sheet.getRange(item.cell);
      cell.values = [[`●  ${MODEL_LABELS[item.model]}`]];
      cell.format = {
        font: { bold: true, color: MODEL_COLORS[item.model], fontSize: 9 },
        horizontalAlignment: "center",
      };
    }
  }
}

function appendDiagnosticDataAndSeries(workbook) {
  const helperSheet = workbook.worksheets.getItem("Diagnostic chart data");
  const sourceStyle = helperSheet.getRange("A1:AB38");
  const destination = helperSheet.getRange("AC1:BD38");
  destination.copyFrom(sourceStyle, "all");
  destination.clear({ applyTo: "contents" });
  helperSheet.getRange("A1").values = [["Formula-backed M0–M3 diagnostic comparison data (%) — S/D = 1, 2, 4"]];
  helperSheet.getRange("AC1").values = [["Additional formula-backed M0/M2 diagnostic comparison data (%)"]];

  const dataBlockStarts = {
    simulated: { log10_ml_error: 3, pmp_error: 9, posterior_mmd: 15 },
    empirical: { log10_ml_error: 21, pmp_error: 27, posterior_mmd: 33 },
  };
  const sourceSheetNames = { simulated: "Simulated data", empirical: "Empirical data" };

  for (const sourceGroup of ["simulated", "empirical"]) {
    for (const metric of Object.keys(METRIC_SPECS)) {
      const helperStart = dataBlockStarts[sourceGroup][metric];
      const m0Block = NEW_DATA_BLOCKS.find((block) => block.metric === metric && block.model === "m0");
      const m2Block = NEW_DATA_BLOCKS.find((block) => block.metric === metric && block.model === "m2");
      const title = `${sourceSheetNames[sourceGroup]} — ${METRIC_SPECS[metric].title} — M0–M3 (4/5/5/7 parameters)`;
      helperSheet.getRange(`A${helperStart}`).values = [[title]];
      helperSheet.getRange(`AC${helperStart}`).values = [[title]];

      for (let diagnosticIndex = 0; diagnosticIndex < DIAGNOSTICS.length; diagnosticIndex += 1) {
        const groupStartColumn = 29 + diagnosticIndex * 7;
        const categoryColumn = columnLetter(groupStartColumn);
        const m0FNRColumn = columnLetter(groupStartColumn + 1);
        const m0FPRColumn = columnLetter(groupStartColumn + 2);
        const m2FNRColumn = columnLetter(groupStartColumn + 3);
        const m2FPRColumn = columnLetter(groupStartColumn + 4);
        const m0AUCColumn = columnLetter(groupStartColumn + 5);
        const m2AUCColumn = columnLetter(groupStartColumn + 6);
        helperSheet.getRange(`${categoryColumn}${helperStart + 1}:${m2AUCColumn}${helperStart + 1}`).values = [[
          "S/D", "M0 FNR", "M0 FPR", "M2 FNR", "M2 FPR", "M0 AUC", "M2 AUC",
        ]];
        helperSheet.getRange(`${categoryColumn}${helperStart + 2}:${categoryColumn}${helperStart + 4}`).values = [["1"], ["2"], ["4"]];
        const dataColumn = columnLetter(3 + diagnosticIndex);
        const formulaRows = [];
        for (let summaryIndex = 0; summaryIndex < CHART_SUMMARIES.length; summaryIndex += 1) {
          const m0SummaryBase = m0Block.start + 3 + summaryIndex * 4;
          const m2SummaryBase = m2Block.start + 3 + summaryIndex * 4;
          formulaRows.push([
            `=IFERROR('${sourceSheetNames[sourceGroup]}'!${dataColumn}${m0SummaryBase}*100,NA())`,
            `=IFERROR('${sourceSheetNames[sourceGroup]}'!${dataColumn}${m0SummaryBase + 1}*100,NA())`,
            `=IFERROR('${sourceSheetNames[sourceGroup]}'!${dataColumn}${m2SummaryBase}*100,NA())`,
            `=IFERROR('${sourceSheetNames[sourceGroup]}'!${dataColumn}${m2SummaryBase + 1}*100,NA())`,
            `=IFERROR('${sourceSheetNames[sourceGroup]}'!${dataColumn}${m0SummaryBase + 3}*100,NA())`,
            `=IFERROR('${sourceSheetNames[sourceGroup]}'!${dataColumn}${m2SummaryBase + 3}*100,NA())`,
          ]);
        }
        helperSheet.getRange(`${m0FNRColumn}${helperStart + 2}:${m2AUCColumn}${helperStart + 4}`).formulas = formulaRows;
      }
    }
  }

  for (const sourceGroup of ["simulated", "empirical"]) {
    const sheetName = sourceGroup === "simulated" ? "Diagnostic - Simulated" : "Diagnostic - Empirical";
    const sheet = workbook.worksheets.getItem(sheetName);
    sheet.getRange("A1").values = [[`${sourceGroup === "simulated" ? "Simulated" : "Empirical"} data — diagnostic comparison: M0–M3 (4, 5, 5, 7 parameters)`]];
    [18, 56, 94].forEach((rowNumber) => updateLegendRow(sheet, rowNumber, true));
    [37, 75, 113].forEach((rowNumber) => updateLegendRow(sheet, rowNumber, false));
    const starts = dataBlockStarts[sourceGroup];
    const charts = sheet.charts.items;
    ["log10_ml_error", "pmp_error", "posterior_mmd"].forEach((metric, metricIndex) => {
      const helperStart = starts[metric];
      const dataStart = helperStart + 2;
      const dataEnd = helperStart + 4;
      for (let diagnosticIndex = 0; diagnosticIndex < DIAGNOSTICS.length; diagnosticIndex += 1) {
        const groupStartColumn = 29 + diagnosticIndex * 7;
        const categoryColumn = columnLetter(groupStartColumn);
        const columns = {
          m0FNR: columnLetter(groupStartColumn + 1),
          m0FPR: columnLetter(groupStartColumn + 2),
          m2FNR: columnLetter(groupStartColumn + 3),
          m2FPR: columnLetter(groupStartColumn + 4),
          m0AUC: columnLetter(groupStartColumn + 5),
          m2AUC: columnLetter(groupStartColumn + 6),
        };
        const categoryFormula = `'Diagnostic chart data'!$${categoryColumn}$${dataStart}:$${categoryColumn}$${dataEnd}`;
        const fnrFprChart = charts[metricIndex * 8 + diagnosticIndex];
        for (const model of ["m0", "m2"]) {
          const fnr = fnrFprChart.series.add(`${MODEL_LABELS[model]} — FNR`);
          fnr.categoryFormula = categoryFormula;
          fnr.formula = `'Diagnostic chart data'!$${columns[`${model}FNR`]}$${dataStart}:$${columns[`${model}FNR`]}$${dataEnd}`;
          styleLineSeries(fnr, MODEL_COLORS[model], "solid");
          const fpr = fnrFprChart.series.add(`${MODEL_LABELS[model]} — FPR`);
          fpr.categoryFormula = categoryFormula;
          fpr.formula = `'Diagnostic chart data'!$${columns[`${model}FPR`]}$${dataStart}:$${columns[`${model}FPR`]}$${dataEnd}`;
          styleLineSeries(fpr, MODEL_COLORS[model], "dashed");
        }
        const aucChart = charts[metricIndex * 8 + 4 + diagnosticIndex];
        for (const model of ["m0", "m2"]) {
          const aucSeries = aucChart.series.add(MODEL_LABELS[model]);
          aucSeries.categoryFormula = categoryFormula;
          aucSeries.formula = `'Diagnostic chart data'!$${columns[`${model}AUC`]}$${dataStart}:$${columns[`${model}AUC`]}$${dataEnd}`;
          styleLineSeries(aucSeries, MODEL_COLORS[model], "solid");
        }
      }
    });
  }
}

async function renderAllSheets(workbook, label, stage) {
  const sheetInfo = await workbook.inspect({ kind: "sheet", include: "id,name", maxChars: 10000 });
  const sheets = sheetInfo.ndjson
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line))
    .filter((row) => row.kind === "sheet");
  const previewDir = path.join(workDir, stage, label.replace(/\.xlsx$/i, ""));
  await fs.mkdir(previewDir, { recursive: true });
  for (const row of sheets) {
    const preview = await workbook.render({ sheetName: row.name, autoCrop: "all", scale: 0.8, format: "png" });
    const safeName = row.name.replace(/[^a-z0-9._-]+/gi, "_");
    await fs.writeFile(
      path.join(previewDir, `${String(row.index).padStart(2, "0")}_${safeName}.png`),
      new Uint8Array(await preview.arrayBuffer()),
    );
  }
}

async function editWorkbook(filePath, payloadPath, outputDir) {
  const workbook = await loadWorkbook(filePath);
  const payload = JSON.parse(await fs.readFile(payloadPath, "utf8"));
  const label = path.basename(filePath);
  console.log(`[${label}] loaded`);
  const newRows = payload.new_rows;
  if (newRows.length !== 192) throw new Error(`Expected 192 new rows for ${label}, found ${newRows.length}`);

  const sourceSheet = workbook.worksheets.getItem("Source rates");
  const existingModels = sourceSheet.getRange("A2:D337").values
    .filter((row) => row[0] === "Diffusion")
    .map((row) => row[3]);
  if (existingModels.includes("m0") || existingModels.includes("m2")) {
    throw new Error(`${label} already contains Diffusion m0/m2 rows`);
  }
  const sourceTable = sourceSheet.tables.items[0];
  sourceTable.rows.add(null, newRows);
  const endRow = 1 + 336 + newRows.length;
  console.log(`[${label}] source rows appended`);

  appendSummaryBlocks(workbook.worksheets.getItem("Simulated data"), "simulated", newRows, endRow);
  appendSummaryBlocks(workbook.worksheets.getItem("Empirical data"), "empirical", newRows, endRow);
  console.log(`[${label}] summary blocks appended`);
  const chartSpecs = appendChartDataBlocks(workbook);
  console.log(`[${label}] chart helper blocks appended`);
  appendTrendCharts(workbook, chartSpecs);
  console.log(`[${label}] trend charts appended`);
  appendDiagnosticDataAndSeries(workbook);
  console.log(`[${label}] diagnostic data and series appended`);

  const keyChecks = {};
  for (const [sheetName, range] of [
    ["Source rates", "A520:R529"],
    ["Simulated data", "A120:F243"],
    ["Empirical data", "A120:F243"],
    ["Chart data", "A145:O221"],
    ["Diagnostic chart data", "AC1:BD38"],
  ]) {
    const inspection = await workbook.inspect({
      kind: "table",
      range: `${sheetName}!${range}`,
      include: "values,formulas",
      tableMaxRows: 12,
      tableMaxCols: 12,
      maxChars: 8000,
    });
    keyChecks[sheetName] = inspection.ndjson;
  }
  const errors = await workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?",
    options: { useRegex: true, maxResults: 300 },
    summary: "final formula error scan excluding intentional N/A",
    maxChars: 10000,
  });
  const diagnosticSeriesCounts = {};
  for (const sheetName of ["Diagnostic - Simulated", "Diagnostic - Empirical"]) {
    const counts = workbook.worksheets.getItem(sheetName).charts.items.map((chart) => chart.series.items.length);
    diagnosticSeriesCounts[sheetName] = counts;
    if (counts.some((count, index) => count !== (index % 8 < 4 ? 8 : 4))) {
      throw new Error(`Unexpected diagnostic series counts in ${sheetName}: ${counts.join(",")}`);
    }
  }
  const trendChartCounts = {};
  for (const sheetName of ["Plots - Simulated", "Plots - Empirical"]) {
    const count = workbook.worksheets.getItem(sheetName).charts.items.length;
    trendChartCounts[sheetName] = count;
    if (count !== 24) throw new Error(`Expected 24 trend charts in ${sheetName}, found ${count}`);
  }

  console.log(`[${label}] structural checks passed; rendering all sheets`);
  await renderAllSheets(workbook, label, "after");
  console.log(`[${label}] all sheets rendered; exporting`);
  await fs.mkdir(outputDir, { recursive: true });
  const output = await SpreadsheetFile.exportXlsx(workbook);
  const outputPath = path.join(outputDir, label);
  await output.save(outputPath);
  await fs.writeFile(
    path.join(workDir, `${label}.verification.json`),
    JSON.stringify({ keyChecks, formulaErrors: errors.ndjson, diagnosticSeriesCounts, trendChartCounts }, null, 2),
    "utf8",
  );
  console.log(JSON.stringify({
    file: label,
    outputPath,
    sourceRows: endRow,
    trendChartCounts,
    diagnosticSeriesCounts: Object.fromEntries(
      Object.entries(diagnosticSeriesCounts).map(([name, counts]) => [name, { charts: counts.length, minSeries: Math.min(...counts), maxSeries: Math.max(...counts) }]),
    ),
    formulaErrors: errors.ndjson,
  }));
}

function valuesMatch(actual, expected) {
  if (typeof expected === "number") {
    return typeof actual === "number" && Number.isFinite(actual) && Math.abs(actual - expected) <= 1e-12;
  }
  if (expected === "N/A") return actual === "N/A" || actual === "#N/A";
  return actual === expected;
}

async function verifyOutputWorkbook(filePath, payloadPath) {
  const workbook = await loadWorkbook(filePath);
  const payload = JSON.parse(await fs.readFile(payloadPath, "utf8"));
  const label = path.basename(filePath);
  const newRows = payload.new_rows;
  const sourceValues = workbook.worksheets.getItem("Source rates").getRange("A338:R529").values;
  if (sourceValues.length !== newRows.length) throw new Error(`${label}: unexpected appended source row count`);
  for (let rowIndex = 0; rowIndex < newRows.length; rowIndex += 1) {
    for (let columnIndex = 0; columnIndex < 18; columnIndex += 1) {
      if (!valuesMatch(sourceValues[rowIndex][columnIndex], newRows[rowIndex][columnIndex])) {
        throw new Error(`${label}: source mismatch at appended row ${rowIndex + 338}, column ${columnIndex + 1}`);
      }
    }
  }
  const modelCounts = sourceValues.reduce((counts, row) => {
    counts[row[3]] = (counts[row[3]] ?? 0) + 1;
    return counts;
  }, {});
  if (modelCounts.m0 !== 96 || modelCounts.m2 !== 96) {
    throw new Error(`${label}: expected 96 m0 and 96 m2 source rows, found ${JSON.stringify(modelCounts)}`);
  }

  const lookup = new Map(newRows.map((row, index) => [rowKey(row), { row, sourceRow: 338 + index }]));
  let summaryCellChecks = 0;
  let boldChecks = 0;
  for (const [sheetName, sourceGroup] of [["Simulated data", "simulated"], ["Empirical data", "empirical"]]) {
    const sheet = workbook.worksheets.getItem(sheetName);
    for (const block of NEW_DATA_BLOCKS) {
      for (let summaryIndex = 0; summaryIndex < SUMMARY_ORDER.length; summaryIndex += 1) {
        const summary = SUMMARY_ORDER[summaryIndex];
        for (let statisticIndex = 0; statisticIndex < 4; statisticIndex += 1) {
          const statistic = ["FNR", "FPR", "F1", "AUC"][statisticIndex];
          const entries = DIAGNOSTICS.map((diagnostic) => lookup.get([
            sourceGroup, diagnostic, block.model, summary, block.metric,
          ].join("|")));
          const expectedValues = entries.map((entry) => entry.row[STAT_SOURCE_INDEXES[statistic]]);
          const numericValues = expectedValues.filter((value) => typeof value === "number" && Number.isFinite(value));
          const winner = numericValues.length
            ? ((statistic === "FNR" || statistic === "FPR") ? Math.min(...numericValues) : Math.max(...numericValues))
            : null;
          const sheetRow = block.start + 3 + summaryIndex * 4 + statisticIndex;
          for (let diagnosticIndex = 0; diagnosticIndex < DIAGNOSTICS.length; diagnosticIndex += 1) {
            const column = columnLetter(3 + diagnosticIndex);
            const cell = sheet.getRange(`${column}${sheetRow}`);
            const actualValue = cell.values[0][0];
            const expectedValue = expectedValues[diagnosticIndex];
            if (!valuesMatch(actualValue, expectedValue)) {
              throw new Error(`${label}: ${sheetName}!${column}${sheetRow} value mismatch`);
            }
            const expectedFormula = sourceRateFormula({ sourceRow: entries[diagnosticIndex].sourceRow, statistic });
            if (cell.formulas[0][0] !== expectedFormula) {
              throw new Error(`${label}: ${sheetName}!${column}${sheetRow} formula mismatch`);
            }
            if (cell.format.numberFormat !== "0.0%") {
              throw new Error(`${label}: ${sheetName}!${column}${sheetRow} number format mismatch`);
            }
            const shouldBeBold = winner !== null && typeof expectedValue === "number" && Math.abs(expectedValue - winner) <= 1e-12;
            if (Boolean(cell.format.font.bold) !== shouldBeBold) {
              throw new Error(`${label}: ${sheetName}!${column}${sheetRow} bold-rule mismatch`);
            }
            summaryCellChecks += 1;
            boldChecks += 1;
          }
        }
      }
    }
  }

  const trendChartCounts = {};
  for (const sheetName of ["Plots - Simulated", "Plots - Empirical"]) {
    const charts = workbook.worksheets.getItem(sheetName).charts.items;
    trendChartCounts[sheetName] = charts.length;
    if (charts.length !== 24) throw new Error(`${label}: ${sheetName} chart count mismatch`);
    for (let index = 12; index < 24; index += 1) {
      const chart = charts[index];
      const expectedSeries = index % 2 === 0 ? 8 : 4;
      if (chart.hasLegend || chart.series.items.length !== expectedSeries) {
        throw new Error(`${label}: ${sheetName} new chart ${index} structure mismatch`);
      }
      for (const series of chart.series.items) {
        if (series.marker.symbol !== "circle" || series.marker.size !== 8) {
          throw new Error(`${label}: ${sheetName} new chart ${index} marker mismatch`);
        }
      }
    }
  }

  const diagnosticSeriesCounts = {};
  for (const sheetName of ["Diagnostic - Simulated", "Diagnostic - Empirical"]) {
    const counts = workbook.worksheets.getItem(sheetName).charts.items.map((chart) => chart.series.items.length);
    diagnosticSeriesCounts[sheetName] = counts;
    if (counts.some((count, index) => count !== (index % 8 < 4 ? 8 : 4))) {
      throw new Error(`${label}: ${sheetName} series counts mismatch`);
    }
  }
  const errors = await workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?",
    options: { useRegex: true, maxResults: 300 },
    maxChars: 10000,
  });
  if (!errors.ndjson.includes("matched 0 entries")) throw new Error(`${label}: formula error scan failed: ${errors.ndjson}`);
  const result = {
    file: label,
    appendedSourceRows: sourceValues.length,
    modelCounts,
    summaryCellChecks,
    boldChecks,
    trendChartCounts,
    diagnosticSeriesCounts: Object.fromEntries(
      Object.entries(diagnosticSeriesCounts).map(([name, counts]) => [name, {
        charts: counts.length,
        minSeries: Math.min(...counts),
        maxSeries: Math.max(...counts),
      }]),
    ),
    formulaErrors: 0,
  };
  await fs.writeFile(path.join(workDir, `${label}.final-verification.json`), JSON.stringify(result, null, 2), "utf8");
  console.log(JSON.stringify(result));
}

const mode = process.argv[2] ?? "inspect";
if (mode === "help") {
  const workbook = await loadWorkbook(inputPaths[0]);
  console.log(workbook.help("*", {
    search: "chart.*series|series.*(line|marker|dash)|markerStyle|lineStyle",
    include: "index,examples,notes",
    maxChars: 12000,
  }).ndjson);
  process.exit(0);
}
if (mode === "edit") {
  const outputDir = path.join(root, "outputs", "20260825_diffusion_m0_m2");
  const payloadPaths = [
    path.join(workDir, "computed_rerun1.json"),
    path.join(workDir, "computed_without_mmd.json"),
  ];
  try {
    for (let index = 0; index < inputPaths.length; index += 1) {
      await editWorkbook(inputPaths[index], payloadPaths[index], outputDir);
    }
  } catch (error) {
    const conciseStack = String(error?.stack ?? "")
      .split("\n")
      .filter((line) => !line.includes("artifact_tool.mjs:2838"))
      .slice(0, 12);
    console.error(JSON.stringify({
      name: error?.name,
      message: error?.message,
      stack: conciseStack,
    }, null, 2));
    process.exit(1);
  }
  process.exit(0);
}
if (mode === "verify") {
  const outputDir = path.join(root, "outputs", "20260825_diffusion_m0_m2");
  const payloadPaths = [
    path.join(workDir, "computed_rerun1.json"),
    path.join(workDir, "computed_without_mmd.json"),
  ];
  try {
    for (let index = 0; index < inputPaths.length; index += 1) {
      await verifyOutputWorkbook(path.join(outputDir, path.basename(inputPaths[index])), payloadPaths[index]);
    }
  } catch (error) {
    console.error(JSON.stringify({ name: error?.name, message: error?.message }, null, 2));
    process.exit(1);
  }
  process.exit(0);
}
for (const filePath of inputPaths) {
  if (mode === "dump") {
    await dumpWorkbook(filePath);
  } else {
    await inspectWorkbook(filePath);
  }
}
