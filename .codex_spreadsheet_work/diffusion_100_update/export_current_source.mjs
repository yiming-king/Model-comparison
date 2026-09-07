import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const projectRoot = "/Users/yimingzang/Documents/Project/benchmark2";
const workDir = path.join(projectRoot, ".codex_spreadsheet_work/diffusion_100_update");
const fileNames = [
  "FNR-FPR tables without MMD rerun1.xlsx",
  "FNR-FPR tables without MMD.xlsx",
];

for (const fileName of fileNames) {
  const filePath = path.join(projectRoot, "FNR-FPR", fileName);
  const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(filePath));
  const source = workbook.worksheets.getItem("Source rates").getRange("A1:R529");
  const method = workbook.worksheets.getItem("Method").getRange("A1:B20");
  const chartTitles = {};
  for (const sheetName of ["Plots - Simulated", "Plots - Empirical", "Plots - Gaussian"]) {
    chartTitles[sheetName] = workbook.worksheets.getItem(sheetName).charts.items.map((chart) => ({
      title: typeof chart.title === "string" ? chart.title : chart.title?.text ?? null,
      seriesCount: chart.series.items.length,
    }));
  }
  const outputPath = path.join(workDir, `${fileName}.current.json`);
  await fs.writeFile(outputPath, JSON.stringify({
    filePath,
    sourceValues: source.values,
    sourceFormulas: source.formulas,
    methodValues: method.values,
    chartTitles,
  }, null, 2));
  console.log(JSON.stringify({ fileName, outputPath, sourceRows: source.values.length, chartTitles }));
}
