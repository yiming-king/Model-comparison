import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const inputPath = "/Users/yimingzang/Documents/Project/benchmark2/FNR-FPR/FNR-FPR tables.xlsx";
const outputPath = "/Users/yimingzang/Documents/Project/benchmark2/outputs/true_roc_auc/data/original_source_rates.csv";
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(inputPath));
const values = workbook.worksheets.getItem("Source rates").getUsedRange().values;

const escape = (value) => {
  if (value === null || value === undefined) return "";
  const text = String(value);
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
};
await fs.writeFile(outputPath, values.map((row) => row.map(escape).join(",")).join("\n") + "\n", "utf8");
console.log(outputPath);
