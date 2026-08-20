import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const inputPath = "/Users/yimingzang/Documents/Project/benchmark2/FNR-FPR/FNR-FPR tables.xlsx";
const outputPath = "/private/tmp/fnr_fpr_source_rates.json";

const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(inputPath));
const sheet = workbook.worksheets.getItem("Source rates");
const used = sheet.getUsedRange();
if (!used) throw new Error("Source rates is empty");

await fs.writeFile(
  outputPath,
  JSON.stringify({ address: used.address, values: used.values }, null, 2),
  "utf8",
);
console.log(outputPath);
