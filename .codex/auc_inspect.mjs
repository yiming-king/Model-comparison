import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const inputPath = "/Users/yimingzang/Documents/Project/benchmark2/FNR-FPR/FNR-FPR tables.xlsx";
const inspectDir = "/private/tmp/fnr_fpr_auc_inspect";
await fs.mkdir(inspectDir, { recursive: true });

const input = await FileBlob.load(inputPath);
const workbook = await SpreadsheetFile.importXlsx(input);

const summary = await workbook.inspect({
  kind: "workbook,sheet,table",
  maxChars: 12000,
  tableMaxRows: 12,
  tableMaxCols: 18,
  tableMaxCellChars: 120,
});
console.log("SUMMARY");
console.log(summary.ndjson);

const sheets = workbook.worksheets.items;
for (const sheet of sheets) {
  const used = sheet.getUsedRange();
  console.log(`SHEET ${sheet.name} USED ${used?.address ?? "<none>"}`);
  if (used) {
    const region = await workbook.inspect({
      kind: "region",
      sheetId: sheet.name,
      range: used.address,
      maxChars: 14000,
    });
    console.log(`REGION ${sheet.name}`);
    console.log(region.ndjson);
    const formulas = await workbook.inspect({
      kind: "formula",
      sheetId: sheet.name,
      range: used.address,
      maxChars: 10000,
      options: { maxResults: 300 },
    });
    console.log(`FORMULAS ${sheet.name}`);
    console.log(formulas.ndjson);
    const preview = await workbook.render({ sheetName: sheet.name, autoCrop: "all", scale: 1, format: "png" });
    await fs.writeFile(`${inspectDir}/${sheet.name.replace(/[^A-Za-z0-9_.-]+/g, "_")}.png`, new Uint8Array(await preview.arrayBuffer()));
  }
}
