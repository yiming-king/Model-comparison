import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const inputPath = "/Users/yimingzang/Documents/Project/benchmark2/FNR-FPR/FNR-FPR tables.xlsx";
const input = await FileBlob.load(inputPath);
const workbook = await SpreadsheetFile.importXlsx(input);

const source = workbook.worksheets.getItem("Source rates");
const sourceValues = source.getRange("A1:I321").values;
for (let i = 0; i < sourceValues.length; i++) {
  const row = sourceValues[i];
  if (i === 0 || i < 40 || i >= 180) console.log(`${i + 1}\t${row.map(v => v ?? "").join("\t")}`);
}

for (const sheetName of ["Simulated data", "Empirical data"]) {
  const sheet = workbook.worksheets.getItem(sheetName);
  console.log(`STYLE ${sheetName}`);
  const style = await workbook.inspect({
    kind: "computedStyle",
    sheetId: sheetName,
    range: "A4:J18",
    maxChars: 8000,
  });
  console.log(style.ndjson);
  console.log(`VALUES ${sheetName}`);
  console.log(JSON.stringify(sheet.getRange("A4:J18").values));
  console.log(`FORMULAS ${sheetName}`);
  console.log(JSON.stringify(sheet.getRange("A4:J18").formulas));
}
