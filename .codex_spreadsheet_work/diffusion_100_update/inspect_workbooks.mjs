import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const projectRoot = "/Users/yimingzang/Documents/Project/benchmark2";
const workDir = path.join(projectRoot, ".codex_spreadsheet_work/diffusion_100_update");
const files = [
  path.join(projectRoot, "FNR-FPR/FNR-FPR tables without MMD rerun1.xlsx"),
  path.join(projectRoot, "FNR-FPR/FNR-FPR tables without MMD.xlsx"),
];

await fs.mkdir(path.join(workDir, "previews_before"), { recursive: true });

for (const filePath of files) {
  const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(filePath));
  const stem = path.basename(filePath, ".xlsx").replaceAll(" ", "_");
  const sheetNames = workbook.worksheets.items.map((sheet) => sheet.name);
  console.log(JSON.stringify({ filePath, sheetNames }));
  console.log((await workbook.inspect({
    kind: "workbook,sheet,drawing",
    maxChars: 12000,
    tableMaxRows: 4,
    tableMaxCols: 12,
  })).ndjson);

  for (const sheetName of sheetNames) {
    const sheet = workbook.worksheets.getItem(sheetName);
    const used = sheet.getUsedRange();
    console.log(JSON.stringify({
      file: path.basename(filePath),
      sheetName,
      usedRange: used?.address ?? null,
      chartCount: sheet.charts.items.length,
    }));

    if (sheetName === "Source rates") {
      for (const range of ["A1:R8", "A30:R38", "A62:R70", "A94:R102", "A126:R134", "A158:R166", "A190:R198"]) {
        console.log((await workbook.inspect({
          kind: "table",
          sheetId: sheetName,
          range,
          include: "values,formulas",
          tableMaxRows: 10,
          tableMaxCols: 18,
          maxChars: 8000,
        })).ndjson);
      }
    }

    const preview = await workbook.render({
      sheetName,
      autoCrop: "all",
      scale: sheetName === "Source rates" ? 0.4 : 0.7,
      format: "png",
    });
    await fs.writeFile(
      path.join(workDir, "previews_before", `${stem}__${sheetName.replaceAll(" ", "_")}.png`),
      new Uint8Array(await preview.arrayBuffer()),
    );
  }
}
