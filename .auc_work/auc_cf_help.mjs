import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";
const input = await FileBlob.load("/Users/yimingzang/Documents/Project/benchmark2/outputs/auc_aligned/FNR-FPR tables_with_AUC_aligned.xlsx");
const workbook = await SpreadsheetFile.importXlsx(input);
console.log(workbook.help("*", { search: "conditionalFormattings|conditionalFormats.*delete|conditionalFormats.*clear", include: "index,examples,notes", maxChars: 8000 }).ndjson);
