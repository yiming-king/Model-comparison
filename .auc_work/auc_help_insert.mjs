import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const input = await FileBlob.load("/Users/yimingzang/Documents/Project/benchmark2/FNR-FPR/FNR-FPR tables.xlsx");
const workbook = await SpreadsheetFile.importXlsx(input);
console.log(workbook.help("*", { search: "insert.*row|row.*insert|range.*insert", include: "index,examples,notes", maxChars: 8000 }).ndjson);
