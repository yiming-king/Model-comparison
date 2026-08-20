import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const files = process.argv.slice(2);

for (const path of files) {
  const input = await FileBlob.load(path);
  const workbook = await SpreadsheetFile.importXlsx(input);
  const sheets = await workbook.inspect({
    kind: "sheet",
    include: "id,name",
    maxChars: 8000,
  });
  console.log(JSON.stringify({ path, sheets: sheets.ndjson }));

  for (const term of ["S=D", "Posterior MMD", "FNR", "FPR", "dimension", "Dimension", "D="]) {
    const matches = await workbook.inspect({
      kind: "match",
      searchTerm: term,
      options: { useRegex: false, maxResults: 80 },
      maxChars: 12000,
    });
    console.log(JSON.stringify({ path, term, matches: matches.ndjson }));
  }

  const drawings = await workbook.inspect({
    kind: "drawing",
    maxChars: 30000,
  });
  console.log(JSON.stringify({ path, drawings: drawings.ndjson }));
}
