import * as cheerio from "cheerio";
import { writeFile } from "fs/promises";

async function extractNames() {
  const url =
    "https://www.in.pampers.com/pregnancy/baby-names/article/indian-baby-boys-names";

  const res = await fetch(url);
  const html = await res.text();

  const $ = cheerio.load(html);
  const names = [];

  $("table tr").each((_, row) => {
    const columns = $(row).find("td");

    if (columns.length >= 2) {
      const name = $(columns[1]).text().trim();
      if (name) {
        names.push(name);
      }
    }
  });

  // Join names with newline
  const output = names.join("\n");

  // Write to file
  await writeFile("baby_names.txt", output, "utf-8");

  console.log(`Saved ${names.length} names to baby_names.txt`);
}

extractNames().catch(console.error);
