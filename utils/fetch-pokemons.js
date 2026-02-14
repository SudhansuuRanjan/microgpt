import { writeFile } from "fs/promises";

const res = await fetch("https://pokeapi.co/api/v2/pokemon?limit=2000");
const data = await res.json();
const names = data.results.map(p => p.name);
await writeFile("pokemon-names.txt", names.join("\n"), "utf-8");
console.log(`Saved ${names.length} pokemon names to pokemon-names.txt`);
