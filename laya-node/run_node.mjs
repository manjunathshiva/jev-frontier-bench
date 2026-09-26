// Laya from Node.js (@receptron/laya, ONNX Runtime, CPU) on the same 400 items the
// Python arms answered, to check the port's "matches Python to four decimal places"
// and its speed. Writes results/node_calls.jsonl. Needs results/items.json and
// results/ext_items.json (python bench.py sample; python ext_bench.py sample).
//
//   cd laya-node && npm install && node run_node.mjs
import { Laya } from "@receptron/laya";
import { readFileSync, appendFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const OUT = fileURLToPath(new URL("../results", import.meta.url));
const items = [
  ...JSON.parse(readFileSync(`${OUT}/items.json`, "utf8")).items,
  // ext items: same option mapping as ext_bench.py (empty descriptions -> the label itself)
  ...JSON.parse(readFileSync(`${OUT}/ext_items.json`, "utf8")).items.map((it) => ({
    ...it,
    options: Object.fromEntries(Object.entries(it.options).map(([k, v]) => [k, v ?? k])),
  })),
];

function question(it) {
  if (it.type === "noul") return { type: "noul", instructions: it.instructions };
  if (it.type === "choice") return { type: "choice", instructions: it.instructions, criteria: it.options };
  return { type: "score", instructions: it.instructions, criteria: Object.values(it.options) };
}

const log = `${OUT}/node_calls.jsonl`;
writeFileSync(log, "");
let t0 = performance.now();
const laya = await Laya.load();
await laya.systemOne({ message: "warm up" }, { q: { type: "noul", instructions: "Is this a test?" } });
console.log(`loaded + warm-up in ${((performance.now() - t0) / 1000).toFixed(1)} s`);

for (const it of items) {
  const rec = { item: it.id, task: it.task, arm: "laya-node" };
  try {
    t0 = performance.now();
    const res = await laya.systemOne(it.state, { q: question(it) });
    rec.wall_s = (performance.now() - t0) / 1000;
    const a = res.answers.q;
    if (a.type === "noul") rec.probs = { yes: a.noul, no: 1 - a.noul };
    else if (a.type === "choice") rec.probs = a.probabilities;
    else {
      const keys = Object.keys(it.options);
      rec.probs = Object.fromEntries(Object.entries(a.probabilities).map(([k, v]) => [keys[Number(k)], v]));
    }
    rec.answer = Object.entries(rec.probs).sort((x, y) => y[1] - x[1])[0][0];
  } catch (e) {
    rec.error = String(e).slice(0, 300);
  }
  appendFileSync(log, JSON.stringify(rec) + "\n");
}
await laya.close();
console.log(`done: ${items.length} items -> ${log}`);
