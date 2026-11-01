/** Cached-model benchmark: no model downloads; compare interleaved padding policies. */
import { getFastembedCacheDir } from "@oh-my-pi/pi-utils";
import type { EmbeddingModel } from "fastembed";
import { fastembedModelDir, isFastembedModelCached } from "../src/core/fastembed-model-cache";
import { loadFastembed, withBatchLongestPadding, withEmbedSessionBudget } from "../src/core/fastembed-runtime";
import { cosineSimilarity } from "../src/core/vector-math";

const modelName = "fast-multilingual-e5-large";
const cacheDir = getFastembedCacheDir();
if (!(await isFastembedModelCached(modelName, cacheDir)))
	throw new Error("Cache multilingual-e5-large before this benchmark");
const { FlagEmbedding } = await loadFastembed();
const model = await withEmbedSessionBudget(cacheDir, () =>
	FlagEmbedding.init({ model: modelName as EmbeddingModel, cacheDir, showDownloadProgress: false }),
);
const referenceEmbed = model.embed.bind(model);
const modelDir = fastembedModelDir(modelName, cacheDir)!;
const config = await Bun.file(`${modelDir}/config.json`).json();
const tokenizerConfig = await Bun.file(`${modelDir}/tokenizer_config.json`).json();
const tokenizer = (
	model as unknown as {
		tokenizer: { setPadding(options: { maxLength?: number; padId: number; padToken: string }): void };
	}
).tokenizer;
const padding = { padId: config.pad_token_id as number, padToken: tokenizerConfig.pad_token as string };
const optimized = await withBatchLongestPadding(model, modelDir);
const optimizedEmbed = optimized.embed.bind(optimized);
const results = [];
for (const [name, texts] of [
	["short-single", ["query: Where are the release checks documented?"]],
	[
		"short-batch",
		["query: memory", "query: Где лежат заметки?", "passage: Local embeddings stay local.", "query: 查找发布检查"],
	],
	["model-limit", [`passage: ${"long transcript ".repeat(600)}`]],
] as const) {
	const timings: Record<string, number[]> = { fixed: [], dynamic: [] };
	const cpuMs: Record<string, number[]> = { fixed: [], dynamic: [] };
	let reference: number[][] = [];
	let minCosine = 1;
	for (let round = 0; round < 4; round++) {
		for (const policy of round % 2 === 0 ? ["fixed", "dynamic"] : ["dynamic", "fixed"]) {
			tokenizer.setPadding(policy === "fixed" ? { ...padding, maxLength: 512 } : padding);
			const embed = policy === "fixed" ? referenceEmbed : optimizedEmbed;
			const cpu = process.cpuUsage();
			const started = performance.now();
			const rows: number[][] = [];
			for await (const batch of embed([...texts])) rows.push(...batch);
			const elapsed = performance.now() - started;
			const used = process.cpuUsage(cpu);
			if (round > 0) {
				timings[policy]!.push(elapsed);
				cpuMs[policy]!.push((used.user + used.system) / 1000);
			}
			if (policy === "fixed") reference = rows;
			else if (reference.length)
				for (let i = 0; i < rows.length; i++)
					minCosine = Math.min(minCosine, cosineSimilarity(rows[i]!, reference[i]!));
		}
	}
	if (minCosine < 0.999999) throw new Error(`${name}: vector parity failed: ${minCosine}`);
	results.push({ name, batch: texts.length, timings, cpuMs, minCosine });
}
console.log(
	JSON.stringify(
		{ platform: process.platform, arch: process.arch, cudaRequested: process.env.OMP_ORT_CUDA, results },
		null,
		2,
	),
);
