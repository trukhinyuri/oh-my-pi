import { describe, expect, spyOn, test } from "bun:test";
import { getFastembedCacheDir } from "@oh-my-pi/pi-utils";
import type * as Fastembed from "fastembed";
import type * as Ort from "onnxruntime-node";
import { defaultLocalModelInitializer, type LocalModelInitOptions } from "../src/core/embeddings";
import { isFastembedModelCached } from "../src/core/fastembed-model-cache";
import * as runtime from "../src/core/fastembed-runtime";
import { cosineSimilarity } from "../src/core/vector-math";

// Runs the real model, so only where it is already in omp's fastembed cache:
// the test never downloads its 2.2 GB. Windows is left out because
// onnxruntime-node's finalizer crashes Bun at exit there (#3031), which is why
// omp itself only loads it in the embed worker subprocess.
const MODEL = "fast-multilingual-e5-large";
const CACHE_DIR = getFastembedCacheDir();
const MODEL_READY = process.platform !== "win32" && (await isFastembedModelCached(MODEL, CACHE_DIR));
/** fastembed's token limit for every bundled model. */
const MAX_TOKENS = 512;
/** Padding must not turn a vector; the same floor as the session budget's bench. */
const PARITY_COSINE_FLOOR = 0.999999;

const TEXTS = [
	"query: What did we decide about omp memory embeddings and recall performance?",
	"query: Где хранится память Mnemopi и как работает поиск?",
	"passage: 记忆子系统把向量存在本地 SQLite 数据库中。",
	"passage: الذاكرة المحلية تحفظ المتجهات في قاعدة بيانات SQLite.",
	"passage: स्मृति उपप्रणाली स्थानीय डेटाबेस में वेक्टर रखती है।",
	"passage: Die Erinnerung bleibt lokal 🧠 — keine Daten verlassen den Rechner.",
	`passage: ${"A retained transcript about embeddings, recall, and the shared embedding server. ".repeat(4)}`,
];
/** Longer than the model's limit, so it is truncated to {@link MAX_TOKENS}. */
const OVERLONG = `passage: ${"Привет, мир! Hello, world! ".repeat(200)}`;

/** What one forward pass was given: its input shape, and whether its last column holds a real token. */
interface ForwardPass {
	dims: readonly number[];
	lastColumnUsed: boolean;
}

async function collect(output: AsyncIterable<number[][]>): Promise<number[][]> {
	const rows: number[][] = [];
	for await (const batch of output) rows.push(...batch);
	return rows;
}

/** The ONNX session a fastembed `FlagEmbedding` runs on; fastembed keeps it in a TypeScript-private field. */
function sessionOf(model: object): Ort.InferenceSession {
	if (!("session" in model)) throw new Error("fastembed FlagEmbedding has no session field");
	// fastembed's own `InferenceSession.create` result; only its `run` is used.
	const session = model.session as Ort.InferenceSession;
	return session;
}

function recordForwardPasses(session: Ort.InferenceSession): ForwardPass[] {
	const passes: ForwardPass[] = [];
	const run = session.run.bind(session);
	spyOn(session, "run").mockImplementation(((feeds: Record<string, Ort.Tensor>, ...rest: unknown[]) => {
		const mask = feeds.attention_mask;
		const [rows = 0, columns = 0] = mask?.dims ?? [];
		const data = mask?.data as BigInt64Array;
		let lastColumnUsed = false;
		for (let row = 0; row < rows; row++) lastColumnUsed ||= data[row * columns + columns - 1] === 1n;
		passes.push({ dims: feeds.input_ids?.dims ?? [], lastColumnUsed });
		return (run as (...args: unknown[]) => Promise<Ort.InferenceSession.ReturnType>)(feeds, ...rest);
	}) as never);
	return passes;
}

function expectSameDirections(actual: number[][], expected: number[][]): void {
	expect(actual).toHaveLength(expected.length);
	for (const [index, vector] of actual.entries()) {
		expect(cosineSimilarity(vector, expected[index] ?? [])).toBeGreaterThanOrEqual(PARITY_COSINE_FLOOR);
	}
}

describe.skipIf(!MODEL_READY)("local fastembed embeddings", () => {
	test("pad each batch only to its longest text, capped at the model limit, without turning a vector", async () => {
		const fastembed = await runtime.loadFastembed();
		// fastembed's own `embed` on the same model and session is the reference.
		let reference: number[][] = [];
		let overlongReference: number[][] = [];
		let passes: ForwardPass[] = [];
		const loadSpy = spyOn(runtime, "loadFastembed").mockResolvedValue({
			...fastembed,
			FlagEmbedding: {
				init: async (options: LocalModelInitOptions) => {
					const model = await fastembed.FlagEmbedding.init(options);
					reference = await collect(model.embed([...TEXTS]));
					overlongReference = await collect(model.embed([OVERLONG]));
					passes = recordForwardPasses(sessionOf(model));
					return model;
				},
			},
		} as never);
		try {
			const model = await defaultLocalModelInitializer({
				model: MODEL as Fastembed.EmbeddingModel.MLE5Large,
				cacheDir: CACHE_DIR,
				showDownloadProgress: false,
			});

			const vectors = await collect(model.embed([...TEXTS]));
			const overlong = await collect(model.embed([OVERLONG]));

			const [batch, truncated] = passes;
			expect(batch?.dims[0]).toBe(TEXTS.length);
			expect(batch?.dims[1]).toBeLessThan(MAX_TOKENS);
			expect(batch?.lastColumnUsed).toBe(true);
			expect(truncated?.dims).toEqual([1, MAX_TOKENS]);
			expectSameDirections(vectors, reference);
			expectSameDirections(overlong, overlongReference);
		} finally {
			loadSpy.mockRestore();
		}
	}, 300_000);
});
