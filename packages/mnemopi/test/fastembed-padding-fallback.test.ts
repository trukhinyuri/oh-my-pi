import { describe, expect, test } from "bun:test";
import { TempDir } from "@oh-my-pi/pi-utils";
import { withBatchLongestPadding } from "../src/core/fastembed-runtime";

// No model download or GPU: optimization failures must leave an initialized
// embedding model usable, even if its optional cache metadata disappears.
describe("optional padding optimization", () => {
	for (const metadata of ["missing", "invalid-json", "negative-id", "fractional-id"] as const) {
		test(`keeps embeddings usable with ${metadata} padding metadata`, async () => {
			using dir = TempDir.createSync("padding-fallback-");
			if (metadata !== "missing") {
				await Bun.write(
					dir.join("config.json"),
					metadata === "invalid-json"
						? "{"
						: JSON.stringify({ pad_token_id: metadata === "negative-id" ? -1 : 0.5 }),
				);
				await Bun.write(dir.join("tokenizer_config.json"), JSON.stringify({ pad_token: "[PAD]" }));
			}
			const model = {
				model: "fast-bge-small-en-v1.5",
				pooling: "mean",
				tokenizer: {
					setPadding() {
						throw new Error("Invalid metadata must not change the tokenizer");
					},
					async encodeBatch() {
						return [];
					},
				},
				session: {
					async run() {
						throw new Error("Unused session");
					},
				},
				async *embed(texts: string[]) {
					yield texts.map(text => [text.length, 1]);
				},
			};
			const optimized = await withBatchLongestPadding(model, dir.path());
			const rows: number[][] = [];
			for await (const batch of optimized.embed(["abc", "d"])) rows.push(...batch);
			expect(rows).toEqual([
				[3, 1],
				[1, 1],
			]);
		});
	}
});
