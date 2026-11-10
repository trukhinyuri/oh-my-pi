/**
 * A Claude model the bundled catalog does not know yet gets its capabilities
 * from the models.dev catalog. While that catalog is unreachable, discovery
 * rebuilds the row text-only and the cached row has to keep image input alive,
 * which the merge only allows when both rows share a base URL. The registry
 * hands Anthropic discovery a model row's base once its catalog is composed
 * (the scoped refresh that resolves a model id missing from the catalog) and
 * nothing before that (the startup refresh), and discovery used to turn the
 * missing base into `https://api.anthropic.com/v1`, so the two refreshes
 * disagreed and the model became text-only until the catalog was back.
 */
import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import type { FetchImpl } from "@oh-my-pi/pi-ai";
import { ModelRegistry } from "@oh-my-pi/pi-coding-agent/config/model-registry";
import { AuthStorage } from "@oh-my-pi/pi-coding-agent/session/auth-storage";
import { removeSyncWithRetries } from "@oh-my-pi/pi-utils";

const MODEL_ID = "claude-unbundled-vision-1";

function anthropicApi(options: { catalogReachable: boolean }): FetchImpl {
	return async input => {
		const url = String(input instanceof Request ? input.url : input);
		if (url === "https://catalog.stencil.so/models.json.zstd") {
			if (!options.catalogReachable) return new Response("unavailable", { status: 503 });
			return Response.json({
				anthropic: {
					models: {
						[MODEL_ID]: {
							name: "Claude Unbundled Vision",
							tool_call: true,
							modalities: { input: ["text", "image"] },
							limit: { context: 1_000_000, output: 128_000 },
						},
					},
				},
			});
		}
		if (url === "https://api.anthropic.com/v1/models") {
			return Response.json({
				data: [{ type: "model", id: MODEL_ID, display_name: "Claude Unbundled Vision" }],
				has_more: false,
			});
		}
		return new Response("not found", { status: 404 });
	};
}

describe("Anthropic discovery while the models.dev catalog is unreachable", () => {
	let tempDir: string;
	let authStorage: AuthStorage;

	beforeEach(async () => {
		tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "pi-test-anthropic-row-base-"));
		authStorage = await AuthStorage.create(":memory:");
		authStorage.keys.setRuntime("anthropic", "sk-ant-test");
	});

	afterEach(() => {
		authStorage.close();
		if (tempDir && fs.existsSync(tempDir)) removeSyncWithRetries(tempDir);
	});

	test("a discovered model keeps the image input its cached row carries", async () => {
		const modelsPath = path.join(tempDir, "models.json");

		// First run: the catalog is composed, then a scoped refresh discovers the model.
		const first = new ModelRegistry(authStorage, modelsPath, { fetch: anthropicApi({ catalogReachable: true }) });
		first.getAll();
		await first.refreshDiscoverableProviders(["anthropic"], "online");
		expect(first.find("anthropic", MODEL_ID)?.input).toEqual(["text", "image"]);

		// Next run: the startup refresh reaches Anthropic but not the catalog.
		const second = new ModelRegistry(authStorage, modelsPath, { fetch: anthropicApi({ catalogReachable: false }) });
		await second.refresh("online");
		expect(second.find("anthropic", MODEL_ID)?.input).toEqual(["text", "image"]);
	});
});
