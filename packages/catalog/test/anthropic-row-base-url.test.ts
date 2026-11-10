/**
 * First-party Anthropic rows carry the provider's bare host
 * (`https://api.anthropic.com`) whatever base the registry hands discovery: an
 * earlier discovered row's `/v1`, or nothing before its catalog is composed.
 * The model-manager merge reads a base that differs from the cached or bundled
 * row of the same model as an endpoint change and then trusts only the new
 * row's capabilities, so a row that moved between the two forms dropped image
 * input whenever it was rebuilt without catalog metadata. Discovery itself
 * still requests `/v1/models` (#6563, which covers the bare-host base).
 */
import { describe, expect, it } from "bun:test";
import { anthropicModelManagerOptions } from "@oh-my-pi/pi-catalog/provider-models/openai-compat";
import type { FetchImpl } from "@oh-my-pi/pi-catalog/types";

const PROVIDER_BASE_URL = "https://api.anthropic.com";

function anthropicApi(modelsUrl: string, requested: string[]): FetchImpl {
	return async input => {
		const url = new URL(input instanceof Request ? input.url : input);
		const endpoint = `${url.origin}${url.pathname}`;
		requested.push(endpoint);
		if (url.href === "https://catalog.stencil.so/models.json.zstd") return Response.json({});
		if (endpoint !== modelsUrl) return new Response("not found", { status: 404 });
		return Response.json({
			data: [{ type: "model", id: "claude-test-row", display_name: "Claude Test Row" }],
			has_more: false,
		});
	};
}

describe("first-party Anthropic rows keep the provider base URL", () => {
	for (const baseUrl of [undefined, `${PROVIDER_BASE_URL}/v1`, `${PROVIDER_BASE_URL}/v1/`]) {
		it(`with config base ${baseUrl ?? "absent"}`, async () => {
			const requested: string[] = [];
			const models = await anthropicModelManagerOptions({
				apiKey: "sk-ant-test",
				baseUrl,
				fetch: anthropicApi(`${PROVIDER_BASE_URL}/v1/models`, requested),
			}).fetchDynamicModels?.();

			expect(requested).toContain(`${PROVIDER_BASE_URL}/v1/models`);
			expect(models?.map(model => model.baseUrl)).toEqual([PROVIDER_BASE_URL]);
		});
	}

	it("keeps a custom endpoint's configured base, `/v1` included", async () => {
		const proxy = "https://proxy.example.test/anthropic/v1";
		const requested: string[] = [];
		const models = await anthropicModelManagerOptions({
			apiKey: "sk-ant-test",
			baseUrl: proxy,
			fetch: anthropicApi(`${proxy}/models`, requested),
		}).fetchDynamicModels?.();

		expect(requested).toContain(`${proxy}/models`);
		expect(models?.map(model => model.baseUrl)).toEqual([proxy]);
	});
});
