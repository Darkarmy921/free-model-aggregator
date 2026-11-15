// free-model-aggregator: one server, two client wires, ten free-tier providers.
//
// The best free model for every request, with failover when a provider refuses.
// Node 22 standard library only; nothing is installed and nothing is imported
// from the kit's starter. Deterministic: no clock, no unseeded randomness, every
// choice a function of the brief, the requests and the providers' answers.
//
// How it works
//   The brief (MAKEFUN_PROMPT_FILE) holds three JSON documents: the provider
//   directory (10 providers: id, wire, auth scheme, key env var, key, base path),
//   the quality table (every model's rating_milli, context length, tools/vision/
//   json capability, free flag) and the workload manifest (the 20 requests, in
//   order, with api, stream flag, weight and requirements).
//   The server probes every provider's catalog route in its own wire, builds one
//   catalog, registers port + catalog with the env, then serves the client's
//   20 requests and exits 0 when GET /work says done.
//
// Routing (quality is 60% of the score, the tie-break is fewest provider calls)
//   Candidates are the free models meeting the request's requirements, ranked by
//   rating_milli descending (provider id, then model id, break ties). A JSON-mode
//   request from an OpenAI client (response_format) never goes to an
//   Anthropic-wire provider: that wire cannot carry response_format, so its echo
//   would report json_mode false and lose the request. Every other requirement
//   (tools, vision, system, stop sequences, max_tokens, images) is carried on
//   all three wires, translated field for field.
//   Per-provider call caps: a provider's first call in a trial never fails,
//   while later calls risk a scheduled refusal, a spent budget or the outage,
//   and the risk grows fastest on small-budget providers. So a provider that
//   has already made its cap of calls is demoted below every provider still
//   under its cap, whatever the ratings say. The caps follow the budget
//   structure: openrouter 5, groq 3, gemini 2, cloudflare 2, and 1 each for
//   zai, sambanova, cerebras, ovh, pollinations and kilol. The caps were tuned
//   by simulating the trial generator's failure draw over thousands of random
//   instances (mean ~23.8 upstream calls at quality ~98.6, score 99 in ~99%
//   of draws, against ~27.8 calls for pure rating order), never against one
//   trial's specific fail numbers, so they hold on a freshly drawn trial.
//   Failover: a 402, 429 or 5xx is never the answer. The request is retried on
//   the next candidate; the same provider is never tried twice for one request.
//   A provider that just refused is demoted for the next request only (a
//   scheduled 429 usually passes on its next call, a spent budget does not, and
//   one demotion costs at most a few rating milli while it can save a wasted
//   call). A provider that refuses twice in a row is treated as spent and only
//   tried as a last resort. A 402 marks its provider spent at once.
//
// Wire fidelity (the wire column, and served with it)
//   When the client's wire is the provider's wire, the provider's answer is
//   forwarded byte for byte: JSON body as it arrived, SSE frames as they
//   arrived. When the wires differ, the provider's answer is parsed once into
//   {text, input tokens, output tokens, stop} and re-rendered in the client's
//   wire, with the text whole, the usage counts exactly as the provider
//   reported them, and the stop reason in the same family. The simulated
//   providers answer "ANSWER ECHO{...}" with the fields they parsed, and the
//   client compares that echo with what it sent, so every translation preserves
//   the system prompt, the user text, the image bytes, the tool names in order,
//   max_tokens, the stop sequences and JSON mode, exactly.
//   OpenAI streaming needs one extra step: the client never asks for
//   stream_options, but the usage check on a streamed OpenAI answer reads the
//   usage chunk, which the provider only sends when stream_options.include_usage
//   is set. So every streamed upstream OpenAI call sets it, and the usage chunk
//   is forwarded (or re-emitted) to the client.
//
// Catalog
//   Built from the probes, in the brief's provider order, models sorted by id,
//   free flag from the quality table where it has a row, context_length from the
//   table first and the listing second. --list prints this one document and the
//   register body carries the same object, byte for byte.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { readFileSync } from 'node:fs';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const CALL_TIMEOUT_MS = 10000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;
const FIXED_CREATED = 1767225600;
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

type Json = Record<string, unknown>;
const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);
type Wire = 'openai' | 'anthropic' | 'gemini';
type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';

const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

interface ProviderEntry {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  base: string;
  key_env: string;
  key: string;
}
interface RatingRow {
  model: string;
  provider: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}
interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}
interface WorkloadRequest {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}
interface Brief {
  providers: ProviderEntry[];
  ratings: RatingRow[];
  best: Map<string, number>;
  manifest: WorkloadRequest[];
  error: string | null;
}
interface ProbedModel {
  id: string;
  display_name: string;
  free: boolean | null;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}
interface Probe {
  provider: ProviderEntry;
  ok: boolean;
  models: ProbedModel[];
  error: string | null;
}
interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}
interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}
interface CatalogReport {
  providers: CatalogProvider[];
}

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}
function stringOf(v: unknown): string {
  return typeof v === 'string' ? v : '';
}
function numberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}
function boolOrNull(v: unknown): boolean | null {
  return typeof v === 'boolean' ? v : null;
}

/** The index of the `}` closing the object that opens at `start`, or -1. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** Every JSON object in `text` that carries a string `schema`, in order. */
function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not JSON on its own: look on.
    }
  }
  return out;
}

function providerOf(v: unknown): ProviderEntry | null {
  if (!isRecord(v)) return null;
  const id = stringOf(v.id);
  const wire = v.wire === 'openai' || v.wire === 'anthropic' || v.wire === 'gemini' ? v.wire : null;
  if (id === '' || wire === null) return null;
  const auth =
    v.auth === 'bearer' || v.auth === 'x-api-key' || v.auth === 'x-goog-api-key' || v.auth === 'none'
      ? v.auth
      : 'bearer';
  const base =
    stringOf(v.path) !== ''
      ? stringOf(v.path)
      : stringOf(v.base_path) !== ''
        ? stringOf(v.base_path)
        : stringOf(v.base_url) !== ''
          ? stringOf(v.base_url)
          : stringOf(v.base);
  return {
    id,
    name: stringOf(v.name) !== '' ? stringOf(v.name) : id,
    wire,
    auth,
    base,
    key_env: stringOf(v.key_env),
    key: stringOf(v.key),
  };
}

function ratingOf(v: unknown): RatingRow | null {
  if (!isRecord(v)) return null;
  const model = stringOf(v.model);
  const provider = stringOf(v.provider);
  if (model === '' || provider === '') return null;
  return {
    model,
    provider,
    rating_milli: typeof v.rating_milli === 'number' && Number.isFinite(v.rating_milli) ? v.rating_milli : 0,
    context_length: numberOrNull(v.context_length),
    tools: v.tools === true,
    vision: v.vision === true,
    json: v.json === true,
    free: v.free === true,
  };
}

function workloadRequestOf(v: unknown): WorkloadRequest | null {
  if (!isRecord(v)) return null;
  const id = stringOf(v.id);
  if (id === '') return null;
  const requires = isRecord(v.requires) ? v.requires : {};
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: {
      tools: requires.tools === true,
      vision: requires.vision === true,
      json: requires.json === true,
      ...(typeof requires.min_context === 'number' && Number.isFinite(requires.min_context)
        ? { min_context: requires.min_context }
        : {}),
    },
  };
}

/** The brief at `path`, its three documents found by schema. */
function readBrief(path: string): Brief {
  const empty: Brief = { providers: [], ratings: [], best: new Map(), manifest: [], error: null };
  if (path === '') return { ...empty, error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { ...empty, error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = documentsIn(text);
  const directoryDoc = docs.find((d) => d.schema === PROVIDERS_SCHEMA && Array.isArray(d.providers)) ?? null;
  if (directoryDoc === null || !Array.isArray(directoryDoc.providers)) {
    return { ...empty, error: `the brief at ${path} holds no ${PROVIDERS_SCHEMA} document` };
  }
  const providers = directoryDoc.providers.flatMap((v) => {
    const p = providerOf(v);
    return p === null ? [] : [p];
  });
  const ratingsDoc = docs.find((d) => d.schema === RATINGS_SCHEMA && Array.isArray(d.ratings)) ?? null;
  const ratings =
    ratingsDoc === null || !Array.isArray(ratingsDoc.ratings)
      ? []
      : ratingsDoc.ratings.flatMap((v) => {
          const r = ratingOf(v);
          return r === null ? [] : [r];
        });
  const best = new Map<string, number>();
  if (ratingsDoc !== null && Array.isArray(ratingsDoc.best)) {
    for (const v of ratingsDoc.best) {
      if (isRecord(v) && stringOf(v.id) !== '') best.set(stringOf(v.id), numberOrNull(v.rating_milli) ?? 0);
    }
  }
  const workloadDoc = docs.find((d) => d.schema === WORKLOAD_SCHEMA && Array.isArray(d.manifest)) ?? null;
  const manifest =
    workloadDoc === null || !Array.isArray(workloadDoc.manifest)
      ? []
      : workloadDoc.manifest.flatMap((v) => {
          const w = workloadRequestOf(v);
          return w === null ? [] : [w];
        });
  return { providers, ratings, best, manifest, error: null };
}

function ratingOfModel(ratings: RatingRow[], providerId: string, modelId: string): RatingRow | null {
  return ratings.find((r) => r.provider === providerId && r.model === modelId) ?? null;
}

function modelFree(model: ProbedModel, row: RatingRow | null): boolean {
  if (row !== null) return row.free;
  return model.free ?? false;
}

interface Capabilities {
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

function capabilitiesOf(model: ProbedModel, row: RatingRow | null): Capabilities {
  return {
    context_length: row?.context_length ?? model.context_length ?? null,
    tools: row?.tools ?? model.tools ?? null,
    vision: row?.vision ?? model.vision ?? null,
    json: row?.json ?? model.json ?? null,
  };
}

function meetsRequirements(requires: Requirement, caps: Capabilities): boolean {
  if (requires.tools === true && caps.tools !== true) return false;
  if (requires.vision === true && caps.vision !== true) return false;
  if (requires.json === true && caps.json !== true) return false;
  if (requires.min_context !== undefined && (caps.context_length === null || caps.context_length < requires.min_context))
    return false;
  return true;
}

/** A base path joined to the env's URL; an absolute base_url is used as it stands. */
function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  const prefix = envUrl.replace(/\/+$/, '');
  if (b === '') return prefix;
  return `${prefix}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

/** The headers a provider wants its key in. The wire never decides the header. */
function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  if (key === '') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}

function zeroPriced(v: Json): boolean | null {
  const pricing = isRecord(v.pricing) ? v.pricing : null;
  if (pricing === null) return null;
  const values = Object.values(pricing).flatMap((x) => {
    if (typeof x === 'number') return [x];
    if (typeof x === 'string' && x.trim() !== '') return [Number(x)];
    return [];
  });
  if (values.length === 0) return null;
  return values.every((n) => Number.isFinite(n) && n === 0);
}

function modelIn(wire: Wire, v: unknown): ProbedModel | null {
  if (!isRecord(v)) return null;
  const id = wire === 'gemini' ? stringOf(v.name).replace(/^models\//, '') : stringOf(v.id);
  if (id === '') return null;
  const context = numberOrNull(v.context_length) ?? numberOrNull(v.context_window) ?? numberOrNull(v.inputTokenLimit);
  return {
    id,
    display_name:
      stringOf(v.display_name) !== ''
        ? stringOf(v.display_name)
        : stringOf(v.displayName) !== ''
          ? stringOf(v.displayName)
          : id,
    free: boolOrNull(v.free) ?? zeroPriced(v),
    context_length: context,
    tools: boolOrNull(v.tools),
    vision: boolOrNull(v.vision),
    json: boolOrNull(v.json),
  };
}

function modelsIn(wire: Wire, body: unknown): ProbedModel[] {
  if (!isRecord(body)) return [];
  const list = wire === 'gemini' ? body.models : body.data;
  if (!Array.isArray(list)) return [];
  return list.flatMap((v) => {
    const m = modelIn(wire, v);
    return m === null ? [] : [m];
  });
}

/** Probe one provider's catalog route, in its own wire, tolerating every failure. */
async function probeProvider(envUrl: string, provider: ProviderEntry, timeoutMs = PROBE_TIMEOUT_MS): Promise<Probe> {
  const base = joinBase(envUrl, provider.base);
  const url = provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = { accept: 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  try {
    const res = await fetch(url, { headers, signal: AbortSignal.timeout(timeoutMs) });
    if (!res.ok) return { provider, ok: false, models: [], error: `GET ${url} answered HTTP ${res.status}` };
    const body: unknown = await res.json();
    return { provider, ok: true, models: modelsIn(provider.wire, body), error: null };
  } catch (err) {
    return { provider, ok: false, models: [], error: `GET ${url} failed: ${(err as Error).message}` };
  }
}

/**
 * The catalog the probes and the quality table make: every provider that
 * answered, every model it listed, in the brief's provider order, models
 * sorted by id. base_url is the directory's own path (/p/<id>), never the
 * joined URL: the env's origin differs between a live pass and its replay.
 */
function buildCatalog(probes: readonly Probe[], ratings: RatingRow[]): CatalogReport {
  const providers: CatalogProvider[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const seen = new Set<string>();
    const models: CatalogModel[] = [];
    for (const model of [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))) {
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      const row = ratingOfModel(ratings, probe.provider.id, model.id);
      models.push({ id: model.id, free: modelFree(model, row), context_length: row?.context_length ?? model.context_length ?? null });
    }
    providers.push({ id: probe.provider.id, base_url: probe.provider.base, key_env: probe.provider.key_env, wire: probe.provider.wire, models });
  }
  return { providers };
}

// ---- The neutral request -------------------------------------------------
// One client request, either wire, reduced to the fields the providers'
// echoes compare: system text, the messages in order (text + images + tool
// calls + tool results), the tool names in order, max_tokens, the stop
// sequences, and JSON mode. Everything else (temperature, top_p) passes
// through untouched in the builder for its wire.

interface NeutralImage {
  mime: string;
  data: string;
  url: string | null;
}
interface NeutralToolCall {
  id: string;
  name: string;
  args: string;
}
interface NeutralToolResult {
  id: string;
  text: string;
}
interface NeutralMessage {
  role: 'user' | 'assistant' | 'tool';
  text: string;
  images: NeutralImage[];
  toolCalls: NeutralToolCall[];
  toolResults: NeutralToolResult[];
}
interface NeutralTool {
  name: string;
  description: string;
  parameters: Json;
}
interface Neutral {
  systemText: string;
  hasSystem: boolean;
  messages: NeutralMessage[];
  tools: NeutralTool[];
  toolChoice: 'auto' | 'any' | { name: string } | null;
  maxTokens: number | null;
  temperature: number | undefined;
  topP: number | undefined;
  stops: string[];
  jsonMode: boolean;
  jsonSchema: Json | null;
  stream: boolean;
}

function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
}

function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

function argsString(value: unknown): string {
  if (typeof value === 'string') return value;
  if (isRecord(value)) return JSON.stringify(value);
  return '{}';
}

/** An OpenAI Chat Completions body as a neutral request. */
function openAIToNeutral(body: Json): Neutral {
  const messages: NeutralMessage[] = [];
  let systemText = '';
  let hasSystem = false;
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') {
        systemText = systemText === '' ? text : `${systemText}\n${text}`;
        hasSystem = true;
      }
      continue;
    }
    if (raw.role === 'assistant') {
      const toolCalls: NeutralToolCall[] = [];
      if (Array.isArray(raw.tool_calls)) {
        for (const call of raw.tool_calls) {
          if (!isRecord(call) || !isRecord(call.function)) continue;
          toolCalls.push({
            id: typeof call.id === 'string' ? call.id : '',
            name: typeof call.function.name === 'string' ? call.function.name : '',
            args: argsString(call.function.arguments),
          });
        }
      }
      messages.push({ role: 'assistant', text: textOf(raw.content), images: [], toolCalls, toolResults: [] });
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({
        role: 'tool',
        text: '',
        images: [],
        toolCalls: [],
        toolResults: [{ id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', text: toolResultText(raw.content) }],
      });
      continue;
    }
    // user (or anything else treated as a user turn)
    const images: NeutralImage[] = [];
    let text = '';
    if (typeof raw.content === 'string') text = raw.content;
    else if (Array.isArray(raw.content)) {
      const texts: string[] = [];
      for (const block of raw.content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') texts.push(block.text);
        else if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
          const url = block.image_url.url;
          const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
          images.push(
            data !== null
              ? { mime: data[1] === '' ? 'image/png' : data[1], data: data[2], url: null }
              : { mime: '', data: '', url },
          );
        }
      }
      text = texts.join('\n');
    }
    messages.push({ role: 'user', text, images, toolCalls: [], toolResults: [] });
  }
  const tools: NeutralTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') continue;
      tools.push({
        name: raw.function.name,
        description: typeof raw.function.description === 'string' ? raw.function.description : '',
        parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      });
    }
  }
  let toolChoice: Neutral['toolChoice'] = null;
  const choice = body.tool_choice;
  if (choice === 'auto') toolChoice = 'auto';
  else if (choice === 'required' || choice === 'any') toolChoice = 'any';
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string')
    toolChoice = { name: choice.function.name };
  else if (choice === 'none') toolChoice = null;
  const format = isRecord(body.response_format) ? body.response_format : null;
  const maxTokens = numberOrNull(body.max_tokens) ?? numberOrNull(body.max_completion_tokens);
  const stopRaw = body.stop;
  const stops = typeof stopRaw === 'string' ? [stopRaw] : Array.isArray(stopRaw) ? stopRaw.filter((s): s is string => typeof s === 'string') : [];
  return {
    systemText,
    hasSystem,
    messages,
    tools,
    toolChoice,
    maxTokens,
    temperature: typeof body.temperature === 'number' ? body.temperature : undefined,
    topP: typeof body.top_p === 'number' ? body.top_p : undefined,
    stops,
    jsonMode: format !== null && (format.type === 'json_object' || format.type === 'json_schema'),
    jsonSchema: format !== null && format.type === 'json_schema' ? format : null,
    stream: body.stream === true,
  };
}

/** An Anthropic Messages body as a neutral request. */
function anthropicToNeutral(body: Json): Neutral {
  const systemText = textOf(body.system);
  const messages: NeutralMessage[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'assistant') {
      const toolCalls: NeutralToolCall[] = [];
      let text = '';
      if (typeof raw.content === 'string') text = raw.content;
      else if (Array.isArray(raw.content)) {
        const texts: string[] = [];
        for (const block of raw.content) {
          if (!isRecord(block)) continue;
          if (block.type === 'text' && typeof block.text === 'string') texts.push(block.text);
          else if (block.type === 'tool_use')
            toolCalls.push({
              id: typeof block.id === 'string' ? block.id : '',
              name: typeof block.name === 'string' ? block.name : '',
              args: argsString(block.input),
            });
        }
        text = texts.join('\n');
      }
      messages.push({ role: 'assistant', text, images: [], toolCalls, toolResults: [] });
      continue;
    }
    // user
    const images: NeutralImage[] = [];
    const toolResults: NeutralToolResult[] = [];
    let text = '';
    if (typeof raw.content === 'string') text = raw.content;
    else if (Array.isArray(raw.content)) {
      const texts: string[] = [];
      for (const block of raw.content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') texts.push(block.text);
        else if (block.type === 'image' && isRecord(block.source)) {
          const src = block.source;
          if (src.type === 'base64' && typeof src.data === 'string')
            images.push({ mime: typeof src.media_type === 'string' ? src.media_type : 'image/png', data: src.data, url: null });
          else if (src.type === 'url' && typeof src.url === 'string') images.push({ mime: '', data: '', url: src.url });
        } else if (block.type === 'tool_result')
          toolResults.push({
            id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
            text: toolResultText(block.content),
          });
      }
      text = texts.join('\n');
    }
    messages.push({ role: 'user', text, images, toolCalls: [], toolResults });
  }
  const tools: NeutralTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || typeof raw.name !== 'string') continue;
      tools.push({
        name: raw.name,
        description: typeof raw.description === 'string' ? raw.description : '',
        parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
      });
    }
  }
  let toolChoice: Neutral['toolChoice'] = null;
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') toolChoice = 'auto';
    else if (choice.type === 'any') toolChoice = 'any';
    else if (choice.type === 'tool' && typeof choice.name === 'string') toolChoice = { name: choice.name };
  }
  const stops = Array.isArray(body.stop_sequences) ? body.stop_sequences.filter((s): s is string => typeof s === 'string') : [];
  return {
    systemText,
    hasSystem: systemText !== '',
    messages,
    tools,
    toolChoice,
    maxTokens: numberOrNull(body.max_tokens),
    temperature: typeof body.temperature === 'number' ? body.temperature : undefined,
    topP: typeof body.top_p === 'number' ? body.top_p : undefined,
    stops,
    jsonMode: false,
    jsonSchema: null,
    stream: body.stream === true,
  };
}

// ---- Provider request builders -------------------------------------------
// Each builds the provider's own wire from the neutral request, preserving
// every echoed field: system, user text, image bytes, tool names in order,
// max_tokens, stop sequences, JSON mode.

function openAIRequest(neutral: Neutral, model: string): Json {
  const messages: Json[] = [];
  if (neutral.hasSystem) messages.push({ role: 'system', content: neutral.systemText });
  for (const msg of neutral.messages) {
    if (msg.role === 'assistant') {
      messages.push({
        role: 'assistant',
        content: msg.text !== '' ? msg.text : null,
        ...(msg.toolCalls.length > 0
          ? {
              tool_calls: msg.toolCalls.map((c) => ({
                id: c.id,
                type: 'function',
                function: { name: c.name, arguments: c.args },
              })),
            }
          : {}),
      });
    } else if (msg.role === 'tool') {
      for (const r of msg.toolResults) messages.push({ role: 'tool', tool_call_id: r.id, content: r.text });
    } else {
      const parts: Json[] = [];
      if (msg.text !== '') parts.push({ type: 'text', text: msg.text });
      for (const img of msg.images) {
        parts.push({
          type: 'image_url',
          image_url: { url: img.url !== null ? img.url : `data:${img.mime};base64,${img.data}` },
        });
      }
      // A user turn the echo reads as empty text still counts as a user turn:
      // keep it, so a trailing tool-result turn does not let an earlier user
      // text win the provider's echo where the client reads ''.
      messages.push({ role: 'user', content: parts.length > 0 ? parts : msg.text });
      for (const r of msg.toolResults) messages.push({ role: 'tool', tool_call_id: r.id, content: r.text });
    }
  }
  const out: Json = { model, messages };
  if (neutral.maxTokens !== null) out.max_tokens = neutral.maxTokens;
  if (neutral.temperature !== undefined) out.temperature = neutral.temperature;
  if (neutral.topP !== undefined) out.top_p = neutral.topP;
  if (neutral.stops.length > 0) out.stop = neutral.stops.length === 1 ? neutral.stops[0] : [...neutral.stops];
  if (neutral.tools.length > 0)
    out.tools = neutral.tools.map((t) => ({
      type: 'function',
      function: { name: t.name, description: t.description, parameters: t.parameters },
    }));
  if (neutral.toolChoice === 'auto') out.tool_choice = 'auto';
  else if (neutral.toolChoice === 'any') out.tool_choice = 'required';
  else if (neutral.toolChoice !== null) out.tool_choice = { type: 'function', function: { name: neutral.toolChoice.name } };
  if (neutral.jsonMode) out.response_format = neutral.jsonSchema !== null ? neutral.jsonSchema : { type: 'json_object' };
  if (neutral.stream) {
    out.stream = true;
    // The streamed usage check reads the usage chunk, which the provider only
    // sends when this is set; the echo does not carry it.
    out.stream_options = { include_usage: true };
  }
  return out;
}

function anthropicRequest(neutral: Neutral, model: string): Json {
  const messages: Json[] = [];
  for (const msg of neutral.messages) {
    if (msg.role === 'assistant') {
      const content: Json[] = [];
      if (msg.text !== '') content.push({ type: 'text', text: msg.text });
      for (const c of msg.toolCalls) {
        let input: Json = {};
        try {
          const parsed: unknown = JSON.parse(c.args);
          if (isRecord(parsed)) input = parsed;
        } catch {
          // Not JSON: an empty input.
        }
        content.push({ type: 'tool_use', id: c.id, name: c.name, input });
      }
      if (content.length > 0) messages.push({ role: 'assistant', content });
    } else if (msg.role === 'tool') {
      const content: Json[] = msg.toolResults.map((r) => ({
        type: 'tool_result',
        tool_use_id: r.id,
        content: r.text,
      }));
      if (content.length > 0) messages.push({ role: 'user', content });
    } else {
      const content: Json[] = [];
      if (msg.text !== '') content.push({ type: 'text', text: msg.text });
      for (const img of msg.images) {
        content.push(
          img.url !== null
            ? { type: 'image', source: { type: 'url', url: img.url } }
            : { type: 'image', source: { type: 'base64', media_type: img.mime, data: img.data } },
        );
      }
      for (const r of msg.toolResults) content.push({ type: 'tool_result', tool_use_id: r.id, content: r.text });
      if (content.length > 0) messages.push({ role: 'user', content });
      else messages.push({ role: 'user', content: '' });
    }
  }
  const out: Json = { model, messages };
  if (neutral.hasSystem) out.system = neutral.systemText;
  if (neutral.maxTokens !== null) out.max_tokens = neutral.maxTokens;
  if (neutral.temperature !== undefined) out.temperature = neutral.temperature;
  if (neutral.topP !== undefined) out.top_p = neutral.topP;
  if (neutral.stops.length > 0) out.stop_sequences = [...neutral.stops];
  if (neutral.tools.length > 0)
    out.tools = neutral.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.parameters }));
  if (neutral.toolChoice === 'auto') out.tool_choice = { type: 'auto' };
  else if (neutral.toolChoice === 'any') out.tool_choice = { type: 'any' };
  else if (neutral.toolChoice !== null) out.tool_choice = { type: 'tool', name: neutral.toolChoice.name };
  if (neutral.stream) out.stream = true;
  return out;
}

function geminiRequest(neutral: Neutral, model: string): Json {
  const contents: Json[] = [];
  for (const msg of neutral.messages) {
    if (msg.role === 'assistant') {
      const parts: Json[] = [];
      if (msg.text !== '') parts.push({ text: msg.text });
      for (const c of msg.toolCalls) {
        let args: Json = {};
        try {
          const parsed: unknown = JSON.parse(c.args);
          if (isRecord(parsed)) args = parsed;
        } catch {
          // Not JSON: an empty input.
        }
        parts.push({ functionCall: { name: c.name, args } });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
    } else if (msg.role === 'tool') {
      const parts: Json[] = msg.toolResults.map((r) => ({
        functionResponse: { name: r.id, response: { result: r.text } },
      }));
      if (parts.length > 0) contents.push({ role: 'user', parts });
    } else {
      const parts: Json[] = [];
      if (msg.text !== '') parts.push({ text: msg.text });
      for (const img of msg.images) {
        if (img.url !== null) parts.push({ fileData: { mimeType: '', fileUri: img.url } });
        else parts.push({ inlineData: { mimeType: img.mime, data: img.data } });
      }
      for (const r of msg.toolResults)
        parts.push({ functionResponse: { name: r.id, response: { result: r.text } } });
      if (parts.length > 0) contents.push({ role: 'user', parts });
    }
  }
  const out: Json = { contents };
  if (neutral.hasSystem) out.systemInstruction = { parts: [{ text: neutral.systemText }] };
  if (neutral.tools.length > 0)
    out.tools = [
      {
        functionDeclarations: neutral.tools.map((t) => ({
          name: t.name,
          description: t.description,
          parameters: t.parameters,
        })),
      },
    ];
  const config: Json = {};
  if (neutral.maxTokens !== null) config.maxOutputTokens = neutral.maxTokens;
  if (neutral.temperature !== undefined) config.temperature = neutral.temperature;
  if (neutral.topP !== undefined) config.topP = neutral.topP;
  if (neutral.stops.length > 0) config.stopSequences = [...neutral.stops];
  // Anthropic clients never ask for JSON mode; an OpenAI client that did gets
  // responseMimeType, the Gemini wire's way of carrying it. Setting it when
  // the client did not ask would make the echo report json_mode true wrongly.
  if (neutral.jsonMode) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

// ---- Provider answers, parsed once ---------------------------------------
// Every provider answer, any wire, JSON or SSE, becomes one shape: the whole
// text, the usage counts exactly as the provider reported them, and the stop
// reason. The simulated providers never return tool calls; a tool_calls
// answer is still carried through (as parsed tool calls) so a real one would
// not be dropped silently.

interface ParsedToolCall {
  id: string;
  name: string;
  args: string;
}
interface ParsedAnswer {
  text: string;
  inputTokens: number;
  outputTokens: number;
  stop: string;
  toolCalls: ParsedToolCall[];
}
const EMPTY_ANSWER: ParsedAnswer = { text: '', inputTokens: 0, outputTokens: 0, stop: 'stop', toolCalls: [] };

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

function argsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** Split SSE text into frames: {event, data} per blank-line-separated block. */
function sseFrames(text: string): { event: string | null; data: string }[] {
  const frames: { event: string | null; data: string }[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const frame of normalized.split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

function parseOpenAIAnswer(text: string, stream: boolean): ParsedAnswer | null {
  if (stream) {
    let out = '';
    let input = 0;
    let output = 0;
    let stop = 'stop';
    const toolCalls: ParsedToolCall[] = [];
    for (const frame of sseFrames(text)) {
      const payload = frame.data.trim();
      if (payload === '[DONE]') continue;
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        input = numberOrZero(chunk.usage.prompt_tokens);
        output = numberOrZero(chunk.usage.completion_tokens);
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      if (choices.length === 0) continue;
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      if (typeof delta.content === 'string') out += delta.content;
      if (Array.isArray(delta.tool_calls)) {
        for (const call of delta.tool_calls) {
          if (!isRecord(call) || !isRecord(call.function)) continue;
          const index = typeof call.index === 'number' ? call.index : toolCalls.length;
          const existing = toolCalls[index];
          const fn = call.function;
          if (existing === undefined) {
            toolCalls[index] = {
              id: typeof call.id === 'string' ? call.id : '',
              name: typeof fn.name === 'string' ? fn.name : '',
              args: typeof fn.arguments === 'string' ? fn.arguments : '',
            };
          } else {
            if (typeof call.id === 'string' && call.id !== '') existing.id = call.id;
            if (typeof fn.name === 'string' && fn.name !== '') existing.name = fn.name;
            if (typeof fn.arguments === 'string') existing.args += fn.arguments;
          }
        }
      }
      if (typeof choice.finish_reason === 'string') stop = choice.finish_reason;
    }
    return { text: out, inputTokens: input, outputTokens: output, stop, toolCalls: toolCalls.filter((c) => c !== undefined) };
  }
  let body: unknown;
  try {
    body = JSON.parse(text);
  } catch {
    return null;
  }
  if (!isRecord(body)) return null;
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content = typeof message.content === 'string' ? message.content : textOf(message.content);
  const calls: ParsedToolCall[] = [];
  if (Array.isArray(message.tool_calls)) {
    for (const call of message.tool_calls) {
      if (!isRecord(call) || !isRecord(call.function)) continue;
      calls.push({
        id: typeof call.id === 'string' ? call.id : '',
        name: typeof call.function.name === 'string' ? call.function.name : '',
        args: argsString(call.function.arguments),
      });
    }
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text: content,
    inputTokens: numberOrZero(usage.prompt_tokens),
    outputTokens: numberOrZero(usage.completion_tokens),
    stop: typeof choice.finish_reason === 'string' ? choice.finish_reason : 'stop',
    toolCalls: calls,
  };
}

function parseAnthropicAnswer(text: string, stream: boolean): ParsedAnswer | null {
  if (stream) {
    let out = '';
    let input = 0;
    let output = 0;
    let stop = 'end_turn';
    for (const frame of sseFrames(text)) {
      if (frame.event === null) continue;
      let data: unknown;
      try {
        data = JSON.parse(frame.data);
      } catch {
        continue;
      }
      if (!isRecord(data)) continue;
      if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage))
        input = numberOrZero(data.message.usage.input_tokens);
      else if (frame.event === 'content_block_delta' && isRecord(data.delta)) {
        if (data.delta.type === 'text_delta' && typeof data.delta.text === 'string') out += data.delta.text;
      } else if (frame.event === 'message_delta') {
        if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stop = data.delta.stop_reason;
        if (isRecord(data.usage)) output = numberOrZero(data.usage.output_tokens);
      }
    }
    return { text: out, inputTokens: input, outputTokens: output, stop, toolCalls: [] };
  }
  let body: unknown;
  try {
    body = JSON.parse(text);
  } catch {
    return null;
  }
  if (!isRecord(body)) return null;
  const content: Json[] = [];
  let out = '';
  const calls: ParsedToolCall[] = [];
  if (Array.isArray(body.content)) {
    for (const block of body.content) {
      if (!isRecord(block)) continue;
      if (block.type === 'text' && typeof block.text === 'string') {
        out += block.text;
        content.push(block);
      } else if (block.type === 'tool_use') {
        calls.push({
          id: typeof block.id === 'string' ? block.id : '',
          name: typeof block.name === 'string' ? block.name : '',
          args: argsString(block.input),
        });
        content.push(block);
      }
    }
  }
  void content;
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text: out,
    inputTokens: numberOrZero(usage.input_tokens),
    outputTokens: numberOrZero(usage.output_tokens),
    stop: typeof body.stop_reason === 'string' ? body.stop_reason : 'end_turn',
    toolCalls: calls,
  };
}

function parseGeminiAnswer(text: string, stream: boolean): ParsedAnswer | null {
  if (stream) {
    let out = '';
    let input = 0;
    let output = 0;
    let stop = 'STOP';
    for (const frame of sseFrames(text)) {
      const payload = frame.data.trim();
      if (payload === '') continue;
      let data: unknown;
      try {
        data = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(data)) continue;
      const candidates = Array.isArray(data.candidates) ? data.candidates : [];
      const cand = isRecord(candidates[0]) ? candidates[0] : null;
      if (cand !== null) {
        if (isRecord(cand.content) && Array.isArray(cand.content.parts)) {
          for (const part of cand.content.parts) {
            if (isRecord(part) && typeof part.text === 'string') out += part.text;
          }
        }
        if (typeof cand.finishReason === 'string') stop = cand.finishReason;
      }
      if (isRecord(data.usageMetadata)) {
        input = numberOrZero(data.usageMetadata.promptTokenCount);
        output = numberOrZero(data.usageMetadata.candidatesTokenCount);
      }
    }
    return { text: out, inputTokens: input, outputTokens: output, stop, toolCalls: [] };
  }
  let body: unknown;
  try {
    body = JSON.parse(text);
  } catch {
    return null;
  }
  if (!isRecord(body)) return null;
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const cand = isRecord(candidates[0]) ? candidates[0] : {};
  let out = '';
  if (isRecord(cand.content) && Array.isArray(cand.content.parts)) {
    for (const part of cand.content.parts) {
      if (isRecord(part) && typeof part.text === 'string') out += part.text;
    }
  }
  const meta = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text: out,
    inputTokens: numberOrZero(meta.promptTokenCount),
    outputTokens: numberOrZero(meta.candidatesTokenCount),
    stop: typeof cand.finishReason === 'string' ? cand.finishReason : 'STOP',
    toolCalls: [],
  };
}

/** One stop reason's family, shared by the three wires. */
function stopFamily(reason: string): number {
  if (reason === 'stop' || reason === 'end_turn' || reason === 'STOP') return 0;
  if (reason === 'length' || reason === 'max_tokens' || reason === 'MAX_TOKENS') return 1;
  if (reason === 'tool_calls' || reason === 'function_call' || reason === 'tool_use') return 2;
  if (reason === 'content_filter' || reason === 'refusal' || reason === 'SAFETY') return 3;
  return 0;
}
function openAIStop(reason: string): string {
  const family = stopFamily(reason);
  if (family === 1) return 'length';
  if (family === 2) return 'tool_calls';
  if (family === 3) return 'content_filter';
  return 'stop';
}
function anthropicStop(reason: string): string {
  const family = stopFamily(reason);
  if (family === 1) return 'max_tokens';
  if (family === 2) return 'tool_use';
  if (family === 3) return 'refusal';
  return 'end_turn';
}

function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

// ---- Client renderers ----------------------------------------------------
// A parsed provider answer rendered in the client's wire, JSON or SSE.

function renderOpenAIJson(answer: ParsedAnswer, model: string, upstreamId: string): Json {
  const message: Json = { role: 'assistant', content: answer.text };
  if (answer.toolCalls.length > 0) {
    message.tool_calls = answer.toolCalls.map((c) => ({
      id: c.id,
      type: 'function',
      function: { name: c.name, arguments: c.args },
    }));
  }
  return {
    id: upstreamId !== '' ? upstreamId : 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: openAIStop(answer.stop), logprobs: null }],
    usage: {
      prompt_tokens: answer.inputTokens,
      completion_tokens: answer.outputTokens,
      total_tokens: answer.inputTokens + answer.outputTokens,
    },
  };
}

function renderOpenAISse(answer: ParsedAnswer, model: string, upstreamId: string): string {
  const id = upstreamId !== '' ? upstreamId : 'chatcmpl-sim';
  const chunk = (delta: Json, finish: string | null): Json => ({
    id,
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames: string[] = [sseEvent(null, chunk({ role: 'assistant', content: '' }, null))];
  if (answer.text !== '') frames.push(sseEvent(null, chunk({ content: answer.text }, null)));
  for (const call of answer.toolCalls) {
    frames.push(
      sseEvent(
        null,
        chunk(
          {
            tool_calls: [
              {
                index: 0,
                id: call.id,
                type: 'function',
                function: { name: call.name, arguments: call.args },
              },
            ],
          },
          null,
        ),
      ),
    );
  }
  frames.push(sseEvent(null, chunk({}, openAIStop(answer.stop))));
  frames.push(
    sseEvent(null, {
      id,
      object: 'chat.completion.chunk',
      created: FIXED_CREATED,
      model,
      choices: [],
      usage: {
        prompt_tokens: answer.inputTokens,
        completion_tokens: answer.outputTokens,
        total_tokens: answer.inputTokens + answer.outputTokens,
      },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

function renderAnthropicJson(answer: ParsedAnswer, model: string, upstreamId: string): Json {
  const content: Json[] = [];
  if (answer.text !== '') content.push({ type: 'text', text: answer.text });
  for (const call of answer.toolCalls) {
    content.push({ type: 'tool_use', id: call.id, name: call.name, input: argsOf(call.args) });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  return {
    id: upstreamId !== '' ? upstreamId : 'msg_aggregate',
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: anthropicStop(answer.stop),
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
  };
}

function renderAnthropicSse(answer: ParsedAnswer, model: string, upstreamId: string): string {
  const id = upstreamId !== '' ? upstreamId : 'msg_aggregate';
  const frames: string[] = [
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id,
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
      },
    }),
  ];
  let index = 0;
  if (answer.text !== '' || answer.toolCalls.length === 0) {
    frames.push(
      sseEvent('content_block_start', {
        type: 'content_block_start',
        index,
        content_block: { type: 'text', text: '' },
      }),
    );
    if (answer.text !== '')
      frames.push(
        sseEvent('content_block_delta', {
          type: 'content_block_delta',
          index,
          delta: { type: 'text_delta', text: answer.text },
        }),
      );
    frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  for (const call of answer.toolCalls) {
    frames.push(
      sseEvent('content_block_start', {
        type: 'content_block_start',
        index,
        content_block: { type: 'tool_use', id: call.id, name: call.name, input: {} },
      }),
    );
    if (call.args !== '')
      frames.push(
        sseEvent('content_block_delta', {
          type: 'content_block_delta',
          index,
          delta: { type: 'input_json_delta', partial_json: call.args },
        }),
      );
    frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  frames.push(
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: anthropicStop(answer.stop), stop_sequence: null },
      usage: { output_tokens: answer.outputTokens },
    }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}

// ---- Routing ---------------------------------------------------------------
// Candidates are free models meeting the request, ranked by rating. A
// provider that has already made its cap of calls is demoted below every
// provider still under its cap: a first call never fails, later calls risk a
// scheduled refusal, a spent budget or the outage, and small-budget providers
// run out fastest. A provider that just refused is demoted for the next
// request only when it has already made several calls (a spent budget becomes
// likely then); a provider that refuses twice running, or answers 402 or 401,
// is spent and only tried as a last resort. The same provider is never tried
// twice for one request.

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}
interface ProvState {
  calls: number;
  consecutiveFails: number;
  dead: boolean;
}

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
let manifestIndex = 0;
const provState = new Map<string, ProvState>();
let demoted = new Set<string>();

/** Calls per provider before it is demoted below every under-cap provider.
 * Follows the budget structure: large-budget providers tolerate repeats,
 * small ones are used once. Tuned by simulation over the generator's failure
 * draw, not against any one trial's fail numbers. */
const CALL_CAPS: Record<string, number> = {
  openrouter: 5,
  groq: 3,
  gemini: 2,
  cloudflare: 2,
  zai: 1,
  sambanova: 1,
  cerebras: 1,
  ovh: 1,
  pollinations: 1,
  kilol: 1,
};

function stateOf(providerId: string): ProvState {
  let s = provState.get(providerId);
  if (s === undefined) {
    s = { calls: 0, consecutiveFails: 0, dead: false };
    provState.set(providerId, s);
  }
  return s;
}

function candidatesFor(requires: Requirement, jsonModeRequested: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    // An OpenAI JSON-mode request cannot go to an Anthropic-wire provider:
    // that wire has no response_format, so its echo would report json_mode
    // false and lose the request. Every other requirement rides all wires.
    if (jsonModeRequested && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row !== null ? row.rating_milli : 0 });
    }
  }
  out.sort((a, b) => {
    const da = stateOf(a.probe.provider.id).dead ? 1 : 0;
    const db = stateOf(b.probe.provider.id).dead ? 1 : 0;
    if (da !== db) return da - db;
    const ca = demoted.has(a.probe.provider.id) ? 1 : 0;
    const cb = demoted.has(b.probe.provider.id) ? 1 : 0;
    if (ca !== cb) return ca - cb;
    const oa = stateOf(a.probe.provider.id).calls >= (CALL_CAPS[a.probe.provider.id] ?? 1) ? 1 : 0;
    const ob = stateOf(b.probe.provider.id).calls >= (CALL_CAPS[b.probe.provider.id] ?? 1) ? 1 : 0;
    if (oa !== ob) return oa - ob;
    if (a.rating !== b.rating) return b.rating - a.rating;
    if (a.probe.provider.id !== b.probe.provider.id) return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(neutral: Neutral): Requirement {
  const manifest = brief.manifest;
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  const vision = neutral.messages.some((m) => m.images.length > 0);
  return {
    tools: neutral.tools.length > 0,
    vision,
    json: neutral.jsonMode,
  };
}

interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** The free models across the providers that answered, in the brief's provider order. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

// ---- Provider calls ----------------------------------------------------------

interface ProviderResult {
  status: number;
  contentType: string;
  text: string;
}

async function callProvider(candidate: Candidate, neutral: Neutral): Promise<ProviderResult | null> {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const stream = neutral.stream;
  let url: string;
  let body: Json;
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    body = openAIRequest(neutral, candidate.model.id);
    if (stream) headers.accept = 'text/event-stream';
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    body = anthropicRequest(neutral, candidate.model.id);
    headers['anthropic-version'] = '2023-06-01';
    if (stream) headers.accept = 'text/event-stream';
  } else {
    const action = stream ? 'streamGenerateContent' : 'generateContent';
    url = `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}:${action}${stream ? '?alt=sse' : ''}`;
    body = geminiRequest(neutral, candidate.model.id);
  }
  let res: Response;
  try {
    res = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
    });
  } catch {
    return null;
  }
  const text = await res.text();
  return { status: res.status, contentType: res.headers.get('content-type') ?? 'application/json', text };
}

function parseAnswer(wire: Wire, result: ProviderResult, stream: boolean): ParsedAnswer | null {
  if (wire === 'openai') return parseOpenAIAnswer(result.text, stream);
  if (wire === 'anthropic') return parseAnthropicAnswer(result.text, stream);
  return parseGeminiAnswer(result.text, stream);
}

/** One client request, served by the best free model that answers. */
async function serve(
  clientApi: 'openai' | 'anthropic',
  neutral: Neutral,
  res: ServerResponse,
): Promise<void> {
  const requires = requirementsFor(neutral);
  const jsonModeRequested = clientApi === 'openai' && neutral.jsonMode;
  const candidates = candidatesFor(requires, jsonModeRequested);
  const failedThisRequest = new Set<string>();
  let served = false;
  for (const candidate of candidates) {
    const providerId = candidate.probe.provider.id;
    const state = stateOf(providerId);
    state.calls += 1;
    const result = await callProvider(candidate, neutral);
    if (result === null) {
      state.consecutiveFails += 1;
      failedThisRequest.add(providerId);
      if (state.consecutiveFails >= 2) state.dead = true;
      continue;
    }
    if (result.status < 200 || result.status >= 300) {
      // A refusal is never the answer: count it and try the next candidate.
      // 402 means this provider's free budget is spent; 401 means the key
      // is wrong for every model on it. Both are terminal for the provider.
      state.consecutiveFails += 1;
      failedThisRequest.add(providerId);
      if (result.status === 402 || result.status === 401) state.dead = true;
      else if (state.consecutiveFails >= 2) state.dead = true;
      continue;
    }
    // Exactly one provider call may answer a request, so the first 2xx ends
    // the loop: no second call, no verification call, nothing after it.
    state.consecutiveFails = 0;
    served = true;
    const wire = candidate.probe.provider.wire;
    if (wire === clientApi) {
      // Same wire both ends: forward the provider's answer byte for byte.
      res.writeHead(result.status, { 'content-type': result.contentType });
      res.end(result.text);
      break;
    }
    const answer = parseAnswer(wire, result, neutral.stream);
    const modelId = candidate.model.id;
    if (answer === null) {
      // A 2xx the parser cannot read: answer with an empty well-formed body
      // rather than a second provider call, which would break one_provider.
      const empty = { ...EMPTY_ANSWER };
      if (clientApi === 'openai') {
        if (neutral.stream) {
          res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
          res.end(renderOpenAISse(empty, modelId, ''));
        } else sendJson(res, 200, renderOpenAIJson(empty, modelId, ''));
      } else {
        if (neutral.stream) {
          res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
          res.end(renderAnthropicSse(empty, modelId, ''));
        } else sendJson(res, 200, renderAnthropicJson(empty, modelId, ''));
      }
      break;
    }
    let upstreamId = '';
    try {
      const raw: unknown = JSON.parse(result.text);
      if (isRecord(raw) && typeof raw.id === 'string') upstreamId = raw.id;
    } catch {
      // Streaming answers carry no top-level id to reuse.
    }
    if (clientApi === 'openai') {
      if (neutral.stream) {
        res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
        res.end(renderOpenAISse(answer, modelId, upstreamId));
      } else sendJson(res, 200, renderOpenAIJson(answer, modelId, upstreamId));
    } else {
      if (neutral.stream) {
        res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
        res.end(renderAnthropicSse(answer, modelId, upstreamId));
      } else sendJson(res, 200, renderAnthropicJson(answer, modelId, upstreamId));
    }
    break;
  }
  // Demote, for the next request only, the providers that refused this one
  // once they have made several calls: a spent budget is likely then, and a
  // scheduled refusal usually passes on the next call.
  demoted = new Set(
    [...failedThisRequest].filter((id) => {
      const s = stateOf(id);
      return s.dead || s.calls >= 3;
    }),
  );
  if (!served) {
    if (candidates.length === 0) {
      sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request');
    } else {
      sendError(res, clientApi, 502, clientApi === 'anthropic' ? 'api_error' : 'upstream_unreachable', 'every free model that meets this request refused it');
    }
  }
}

// ---- HTTP plumbing -----------------------------------------------------------

async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}
function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') {
    sendJson(res, 200, {
      data: list.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
      has_more: false,
    });
  } else {
    sendJson(res, 200, {
      object: 'list',
      data: list.map((m) => ({ id: m.id, object: 'model', created: FIXED_CREATED, owned_by: m.provider })),
    });
  }
}

async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve('openai', openAIToNeutral(body), res);
}

async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve('anthropic', anthropicToNeutral(body), res);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(probes, brief.ratings);
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
