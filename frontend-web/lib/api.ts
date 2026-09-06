import type { Agent, ChatStreamEvent, Conversation, IngestResult, Kb, KbDocument, Message } from "@/lib/types";

export const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";

/** FastAPI's HTTPException responses are {"detail": "..."} — surface that
 * real message instead of a generic one wherever it's available. */
async function errorMessage(res: Response, fallback: string): Promise<string> {
  try {
    const body = await res.json();
    return typeof body.detail === "string" ? body.detail : fallback;
  } catch {
    return fallback;
  }
}

export async function getAgents(): Promise<Agent[]> {
  const res = await fetch(`${BACKEND_URL}/agents`);
  if (!res.ok) throw new Error("Failed to load agents");
  return res.json();
}

export async function getKbs(): Promise<Kb[]> {
  const res = await fetch(`${BACKEND_URL}/kbs`);
  if (!res.ok) throw new Error("Failed to load knowledge bases");
  return res.json();
}

export async function createKb(name: string, description: string, slug?: string): Promise<Kb> {
  const res = await fetch(`${BACKEND_URL}/kbs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, description, slug: slug || undefined }),
  });
  if (!res.ok) throw new Error(await errorMessage(res, "Failed to create knowledge base"));
  return res.json();
}

export async function deleteKb(slug: string): Promise<void> {
  const res = await fetch(`${BACKEND_URL}/kbs/${slug}`, { method: "DELETE" });
  if (!res.ok) throw new Error(await errorMessage(res, "Failed to delete knowledge base"));
}

export async function listKbDocuments(slug: string): Promise<KbDocument[]> {
  const res = await fetch(`${BACKEND_URL}/kbs/${slug}/documents`);
  if (!res.ok) throw new Error(await errorMessage(res, "Failed to load documents"));
  return res.json();
}

export async function uploadKbDocument(slug: string, file: File): Promise<void> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${BACKEND_URL}/kbs/${slug}/documents`, { method: "POST", body: formData });
  if (!res.ok) throw new Error(await errorMessage(res, `Failed to upload ${file.name}`));
}

export async function deleteKbDocument(slug: string, filename: string): Promise<void> {
  const res = await fetch(`${BACKEND_URL}/kbs/${slug}/documents/${encodeURIComponent(filename)}`, {
    method: "DELETE",
  });
  if (!res.ok) throw new Error(await errorMessage(res, `Failed to delete ${filename}`));
}

export async function ingestKb(slug: string): Promise<IngestResult> {
  const res = await fetch(`${BACKEND_URL}/kbs/${slug}/ingest`, { method: "POST" });
  if (!res.ok) throw new Error(await errorMessage(res, "Ingest failed"));
  return res.json();
}

export async function getConversation(id: string | number): Promise<Conversation | null> {
  const res = await fetch(`${BACKEND_URL}/conversations/${id}`);
  if (!res.ok) return null;
  return res.json();
}

export async function listConversations(
  agentType: string,
  sourceType: string,
  sourceRef: string,
): Promise<Conversation[]> {
  const params = new URLSearchParams({ agent_type: agentType, source_type: sourceType, source_ref: sourceRef });
  const res = await fetch(`${BACKEND_URL}/conversations?${params}`);
  if (!res.ok) throw new Error("Failed to load conversations");
  return res.json();
}

export async function getMessages(conversationId: number): Promise<Message[]> {
  const res = await fetch(`${BACKEND_URL}/conversations/${conversationId}/messages`);
  if (!res.ok) throw new Error("Failed to load messages");
  return res.json();
}

export async function createConversation(
  agentType: string,
  sourceType: string,
  sourceRef: string,
): Promise<Conversation> {
  const res = await fetch(`${BACKEND_URL}/conversations`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ agent_type: agentType, source_type: sourceType, source_ref: sourceRef }),
  });
  if (!res.ok) throw new Error("Failed to create conversation");
  return res.json();
}

/**
 * Streams /chat's text/event-stream response. The backend always streams —
 * there's no non-streaming mode — and frames look like:
 *   data: {"delta": "..."}\n\n
 *   data: {"product_images": [...]}\n\n
 *   data: [DONE]\n\n
 * The browser's native EventSource only supports GET, so this is a manual
 * fetch + reader port of the same parsing loop frontend/app.py does with
 * requests.iter_lines() (see main.py's event_stream()).
 */
export async function* streamChat(conversationId: number, message: string): AsyncGenerator<ChatStreamEvent> {
  const res = await fetch(`${BACKEND_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ conversation_id: conversationId, message }),
  });
  if (!res.ok || !res.body) throw new Error("Chat request failed");

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const data = line.slice("data: ".length);
      if (data === "[DONE]") return;

      const payload = JSON.parse(data);
      if (typeof payload.delta === "string") {
        yield { type: "delta", text: payload.delta };
      } else if (Array.isArray(payload.product_images)) {
        yield { type: "product_images", names: payload.product_images };
      }
    }
  }
}
