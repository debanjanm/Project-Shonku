export interface Agent {
  id: string;
  name: string;
  description: string;
}

export interface Kb {
  slug: string;
  name: string;
  description: string;
}

export interface Conversation {
  id: number;
  agent_type: string;
  source_type: string;
  source_ref: string;
  title: string | null;
  created_at: string;
  updated_at: string;
}

// GET /conversations/{id}/messages returns exactly these three fields
// (backend/db.py's get_messages: SELECT role, content, created_at) — no id.
export interface Message {
  role: "user" | "assistant";
  content: string;
  created_at: string;
}

export type ChatStreamEvent =
  | { type: "delta"; text: string }
  | { type: "product_images"; names: string[] };

export interface KbDocument {
  filename: string;
  size: number;
}

export interface IngestResult {
  new: number;
  changed: number;
  removed: number;
  unchanged: number;
  total_chunks: number;
}
