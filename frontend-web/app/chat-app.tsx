"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";

import { AgentPicker } from "@/components/agent-picker";
import { SourcePicker } from "@/components/source-picker";
import { ConversationHistory } from "@/components/conversation-history";
import { ChatMessages } from "@/components/chat-messages";
import { ChatInput } from "@/components/chat-input";
import { Skeleton } from "@/components/ui/skeleton";
import {
  createConversation,
  getAgents,
  getConversation,
  getKbs,
  getMessages,
  listConversations,
  streamChat,
} from "@/lib/api";
import type { Agent, Conversation, Kb, Message } from "@/lib/types";

/** Derives {source_type, source_ref} the same way frontend/app.py's sidebar
 * branch does per agent_type — docqa needs a KB picked, the rest are fixed. */
function deriveSource(agentType: string, kbSlug: string | null): { sourceType: string; sourceRef: string | null } {
  if (agentType === "docqa") return { sourceType: "kb", sourceRef: kbSlug };
  if (agentType === "recommendation") return { sourceType: "catalog", sourceRef: "fashion-500" };
  return { sourceType: "brief", sourceRef: "freeform" };
}

export function ChatApp() {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const conversationIdParam = searchParams.get("conversation_id");

  const [agents, setAgents] = useState<Agent[]>([]);
  const [kbs, setKbs] = useState<Kb[]>([]);
  const [agentType, setAgentType] = useState<string | null>(null);
  const [kbSlug, setKbSlug] = useState<string | null>(null);

  const [activeConversation, setActiveConversation] = useState<Conversation | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [conversations, setConversations] = useState<Conversation[]>([]);

  const [isStreaming, setIsStreaming] = useState(false);
  const [streamingText, setStreamingText] = useState("");
  const [error, setError] = useState<string | null>(null);

  const bottomRef = useRef<HTMLDivElement | null>(null);
  // Tracks the conversation id whose data is already loaded into local state
  // (set both here and by handleSend) so the URL-driven effect below doesn't
  // redundantly refetch — and race — a conversation handleSend just created
  // and is actively streaming into.
  const loadedConversationIdRef = useRef<number | null>(null);

  const { sourceType, sourceRef } = deriveSource(agentType ?? "", kbSlug);

  // Load agents once, default to the first one.
  useEffect(() => {
    getAgents()
      .then((loaded) => {
        setAgents(loaded);
        setAgentType((current) => current ?? loaded[0]?.id ?? null);
      })
      .catch(() => setError("Couldn't reach the backend. Is it running on :8000?"));
  }, []);

  // Load KBs once docqa is selected (cached after first load, like Streamlit's @st.cache_data).
  useEffect(() => {
    if (agentType === "docqa" && kbs.length === 0) {
      getKbs()
        .then((loaded) => {
          setKbs(loaded);
          setKbSlug((current) => current ?? loaded[0]?.slug ?? null);
        })
        .catch(() => setError("Couldn't load knowledge bases."));
    }
  }, [agentType, kbs.length]);

  // The conversation_id URL param is the single source of truth for the
  // active conversation — reload it (and its messages) whenever it changes,
  // and sync the pickers to match (mirrors app.py's default_agent_type).
  useEffect(() => {
    let cancelled = false;

    async function sync() {
      if (!conversationIdParam) {
        loadedConversationIdRef.current = null;
        if (cancelled) return;
        setActiveConversation(null);
        setMessages([]);
        return;
      }
      if (loadedConversationIdRef.current === Number(conversationIdParam)) {
        // Already loaded locally — e.g. handleSend just created this
        // conversation and is actively streaming into it. Refetching here
        // would race that in-progress local state.
        return;
      }
      const conv = await getConversation(conversationIdParam);
      if (cancelled) return;
      if (!conv) {
        setActiveConversation(null);
        setMessages([]);
        return;
      }
      setActiveConversation(conv);
      setAgentType(conv.agent_type);
      if (conv.agent_type === "docqa") setKbSlug(conv.source_ref);
      const msgs = await getMessages(conv.id);
      if (cancelled) return;
      setMessages(msgs);
      loadedConversationIdRef.current = conv.id;
    }

    sync();
    return () => {
      cancelled = true;
    };
  }, [conversationIdParam]);

  // A conversation stays locked to the agent+source it started with —
  // switching away from it in the pickers starts a fresh chat instead of
  // showing mismatched history.
  useEffect(() => {
    if (!activeConversation || !sourceRef) return;
    if (activeConversation.agent_type !== agentType || activeConversation.source_ref !== sourceRef) {
      router.replace(pathname);
    }
  }, [activeConversation, agentType, sourceRef, pathname, router]);

  const refreshHistory = useCallback(() => {
    if (!agentType || !sourceRef) return;
    listConversations(agentType, sourceType, sourceRef)
      .then(setConversations)
      .catch(() => setConversations([]));
  }, [agentType, sourceType, sourceRef]);

  useEffect(() => {
    refreshHistory();
  }, [refreshHistory]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streamingText]);

  function goToConversation(id: number | null) {
    router.push(id ? `${pathname}?conversation_id=${id}` : pathname);
  }

  async function handleNewChat() {
    if (!agentType || !sourceRef) return;
    const conv = await createConversation(agentType, sourceType, sourceRef);
    goToConversation(conv.id);
  }

  async function handleSend(text: string) {
    if (!agentType || !sourceRef) return;
    setError(null);

    let conv = activeConversation;
    if (!conv) {
      conv = await createConversation(agentType, sourceType, sourceRef);
      setActiveConversation(conv);
      loadedConversationIdRef.current = conv.id;
      router.replace(`${pathname}?conversation_id=${conv.id}`, { scroll: false });
    }

    setMessages((prev) => [
      ...prev,
      { role: "user", content: text, created_at: new Date().toISOString() },
    ]);
    setIsStreaming(true);
    setStreamingText("");

    let finalText = "";
    try {
      for await (const event of streamChat(conv.id, text)) {
        if (event.type === "delta") {
          finalText += event.text;
          setStreamingText(finalText);
        }
      }
    } catch {
      setError("The response stream failed partway through.");
    } finally {
      setIsStreaming(false);
    }

    if (finalText) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: finalText, created_at: new Date().toISOString() },
      ]);
    }
    setStreamingText("");
    refreshHistory();
  }

  const pickersReady = agentType !== null && (agentType !== "docqa" || kbSlug !== null);

  return (
    <div className="flex h-full">
      <aside className="flex w-72 shrink-0 flex-col gap-4 border-r bg-muted/30 p-4">
        <h1 className="text-sm font-semibold tracking-tight">Project Shonku</h1>
        {agents.length === 0 ? (
          <Skeleton className="h-20 w-full" />
        ) : (
          <AgentPicker agents={agents} value={agentType ?? agents[0].id} onChange={(id) => setAgentType(id)} />
        )}
        {agentType && <SourcePicker agentType={agentType} kbs={kbs} selectedKbSlug={kbSlug} onSelectKb={setKbSlug} />}
        <ConversationHistory
          conversations={conversations}
          activeConversationId={activeConversation?.id ?? null}
          canStartNewChat={pickersReady && !!sourceRef}
          onNewChat={handleNewChat}
          onSelect={goToConversation}
        />
      </aside>

      <main className="flex min-w-0 flex-1 flex-col">
        <div className="min-h-0 flex-1 overflow-y-auto">
          {error && (
            <p className="mx-auto mt-4 w-full max-w-3xl rounded-md border border-destructive/30 bg-destructive/10 px-4 py-2 text-sm text-destructive">
              {error}
            </p>
          )}
          {messages.length === 0 && !isStreaming ? (
            <p className="mx-auto max-w-3xl px-4 py-6 text-sm text-muted-foreground">
              Start typing to begin a new chat, or pick one from the sidebar.
            </p>
          ) : (
            <ChatMessages messages={messages} streamingText={streamingText} isStreaming={isStreaming} />
          )}
          <div ref={bottomRef} />
        </div>
        <ChatInput agentType={agentType ?? ""} disabled={!pickersReady || !sourceRef || isStreaming} onSend={handleSend} />
      </main>
    </div>
  );
}
