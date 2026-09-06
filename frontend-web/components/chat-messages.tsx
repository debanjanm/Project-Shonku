import { Bot, User } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { extractProductImages, ProductImageGrid } from "@/components/product-image-grid";
import type { Message } from "@/lib/types";

function Bubble({ role, children }: { role: "user" | "assistant"; children: React.ReactNode }) {
  const isUser = role === "user";
  return (
    <div className={`flex gap-3 ${isUser ? "flex-row-reverse" : ""}`}>
      <Avatar className="h-8 w-8 shrink-0">
        <AvatarFallback className={isUser ? "bg-primary text-primary-foreground" : "bg-muted"}>
          {isUser ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
        </AvatarFallback>
      </Avatar>
      <div
        className={`max-w-[75%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed ${
          isUser ? "bg-primary text-primary-foreground" : "bg-muted"
        }`}
      >
        {children}
      </div>
    </div>
  );
}

function MarkdownContent({ text }: { text: string }) {
  return (
    <div className="prose prose-sm dark:prose-invert max-w-none prose-p:my-1.5 prose-pre:my-2">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{text}</ReactMarkdown>
    </div>
  );
}

export function ChatMessages({
  messages,
  streamingText,
  isStreaming,
}: {
  messages: Message[];
  streamingText: string;
  isStreaming: boolean;
}) {
  return (
    <div className="mx-auto flex w-full max-w-3xl flex-col gap-6 px-4 py-6">
      {messages.map((msg, i) => {
        const { text, names } = extractProductImages(msg.content);
        return (
          // Messages are append-only and never reordered/deleted within a
          // render, and the backend doesn't return a message id (see
          // db.get_messages) — index is a safe, stable key here.
          <Bubble key={i} role={msg.role}>
            <MarkdownContent text={text} />
            {msg.role === "assistant" && <ProductImageGrid names={names} />}
          </Bubble>
        );
      })}
      {isStreaming && (
        <Bubble role="assistant">
          <MarkdownContent text={extractProductImages(streamingText).text || "…"} />
        </Bubble>
      )}
    </div>
  );
}
