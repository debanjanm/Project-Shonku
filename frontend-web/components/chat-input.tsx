"use client";

import { SendHorizontal } from "lucide-react";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";

const PLACEHOLDERS: Record<string, string> = {
  story_developer: "Describe your story idea...",
  mystery_generator: "Describe your story idea — I'll turn it into a mystery you solve...",
};

export function ChatInput({
  agentType,
  disabled,
  onSend,
}: {
  agentType: string;
  disabled: boolean;
  onSend: (message: string) => void;
}) {
  const [value, setValue] = useState("");

  function submit() {
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setValue("");
  }

  return (
    <div className="mx-auto flex w-full max-w-3xl items-end gap-2 px-4 pb-4">
      <Textarea
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            submit();
          }
        }}
        placeholder={PLACEHOLDERS[agentType] ?? "Ask something..."}
        disabled={disabled}
        rows={1}
        className="max-h-40 min-h-11 resize-none"
      />
      <Button size="icon" onClick={submit} disabled={disabled || !value.trim()} aria-label="Send message">
        <SendHorizontal className="h-4 w-4" />
      </Button>
    </div>
  );
}
