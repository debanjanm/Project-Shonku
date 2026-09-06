import { Plus } from "lucide-react";

import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import type { Conversation } from "@/lib/types";

export function ConversationHistory({
  conversations,
  activeConversationId,
  canStartNewChat,
  onNewChat,
  onSelect,
}: {
  conversations: Conversation[];
  activeConversationId: number | null;
  canStartNewChat: boolean;
  onNewChat: () => void;
  onSelect: (id: number) => void;
}) {
  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3">
      <Separator />
      <Button onClick={onNewChat} disabled={!canStartNewChat} className="w-full" size="sm">
        <Plus className="h-4 w-4" />
        New chat
      </Button>

      {conversations.length > 0 && (
        <>
          <p className="px-1 text-xs font-medium text-muted-foreground">History</p>
          <ScrollArea className="min-h-0 flex-1">
            <div className="flex flex-col gap-1 pr-2">
              {conversations.map((conv) => (
                <Button
                  key={conv.id}
                  variant={conv.id === activeConversationId ? "secondary" : "ghost"}
                  className="h-auto w-full justify-start truncate py-2 text-left text-sm font-normal"
                  onClick={() => onSelect(conv.id)}
                >
                  <span className="truncate">{conv.title || "New chat"}</span>
                </Button>
              ))}
            </div>
          </ScrollArea>
        </>
      )}
    </div>
  );
}
