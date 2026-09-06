import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import type { Kb } from "@/lib/types";

/** Mirrors frontend/app.py's sidebar source-picker branch per agent_type:
 * docqa gets a real KB dropdown, recommendation and the freeform agents
 * (story_developer/mystery_generator) have nothing to pick — just a caption. */
export function SourcePicker({
  agentType,
  kbs,
  selectedKbSlug,
  onSelectKb,
}: {
  agentType: string;
  kbs: Kb[];
  selectedKbSlug: string | null;
  onSelectKb: (slug: string) => void;
}) {
  if (agentType === "docqa") {
    const selected = kbs.find((kb) => kb.slug === selectedKbSlug);
    return (
      <div className="space-y-1.5">
        <p className="text-xs font-medium text-muted-foreground">Knowledge Base</p>
        <Select value={selectedKbSlug ?? undefined} onValueChange={onSelectKb}>
          <SelectTrigger className="w-full">
            <SelectValue placeholder="Choose a knowledge base" />
          </SelectTrigger>
          <SelectContent>
            {kbs.map((kb) => (
              <SelectItem key={kb.slug} value={kb.slug}>
                {kb.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        {selected && <p className="text-xs text-muted-foreground">{selected.description}</p>}
      </div>
    );
  }

  if (agentType === "recommendation") {
    return <p className="text-xs text-muted-foreground">Searching a 425-item fashion product catalog.</p>;
  }

  return <p className="text-xs text-muted-foreground">No source to pick — just describe your idea in the chat.</p>;
}
