import { Button } from "@/components/ui/button";
import { KbCreateDialog } from "@/components/kb-create-dialog";
import type { Kb } from "@/lib/types";

export function KbList({
  kbs,
  selectedSlug,
  onSelect,
  onCreate,
}: {
  kbs: Kb[];
  selectedSlug: string | null;
  onSelect: (slug: string) => void;
  onCreate: (name: string, description: string) => Promise<void>;
}) {
  return (
    <aside className="flex w-72 shrink-0 flex-col gap-3 border-r bg-muted/30 p-4">
      <h2 className="text-sm font-semibold tracking-tight">Knowledge Bases</h2>
      <KbCreateDialog onCreate={onCreate} />
      <div className="flex flex-col gap-1">
        {kbs.map((kb) => (
          <Button
            key={kb.slug}
            variant={kb.slug === selectedSlug ? "secondary" : "ghost"}
            className="h-auto w-full flex-col items-start gap-0.5 whitespace-normal px-3 py-2 text-left"
            onClick={() => onSelect(kb.slug)}
          >
            <span className="w-full truncate text-sm font-medium">{kb.name}</span>
            {kb.description && (
              <span className="w-full truncate text-xs font-normal text-muted-foreground">{kb.description}</span>
            )}
          </Button>
        ))}
        {kbs.length === 0 && <p className="px-1 text-sm text-muted-foreground">No knowledge bases yet.</p>}
      </div>
    </aside>
  );
}
