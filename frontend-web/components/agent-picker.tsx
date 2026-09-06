import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import type { Agent } from "@/lib/types";

export function AgentPicker({
  agents,
  value,
  onChange,
}: {
  agents: Agent[];
  value: string;
  onChange: (agentId: string) => void;
}) {
  const selected = agents.find((a) => a.id === value);

  return (
    <div className="space-y-1.5">
      <p className="text-xs font-medium text-muted-foreground">Agent</p>
      <Select value={value} onValueChange={onChange}>
        <SelectTrigger className="w-full">
          <SelectValue placeholder="Choose an agent" />
        </SelectTrigger>
        <SelectContent>
          {agents.map((agent) => (
            <SelectItem key={agent.id} value={agent.id}>
              {agent.name}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      {selected && <p className="text-xs text-muted-foreground">{selected.description}</p>}
    </div>
  );
}
