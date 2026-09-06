import { BACKEND_URL } from "@/lib/api";

export function ProductImageGrid({ names }: { names: string[] }) {
  if (names.length === 0) return null;

  return (
    <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-3">
      {names.map((name) => (
        // eslint-disable-next-line @next/next/no-img-element -- backend-served, dynamic filenames
        <img
          key={name}
          src={`${BACKEND_URL}/products/images/${name}`}
          alt={name}
          className="aspect-square w-full rounded-md border object-cover"
        />
      ))}
    </div>
  );
}

/** Strips `[Image: filename]` and `[ID: product_id]` tags out of message
 * text, e.g. from product recommendation results — both exist for the
 * backend/agent contract (the ID tag lets the agent look up a specific
 * product in a follow-up), not for the user to read. The system prompt
 * already tells the model never to echo an ID tag, but strip it here too
 * as a defensive backstop. Returns the cleaned text and extracted image
 * names. */
export function extractProductImages(content: string): { text: string; names: string[] } {
  const names: string[] = [];
  const text = content
    .replace(/\[Image:\s*([^\]]+)\]/g, (_match, name: string) => {
      const trimmed = name.trim();
      if (!names.includes(trimmed)) names.push(trimmed);
      return "";
    })
    .replace(/\[ID:\s*[^\]]+\]/g, "");
  return { text: text.trim(), names };
}
