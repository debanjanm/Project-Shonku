"use client";

import { useCallback, useEffect, useState } from "react";

import { KbList } from "@/components/kb-list";
import { KbDetail } from "@/components/kb-detail";
import {
  createKb,
  deleteKb,
  deleteKbDocument,
  getKbs,
  ingestKb,
  listKbDocuments,
  uploadKbDocument,
} from "@/lib/api";
import type { IngestResult, Kb, KbDocument } from "@/lib/types";

export default function KbsPage() {
  const [kbs, setKbs] = useState<Kb[]>([]);
  const [selectedSlug, setSelectedSlug] = useState<string | null>(null);
  const [documents, setDocuments] = useState<KbDocument[]>([]);
  const [ingesting, setIngesting] = useState(false);
  const [ingestResult, setIngestResult] = useState<IngestResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refreshKbs = useCallback(async () => {
    try {
      const loaded = await getKbs();
      setKbs(loaded);
      return loaded;
    } catch {
      setError("Couldn't reach the backend. Is it running on :8000?");
      return [];
    }
  }, []);

  useEffect(() => {
    async function load() {
      await refreshKbs();
    }
    load();
  }, [refreshKbs]);

  const refreshDocuments = useCallback(async (slug: string) => {
    try {
      setDocuments(await listKbDocuments(slug));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load documents");
    }
  }, []);

  function handleSelect(slug: string) {
    setSelectedSlug(slug);
    setIngestResult(null);
    setError(null);
    refreshDocuments(slug);
  }

  async function handleCreate(name: string, description: string) {
    const kb = await createKb(name, description);
    await refreshKbs();
    handleSelect(kb.slug);
  }

  async function handleUpload(files: FileList) {
    if (!selectedSlug) return;
    setError(null);
    for (const file of Array.from(files)) {
      try {
        await uploadKbDocument(selectedSlug, file);
      } catch (e) {
        setError(e instanceof Error ? e.message : `Failed to upload ${file.name}`);
      }
    }
    await refreshDocuments(selectedSlug);
  }

  async function handleDeleteDocument(filename: string) {
    if (!selectedSlug) return;
    try {
      await deleteKbDocument(selectedSlug, filename);
      await refreshDocuments(selectedSlug);
    } catch (e) {
      setError(e instanceof Error ? e.message : `Failed to delete ${filename}`);
    }
  }

  async function handleIngest() {
    if (!selectedSlug) return;
    setIngesting(true);
    setError(null);
    try {
      setIngestResult(await ingestKb(selectedSlug));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Ingest failed");
    } finally {
      setIngesting(false);
    }
  }

  async function handleDeleteKb() {
    if (!selectedSlug) return;
    try {
      await deleteKb(selectedSlug);
      setSelectedSlug(null);
      setDocuments([]);
      setIngestResult(null);
      await refreshKbs();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to delete knowledge base");
    }
  }

  const selectedKb = kbs.find((kb) => kb.slug === selectedSlug) ?? null;

  return (
    <div className="flex h-full">
      <KbList kbs={kbs} selectedSlug={selectedSlug} onSelect={handleSelect} onCreate={handleCreate} />
      <main className="min-w-0 flex-1 overflow-y-auto">
        {error && (
          <p className="mx-6 mt-4 max-w-3xl rounded-md border border-destructive/30 bg-destructive/10 px-4 py-2 text-sm text-destructive">
            {error}
          </p>
        )}
        {selectedKb ? (
          <KbDetail
            kb={selectedKb}
            documents={documents}
            onUpload={handleUpload}
            onDeleteDocument={handleDeleteDocument}
            onIngest={handleIngest}
            ingesting={ingesting}
            ingestResult={ingestResult}
            onDeleteKb={handleDeleteKb}
          />
        ) : (
          <p className="p-6 text-sm text-muted-foreground">
            Pick a knowledge base from the sidebar, or create a new one.
          </p>
        )}
      </main>
    </div>
  );
}
