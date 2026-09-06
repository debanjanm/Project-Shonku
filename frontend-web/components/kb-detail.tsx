"use client";

import { Trash2, Upload } from "lucide-react";
import { useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import type { IngestResult, Kb, KbDocument } from "@/lib/types";

const ACCEPTED_EXTENSIONS = ".md,.txt,.pdf,.docx";

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  return `${(bytes / 1024).toFixed(1)} KB`;
}

export function KbDetail({
  kb,
  documents,
  onUpload,
  onDeleteDocument,
  onIngest,
  ingesting,
  ingestResult,
  onDeleteKb,
}: {
  kb: Kb;
  documents: KbDocument[];
  onUpload: (files: FileList) => Promise<void>;
  onDeleteDocument: (filename: string) => Promise<void>;
  onIngest: () => Promise<void>;
  ingesting: boolean;
  ingestResult: IngestResult | null;
  onDeleteKb: () => Promise<void>;
}) {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [uploading, setUploading] = useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [deleting, setDeleting] = useState(false);

  async function handleFilesSelected(files: FileList | null) {
    if (!files || files.length === 0) return;
    setUploading(true);
    try {
      await onUpload(files);
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }

  async function handleDeleteKb() {
    setDeleting(true);
    try {
      await onDeleteKb();
    } finally {
      setDeleting(false);
      setDeleteDialogOpen(false);
    }
  }

  return (
    <div className="mx-auto flex w-full max-w-3xl flex-col gap-4 overflow-y-auto p-6">
      <div>
        <h1 className="text-lg font-semibold">{kb.name}</h1>
        {kb.description && <p className="text-sm text-muted-foreground">{kb.description}</p>}
        <p className="text-xs text-muted-foreground">slug: {kb.slug}</p>
      </div>

      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle className="text-sm">Documents</CardTitle>
          <div className="flex items-center gap-2">
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept={ACCEPTED_EXTENSIONS}
              className="hidden"
              onChange={(e) => handleFilesSelected(e.target.files)}
            />
            <Button size="sm" variant="outline" onClick={() => fileInputRef.current?.click()} disabled={uploading}>
              <Upload className="h-4 w-4" />
              {uploading ? "Uploading..." : "Upload"}
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          {documents.length === 0 ? (
            <p className="text-sm text-muted-foreground">
              No documents yet. Upload {ACCEPTED_EXTENSIONS} files, then ingest.
            </p>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>File</TableHead>
                  <TableHead className="w-24">Size</TableHead>
                  <TableHead className="w-10" />
                </TableRow>
              </TableHeader>
              <TableBody>
                {documents.map((doc) => (
                  <TableRow key={doc.filename}>
                    <TableCell className="font-mono text-sm">{doc.filename}</TableCell>
                    <TableCell className="text-sm text-muted-foreground">{formatSize(doc.size)}</TableCell>
                    <TableCell>
                      <Button size="icon" variant="ghost" onClick={() => onDeleteDocument(doc.filename)}>
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Ingest</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col gap-3">
          <Button onClick={onIngest} disabled={ingesting || documents.length === 0} className="w-fit">
            {ingesting ? "Ingesting..." : "Ingest now"}
          </Button>
          {ingestResult && (
            <p className="text-sm text-muted-foreground">
              new={ingestResult.new} changed={ingestResult.changed} removed={ingestResult.removed}{" "}
              unchanged={ingestResult.unchanged} → {ingestResult.total_chunks} chunks total
            </p>
          )}
        </CardContent>
      </Card>

      <Card className="border-destructive/30">
        <CardHeader>
          <CardTitle className="text-sm text-destructive">Danger zone</CardTitle>
        </CardHeader>
        <CardContent>
          <Dialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
            <DialogTrigger asChild>
              <Button variant="destructive" size="sm">
                Delete this Knowledge Base
              </Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Delete &quot;{kb.name}&quot;?</DialogTitle>
              </DialogHeader>
              <p className="text-sm text-muted-foreground">
                This permanently deletes all {documents.length} document(s) and the search index for this
                knowledge base. This can&apos;t be undone.
              </p>
              <DialogFooter>
                <Button variant="outline" onClick={() => setDeleteDialogOpen(false)}>
                  Cancel
                </Button>
                <Button variant="destructive" onClick={handleDeleteKb} disabled={deleting}>
                  {deleting ? "Deleting..." : "Delete"}
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>
        </CardContent>
      </Card>
    </div>
  );
}
