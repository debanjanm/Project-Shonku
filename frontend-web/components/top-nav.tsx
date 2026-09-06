"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

import { cn } from "@/lib/utils";

const LINKS = [
  { href: "/", label: "Chat" },
  { href: "/kbs", label: "Knowledge Bases" },
];

export function TopNav() {
  const pathname = usePathname();

  return (
    <header className="flex h-12 shrink-0 items-center gap-6 border-b px-4">
      <span className="text-sm font-semibold tracking-tight">Project Shonku</span>
      <nav className="flex gap-1">
        {LINKS.map((link) => {
          const active = link.href === "/" ? pathname === "/" : pathname.startsWith(link.href);
          return (
            <Link
              key={link.href}
              href={link.href}
              className={cn(
                "rounded-md px-3 py-1.5 text-sm transition-colors",
                active ? "bg-muted font-medium text-foreground" : "text-muted-foreground hover:text-foreground",
              )}
            >
              {link.label}
            </Link>
          );
        })}
      </nav>
    </header>
  );
}
