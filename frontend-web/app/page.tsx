import { Suspense } from "react";

import { ChatApp } from "@/app/chat-app";

export default function Home() {
  return (
    <Suspense fallback={null}>
      <ChatApp />
    </Suspense>
  );
}
