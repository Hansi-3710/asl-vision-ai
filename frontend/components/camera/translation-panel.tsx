"use client";

import { AnimatePresence, motion } from "framer-motion";
import { MessageSquareText, RotateCcw } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { formatConfidence } from "@/lib/utils";
import type { ConversationEntry, WordPrediction } from "@/types/streaming";

interface TranslationPanelProps {
  transcript: string;
  words: WordPrediction[];
  history: ConversationEntry[];
  onStartNewConversation: () => void;
}

export function TranslationPanel({ transcript, words, history, onStartNewConversation }: TranslationPanelProps) {
  return (
    <Card>
      <CardHeader className="flex-row items-center justify-between space-y-0">
        <CardTitle className="flex items-center gap-2 text-base">
          <MessageSquareText className="h-4 w-4 text-signal" />
          Live Translation
        </CardTitle>
        <Button variant="ghost" size="sm" onClick={onStartNewConversation}>
          <RotateCcw className="h-3.5 w-3.5" />
          New conversation
        </Button>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="min-h-[4.5rem] rounded-xl border border-glass-border bg-background-elevated/60 p-4">
          {transcript ? (
            <p className="font-display text-lg leading-snug text-ink">{transcript}</p>
          ) : (
            <p className="text-sm text-ink-faint">Start signing -- your translated sentence will appear here.</p>
          )}
        </div>

        {words.length > 0 && (
          <div className="flex flex-wrap gap-1.5">
            {words.map((w, i) => (
              <Badge key={`${w.word}-${i}`} variant="neutral" className="font-mono">
                {w.word}
                <span className="opacity-60">{formatConfidence(w.confidence)}</span>
              </Badge>
            ))}
          </div>
        )}

        {history.length > 0 && (
          <div className="space-y-2 border-t border-glass-border pt-4">
            <p className="text-xs font-medium uppercase tracking-wide text-ink-faint">Earlier this session</p>
            <div className="max-h-48 space-y-2 overflow-y-auto scrollbar-none">
              <AnimatePresence initial={false}>
                {[...history].reverse().map((entry) => (
                  <motion.p
                    key={entry.id}
                    initial={{ opacity: 0, y: -6 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="text-sm text-ink-muted"
                  >
                    {entry.transcript}
                  </motion.p>
                ))}
              </AnimatePresence>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
