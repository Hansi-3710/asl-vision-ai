"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { toast } from "sonner";
import { RotateCcw } from "lucide-react";
import { Navbar } from "@/components/navbar";
import { Footer } from "@/components/footer";
import { Dropzone } from "@/components/upload/dropzone";
import { UploadResult } from "@/components/upload/upload-result";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { predictUpload, describeApiError } from "@/lib/api";
import type { Prediction } from "@/types/prediction";

export default function UploadPage() {
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleFileSelected = async (file: File) => {
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    setPrediction(null);
    setIsLoading(true);

    try {
      const result = await predictUpload(file);
      setPrediction(result);
      toast.success(`Recognized "${result.predicted_class.toUpperCase()}"`, {
        description: `${(result.confidence * 100).toFixed(1)}% confidence`,
      });
    } catch (error) {
      toast.error("Prediction failed", { description: describeApiError(error) });
      setPreviewUrl(null);
    } finally {
      setIsLoading(false);
    }
  };

  const reset = () => {
    setPreviewUrl(null);
    setPrediction(null);
  };

  return (
    <main className="min-h-screen">
      <Navbar />

      <section className="mx-auto max-w-4xl px-6 py-12">
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="mb-8 flex items-center justify-between"
        >
          <div>
            <h1 className="font-display text-3xl font-semibold tracking-tight text-ink">Upload Image</h1>
            <p className="mt-2 text-ink-muted">Drop in a photo of a hand sign to get an instant prediction.</p>
          </div>
          {(previewUrl || prediction) && (
            <Button variant="ghost" size="sm" onClick={reset}>
              <RotateCcw className="h-4 w-4" />
              Reset
            </Button>
          )}
        </motion.div>

        {!previewUrl && <Dropzone onFileSelected={handleFileSelected} disabled={isLoading} />}

        {previewUrl && isLoading && (
          <div className="grid gap-6 md:grid-cols-2">
            <Skeleton className="aspect-square w-full rounded-2xl" />
            <div className="space-y-6">
              <Skeleton className="h-32 w-full rounded-2xl" />
              <Skeleton className="h-48 w-full rounded-2xl" />
              <Skeleton className="h-16 w-full rounded-2xl" />
            </div>
          </div>
        )}

        {previewUrl && prediction && !isLoading && (
          <UploadResult imagePreviewUrl={previewUrl} prediction={prediction} />
        )}
      </section>

      <Footer />
    </main>
  );
}
