"use client";

import { motion } from "framer-motion";
import { Camera, Gauge, LineChart, ShieldCheck, Sparkles, Upload } from "lucide-react";
import { ViewfinderFrame } from "@/components/viewfinder-frame";

const FEATURES = [
  {
    icon: Camera,
    title: "Live webcam recognition",
    description:
      "Point your camera at your hand and get a prediction the moment you hold a letter steady -- no capture button, no page reloads.",
  },
  {
    icon: Upload,
    title: "Instant image upload",
    description:
      "Drag in a photo and get the predicted letter, confidence, and inference time back in under a second.",
  },
  {
    icon: Gauge,
    title: "Top-5 predictions",
    description:
      "See exactly what the model considered, not just its top guess -- useful for letters that look alike.",
  },
  {
    icon: LineChart,
    title: "Live dashboard",
    description:
      "Every prediction is logged: total volume, average confidence, latency, and which letters come up most.",
  },
  {
    icon: Sparkles,
    title: "Fine-tuned EfficientNetV2",
    description:
      "Built on ImageNet-pretrained weights and fine-tuned on the ASL alphabet dataset for strong accuracy without training from zero.",
  },
  {
    icon: ShieldCheck,
    title: "Your data, your call",
    description:
      "Webcam frames are analyzed in real time and never saved to disk. Uploaded photos are stored only so you can review them in history.",
  },
];

export function Features() {
  return (
    <section className="px-6 py-24">
      <div className="mx-auto max-w-6xl">
        <div className="mx-auto max-w-2xl text-center">
          <h2 className="font-display text-3xl font-semibold tracking-tight text-ink md:text-4xl">
            Everything you need to test it
          </h2>
          <p className="mt-4 text-ink-muted">
            From a single photo to a live feed, ASL Vision AI gives you a clear read on what the
            model sees and how confident it is.
          </p>
        </div>

        <div className="mt-16 grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {FEATURES.map((feature, i) => (
            <motion.div
              key={feature.title}
              initial={{ opacity: 0, y: 16 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-60px" }}
              transition={{ duration: 0.5, delay: (i % 3) * 0.08 }}
            >
              <ViewfinderFrame revealOnHover size={16} className="h-full rounded-2xl">
                <div className="glass-panel h-full rounded-2xl p-6 transition-colors duration-300 hover:bg-glass-hover">
                  <div className="mb-4 flex h-10 w-10 items-center justify-center rounded-lg bg-signal/10 border border-signal/20">
                    <feature.icon className="h-5 w-5 text-signal" />
                  </div>
                  <h3 className="font-display text-lg font-semibold text-ink">{feature.title}</h3>
                  <p className="mt-2 text-sm leading-relaxed text-ink-muted">{feature.description}</p>
                </div>
              </ViewfinderFrame>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
