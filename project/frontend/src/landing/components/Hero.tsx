import { motion } from "framer-motion";
import { TextGenerateEffect } from "./aceternity/TextGenerateEffect";
import { FlipWords } from "./aceternity/FlipWords";
import { BackgroundBeams } from "./aceternity/BackgroundBeams";
import { SparklesCore } from "./aceternity/Sparkles";
import { Button } from "@/components/ui/button";
import { Download, Play } from "@phosphor-icons/react";
import { useNavigate } from "react-router-dom";

export default function Hero() {
  const navigate = useNavigate();
  const words = ["PyTorch", "TensorFlow", "Vision Models"];

  return (
    <div className="min-h-screen w-full bg-slate-950 relative flex flex-col items-center justify-center overflow-hidden">
      <BackgroundBeams className="absolute top-0 left-0" />

      {/* Hero Content */}
      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-32 pb-20">
        {/* Logo */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="flex justify-center mb-8"
        >
          <img
            src="/vision_logo.png"
            alt="VisionForge Logo"
            className="w-32 h-32 sm:w-48 sm:h-48 object-contain drop-shadow-[0_0_50px_rgba(0,229,255,0.5)]"
          />
        </motion.div>

        {/* Main Headline */}
        <div className="text-center mb-6">
          <motion.h1
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 1, delay: 0.3 }}
            className="text-5xl sm:text-6xl md:text-7xl font-bold text-white mb-6"
          >
            Build AI Models <br />
            <span className="text-cyan-400">Visually</span>
          </motion.h1>
        </div>

        {/* Subheadline with FlipWords */}
        <div className="text-center mb-12 max-w-3xl mx-auto">
          <div className="text-xl sm:text-2xl md:text-3xl text-gray-300 flex flex-wrap justify-center items-center gap-2">
            <span>Design neural networks for</span>
            <FlipWords words={words} className="text-cyan-400 font-bold" />
          </div>
          <TextGenerateEffect
            words="Drag, drop, and export production-ready code in minutes."
            className="text-lg sm:text-xl text-gray-400 mt-4"
          />
        </div>

        {/* CTA Buttons */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.8 }}
          className="flex flex-col sm:flex-row gap-4 justify-center items-center mb-16"
        >
          <div className="relative">
            <SparklesCore
              id="cta-sparkles"
              background="transparent"
              minSize={0.4}
              maxSize={1.4}
              particleDensity={50}
              className="w-full h-full absolute inset-0"
              particleColor="#00E5FF"
            />
            <Button
              size="lg"
              onClick={() => navigate("/app")}
              className="relative z-10 bg-gradient-to-r from-cyan-500 to-cyan-600 hover:from-cyan-600 hover:to-cyan-700 text-white border-0 px-8 py-6 text-lg font-semibold shadow-[0_0_30px_rgba(0,229,255,0.3)] hover:shadow-[0_0_50px_rgba(0,229,255,0.5)] transition-all duration-300"
            >
              <Download size={24} className="mr-2" weight="bold" />
              Start Building Free
            </Button>
          </div>

          <Button
            size="lg"
            variant="outline"
            className="border-cyan-500/50 text-cyan-400 hover:bg-cyan-500/10 hover:border-cyan-400 px-8 py-6 text-lg font-semibold transition-all duration-300"
          >
            <Play size={24} className="mr-2" weight="fill" />
            Watch Demo
          </Button>
        </motion.div>

        {/* Preview Image/Screenshot */}
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, delay: 1 }}
          className="relative max-w-5xl mx-auto"
        >
          <div className="relative rounded-xl overflow-hidden border border-cyan-500/30 shadow-[0_0_50px_rgba(0,229,255,0.2)]">
            {/* Placeholder for canvas screenshot - you'll replace this with an actual screenshot */}
            <div className="aspect-video bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex items-center justify-center">
              <div className="text-center p-8">
                <div className="text-cyan-400 text-2xl font-bold mb-2">Canvas Preview</div>
                <div className="text-gray-500">
                  Screenshot of the drag-and-drop model builder will go here
                </div>
              </div>
            </div>

            {/* Glow effect overlay */}
            <div className="absolute inset-0 bg-gradient-to-t from-cyan-500/20 via-transparent to-transparent pointer-events-none"></div>
          </div>

          {/* Decorative elements */}
          <div className="absolute -top-4 -right-4 w-24 h-24 bg-cyan-500/20 rounded-full blur-3xl"></div>
          <div className="absolute -bottom-4 -left-4 w-32 h-32 bg-cyan-400/20 rounded-full blur-3xl"></div>
        </motion.div>

        {/* Framework badges */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.8, delay: 1.2 }}
          className="flex justify-center items-center gap-6 mt-16"
        >
          <div className="text-gray-500 text-sm">Supports:</div>
          <div className="flex gap-4">
            <div className="px-4 py-2 rounded-full bg-slate-800/50 border border-cyan-500/30 text-cyan-400 font-semibold text-sm">
              PyTorch
            </div>
            <div className="px-4 py-2 rounded-full bg-slate-800/50 border border-cyan-500/30 text-cyan-400 font-semibold text-sm">
              TensorFlow
            </div>
          </div>
        </motion.div>
      </div>

      {/* Scroll indicator */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 1, delay: 1.5 }}
        className="absolute bottom-8 left-1/2 transform -translate-x-1/2"
      >
        <motion.div
          animate={{ y: [0, 10, 0] }}
          transition={{ duration: 1.5, repeat: Infinity }}
          className="w-6 h-10 border-2 border-cyan-500/50 rounded-full p-1"
        >
          <div className="w-1.5 h-1.5 bg-cyan-400 rounded-full mx-auto"></div>
        </motion.div>
      </motion.div>
    </div>
  );
}
