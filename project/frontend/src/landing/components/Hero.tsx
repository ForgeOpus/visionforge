import { motion } from "framer-motion";
import { TextGenerateEffect } from "./aceternity/TextGenerateEffect";
import { FlipWords } from "./aceternity/FlipWords";
import { BackgroundBeams } from "./aceternity/BackgroundBeams";
import { SparklesCore } from "./aceternity/Sparkles";
import { Button } from "@/components/ui/button";
import { ArrowRight, Play } from "@phosphor-icons/react";
import { useNavigate } from "react-router-dom";

export default function Hero() {
  const navigate = useNavigate();
  const words = ["PyTorch", "TensorFlow"];

  return (
    <div className="min-h-screen w-full bg-slate-950 relative flex flex-col items-center justify-center overflow-hidden">
      <BackgroundBeams className="absolute top-0 left-0" />

      {/* Grid pattern overlay */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#0a1628_1px,transparent_1px),linear-gradient(to_bottom,#0a1628_1px,transparent_1px)] bg-[size:4rem_4rem] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_50%,#000_70%,transparent_100%)] opacity-20" />

      {/* Hero Content */}
      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-32 pb-20">
        {/* Logo */}
        <motion.div
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
          className="flex justify-center mb-12"
        >
          <div className="relative">
            <div className="absolute inset-0 blur-3xl bg-cyan-500/30 rounded-full"></div>
            <img
              src="/vision_logo.png"
              alt="VisionForge Logo"
              className="relative w-40 h-40 sm:w-56 sm:h-56 object-contain drop-shadow-[0_0_80px_rgba(0,229,255,0.6)] hover:drop-shadow-[0_0_100px_rgba(0,229,255,0.8)] transition-all duration-300"
            />
          </div>
        </motion.div>

        {/* Main Headline */}
        <div className="text-center mb-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.3 }}
          >
            <h1 className="text-5xl sm:text-6xl md:text-7xl lg:text-8xl font-extrabold text-white mb-6 leading-tight">
              <span className="block mb-2">Design Neural Networks</span>
              <span className="block bg-gradient-to-r from-cyan-400 via-cyan-300 to-blue-400 bg-clip-text text-transparent">
                Without Writing Code
              </span>
            </h1>
          </motion.div>
        </div>

        {/* Subheadline with FlipWords */}
        <div className="text-center mb-12 max-w-4xl mx-auto">
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.8, delay: 0.6 }}
            className="text-xl sm:text-2xl md:text-3xl text-gray-300 flex flex-wrap justify-center items-center gap-3 mb-6"
          >
            <span>Build production-ready models for</span>
            <FlipWords words={words} className="text-cyan-400 font-bold" />
          </motion.div>

          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.8, delay: 0.8 }}
            className="text-lg sm:text-xl text-gray-400 max-w-2xl mx-auto leading-relaxed"
          >
            Visually design, validate, and export clean PyTorch or TensorFlow code.
            <span className="block mt-2 text-cyan-400 font-semibold">From idea to deployment in minutes.</span>
          </motion.p>
        </div>

        {/* CTA Buttons */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 1 }}
          className="flex flex-col sm:flex-row gap-4 justify-center items-center mb-20"
        >
          <div className="relative group">
            <div className="absolute -inset-1 bg-gradient-to-r from-cyan-500 to-blue-600 rounded-lg blur opacity-75 group-hover:opacity-100 transition duration-300"></div>
            <Button
              size="lg"
              onClick={() => navigate("/project")}
              className="relative bg-gradient-to-r from-cyan-500 to-cyan-600 hover:from-cyan-600 hover:to-blue-600 text-white border-0 px-10 py-7 text-lg font-bold shadow-2xl transition-all duration-300 transform hover:scale-105"
            >
              Start Building Free
              <ArrowRight size={24} className="ml-3" weight="bold" />
            </Button>
          </div>

          <Button
            size="lg"
            variant="outline"
            className="border-2 border-cyan-500/50 text-cyan-400 hover:bg-cyan-500/10 hover:border-cyan-400 bg-slate-900/50 backdrop-blur-sm px-10 py-7 text-lg font-semibold transition-all duration-300 hover:scale-105"
          >
            <Play size={24} className="mr-3" weight="fill" />
            Watch Demo
          </Button>
        </motion.div>

        {/* Preview Image/Screenshot */}
        <motion.div
          initial={{ opacity: 0, y: 60 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, delay: 1.2 }}
          className="relative max-w-6xl mx-auto"
        >
          {/* Glow effects */}
          <div className="absolute -inset-4 bg-gradient-to-r from-cyan-500/20 via-blue-500/20 to-purple-500/20 rounded-3xl blur-3xl"></div>

          <div className="relative rounded-2xl overflow-hidden border-2 border-cyan-500/40 shadow-[0_0_80px_rgba(0,229,255,0.3)] backdrop-blur-sm bg-slate-900/40">
            {/* Placeholder for canvas screenshot */}
            <div className="aspect-video bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex items-center justify-center relative overflow-hidden">
              {/* Grid background */}
              <div className="absolute inset-0 bg-[linear-gradient(to_right,#1e293b_1px,transparent_1px),linear-gradient(to_bottom,#1e293b_1px,transparent_1px)] bg-[size:2rem_2rem] opacity-20"></div>

              {/* Mock nodes */}
              <div className="relative z-10 flex items-center justify-center gap-8 p-8">
                <div className="flex flex-col gap-4">
                  {[1, 2, 3].map((i) => (
                    <motion.div
                      key={i}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: 1.5 + i * 0.1 }}
                      className="w-32 h-20 rounded-lg bg-gradient-to-br from-cyan-500/20 to-cyan-600/20 border-2 border-cyan-500/40 flex items-center justify-center backdrop-blur-sm"
                    >
                      <div className="w-3 h-3 bg-cyan-400 rounded-full shadow-[0_0_20px_rgba(0,229,255,0.8)]"></div>
                    </motion.div>
                  ))}
                </div>
                <div className="text-cyan-400/50 text-6xl font-thin">→</div>
                <motion.div
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 1.8 }}
                  className="w-40 h-32 rounded-lg bg-gradient-to-br from-blue-500/30 to-purple-500/30 border-2 border-blue-400/40 flex items-center justify-center backdrop-blur-sm"
                >
                  <div className="w-4 h-4 bg-blue-400 rounded-full shadow-[0_0_20px_rgba(59,130,246,0.8)]"></div>
                </motion.div>
              </div>

              {/* Overlay text */}
              <div className="absolute bottom-6 left-1/2 transform -translate-x-1/2 text-center">
                <div className="text-cyan-400/60 text-sm font-semibold backdrop-blur-sm bg-slate-900/50 px-6 py-2 rounded-full border border-cyan-500/30">
                  Visual Model Builder
                </div>
              </div>
            </div>
          </div>

          {/* Decorative elements */}
          <div className="absolute -top-8 -right-8 w-32 h-32 bg-cyan-500/20 rounded-full blur-3xl animate-pulse"></div>
          <div className="absolute -bottom-8 -left-8 w-40 h-40 bg-blue-500/20 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }}></div>
        </motion.div>

        {/* Framework badges and trust indicators */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.8, delay: 1.8 }}
          className="mt-20 space-y-6"
        >
          {/* Framework support */}
          <div className="flex justify-center items-center gap-6">
            <div className="text-gray-500 text-sm font-medium">Supports:</div>
            <div className="flex gap-4">
              <div className="group relative px-6 py-3 rounded-full bg-gradient-to-r from-slate-800/80 to-slate-900/80 border border-cyan-500/30 backdrop-blur-sm hover:border-cyan-400/50 transition-all duration-300">
                <div className="absolute inset-0 bg-gradient-to-r from-orange-500/10 to-red-500/10 rounded-full opacity-0 group-hover:opacity-100 transition-opacity"></div>
                <span className="relative text-cyan-400 font-bold text-sm">PyTorch</span>
              </div>
              <div className="group relative px-6 py-3 rounded-full bg-gradient-to-r from-slate-800/80 to-slate-900/80 border border-cyan-500/30 backdrop-blur-sm hover:border-cyan-400/50 transition-all duration-300">
                <div className="absolute inset-0 bg-gradient-to-r from-orange-500/10 to-orange-600/10 rounded-full opacity-0 group-hover:opacity-100 transition-opacity"></div>
                <span className="relative text-cyan-400 font-bold text-sm">TensorFlow</span>
              </div>
            </div>
          </div>

          {/* Trust indicators */}
          <div className="flex justify-center items-center gap-8 text-sm text-gray-500">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-green-400 rounded-full"></div>
              <span>Free & Open Source</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-cyan-400 rounded-full"></div>
              <span>No Credit Card Required</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-blue-400 rounded-full"></div>
              <span>Export Unlimited Projects</span>
            </div>
          </div>
        </motion.div>
      </div>

      {/* Scroll indicator */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 1, delay: 2 }}
        className="absolute bottom-10 left-1/2 transform -translate-x-1/2"
      >
        <motion.div
          animate={{ y: [0, 12, 0] }}
          transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
          className="w-6 h-10 border-2 border-cyan-500/50 rounded-full p-1 backdrop-blur-sm bg-slate-900/30"
        >
          <motion.div
            animate={{ y: [0, 16, 0], opacity: [1, 0, 1] }}
            transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
            className="w-1.5 h-1.5 bg-cyan-400 rounded-full mx-auto"
          ></motion.div>
        </motion.div>
      </motion.div>
    </div>
  );
}
