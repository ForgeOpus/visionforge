import { motion } from "framer-motion";

export default function DemoVideo() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 60 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 1, delay: 1.2 }}
      className="relative max-w-6xl mx-auto"
    >
      {/* Glow effects */}
      <div className="absolute -inset-4 bg-gradient-to-r from-cyan-500/20 via-blue-500/20 to-purple-500/20 rounded-3xl blur-3xl"></div>

      <div className="relative rounded-2xl overflow-hidden border-2 border-cyan-500/40 shadow-[0_0_80px_rgba(0,229,255,0.3)] backdrop-blur-sm bg-slate-900/40">
        {/* Video Container */}
        <div className="aspect-video bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex items-center justify-center relative overflow-hidden">
          <video
            src="/demo-video.mov"
            autoPlay
            loop
            muted
            playsInline
            className="w-full h-full object-cover"
          />

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
  );
}
