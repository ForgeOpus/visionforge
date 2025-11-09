import { motion } from "framer-motion";
import { SpotlightCard } from "./aceternity/SpotlightCard";
import { TiltCard } from "./aceternity/TiltCard";
import {
  Lightning,
  Code,
  CheckCircle,
  ArrowsClockwise,
  GitBranch,
  Download,
  Cube,
  Wrench
} from "@phosphor-icons/react";

export default function Features() {
  const features = [
    // Row 1 - Core Building Blocks
    {
      title: "Intuitive Visual Builder",
      description: "Design complex neural network architectures with our powerful drag-and-drop canvas. No coding required.",
      icon: <Cube size={36} className="text-cyan-400" weight="duotone" />,
      visual: (
        <div className="mb-6 p-4 bg-slate-800/40 rounded-xl border border-cyan-500/20 h-32 flex items-center justify-center">
          <div className="grid grid-cols-3 gap-2">
            {[1, 2, 3, 4, 5, 6].map((i) => (
              <motion.div
                key={i}
                whileHover={{ scale: 1.1, rotate: 5 }}
                className="w-10 h-10 rounded-lg bg-gradient-to-br from-cyan-500/30 to-cyan-600/30 border border-cyan-500/40 flex items-center justify-center"
              >
                <div className="w-2 h-2 bg-cyan-400 rounded-full"></div>
              </motion.div>
            ))}
          </div>
        </div>
      ),
    },
    {
      title: "Automatic Shape Inference",
      description: "Tensor dimensions calculated automatically as you build. No manual dimension tracking needed.",
      icon: <ArrowsClockwise size={36} className="text-cyan-400" weight="duotone" />,
      visual: (
        <div className="mb-6 text-center p-4 bg-slate-800/40 rounded-xl border border-cyan-500/20 h-32 flex flex-col items-center justify-center">
          <div className="text-xs text-gray-500 font-mono">Input</div>
          <div className="text-sm text-cyan-400 font-mono font-semibold mb-2">[B, 3, 224, 224]</div>
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
            className="inline-block"
          >
            <ArrowsClockwise size={20} className="text-cyan-400" />
          </motion.div>
          <div className="text-sm text-cyan-400 font-mono font-semibold mt-2">[B, 1000]</div>
          <div className="text-xs text-gray-500 font-mono">Output</div>
        </div>
      ),
    },
    {
      title: "Intelligent Validation",
      description: "Real-time architecture validation catches errors instantly. Get helpful suggestions before export.",
      icon: <CheckCircle size={36} className="text-cyan-400" weight="duotone" />,
      visual: (
        <div className="mb-6 h-32 flex flex-col items-center justify-center space-y-2">
          <div className="flex items-center gap-2 p-2 rounded-lg bg-green-500/10 border border-green-500/30 w-full">
            <CheckCircle size={16} className="text-green-400" weight="fill" />
            <span className="text-xs text-green-400 font-medium">Architecture valid</span>
          </div>
          <div className="flex items-center gap-2 p-2 rounded-lg bg-yellow-500/10 border border-yellow-500/30 w-full">
            <div className="w-2 h-2 bg-yellow-400 rounded-full"></div>
            <span className="text-xs text-yellow-400">2 suggestions</span>
          </div>
        </div>
      ),
    },
    {
      title: "Save & Share Projects",
      description: "Export architectures as JSON. Import pre-built models or share with your team effortlessly.",
      icon: <Download size={36} className="text-cyan-400" weight="duotone" />,
      visual: (
        <div className="mb-6 flex items-center justify-center bg-slate-800/40 rounded-xl border border-cyan-500/20 h-32">
          <motion.div
            whileHover={{ scale: 1.1, rotate: 5 }}
            transition={{ type: "spring", stiffness: 300 }}
            className="text-5xl"
          >
            📦
          </motion.div>
        </div>
      ),
    },

    // Row 2 - Advanced Features
    {
      title: "Multi-Framework Support",
      description: "Export to PyTorch or TensorFlow with a single click. Maintain flexibility in your ML workflow.",
      icon: <Lightning size={36} className="text-cyan-400" weight="duotone" />,
      visual: (
        <div className="mb-6 flex flex-col gap-3 p-4 bg-slate-800/40 rounded-xl border border-cyan-500/20 h-32 justify-center">
          <div className="px-4 py-2 rounded-lg bg-gradient-to-r from-orange-500/20 to-red-500/20 border border-orange-500/40 text-orange-400 text-sm font-bold text-center">
            PyTorch
          </div>
          <div className="px-4 py-2 rounded-lg bg-gradient-to-r from-orange-500/20 to-orange-600/20 border border-orange-500/40 text-orange-400 text-sm font-bold text-center">
            TensorFlow
          </div>
        </div>
      ),
    },
    {
      title: "Clean, Production Code",
      description: "Generate well-structured, documented code with type hints and best practices built-in.",
      icon: <Code size={36} className="text-cyan-400" weight="duotone" />,
      visual: (
        <div className="mb-6 p-4 bg-slate-800/40 rounded-xl border border-cyan-500/20 font-mono text-xs text-cyan-400 h-32 flex flex-col justify-center space-y-1">
          <div className="text-purple-400">class <span className="text-cyan-300">Model</span>:</div>
          <div className="text-gray-500 pl-4">def __init__(self):</div>
          <div className="text-gray-600 pl-8">self.layers = [...]</div>
          <div className="text-gray-700 pl-4"># Type hints included</div>
        </div>
      ),
    },
    {
      title: "Complex Architectures",
      description: "Build multi-branch models with skip connections, residual blocks, and merge operations. ResNet, U-Net, and beyond.",
      icon: <GitBranch size={36} className="text-cyan-400" weight="duotone" />,
      visual: (
        <div className="mb-6 p-4 bg-slate-800/40 rounded-xl border border-cyan-500/20 h-32 flex items-center justify-center">
          <svg className="w-full h-20" viewBox="0 0 200 80">
            <circle cx="20" cy="40" r="6" fill="#00E5FF" opacity="0.8" />
            <line x1="26" y1="40" x2="60" y2="20" stroke="#00E5FF" strokeWidth="2" opacity="0.6" />
            <rect x="60" y="12" width="24" height="16" rx="3" fill="#00E5FF" opacity="0.3" stroke="#00E5FF" strokeWidth="1.5" />
            <line x1="26" y1="40" x2="60" y2="60" stroke="#00E5FF" strokeWidth="2" opacity="0.6" />
            <rect x="60" y="52" width="24" height="16" rx="3" fill="#00E5FF" opacity="0.3" stroke="#00E5FF" strokeWidth="1.5" />
            <line x1="26" y1="40" x2="120" y2="40" stroke="#FF6B6B" strokeWidth="2" opacity="0.5" strokeDasharray="3,3" />
            <line x1="84" y1="20" x2="120" y2="40" stroke="#00E5FF" strokeWidth="2" opacity="0.6" />
            <line x1="84" y1="60" x2="120" y2="40" stroke="#00E5FF" strokeWidth="2" opacity="0.6" />
            <circle cx="120" cy="40" r="8" fill="#A78BFA" opacity="0.8" />
            <line x1="128" y1="40" x2="160" y2="40" stroke="#00E5FF" strokeWidth="2" opacity="0.6" />
            <circle cx="170" cy="40" r="6" fill="#3B82F6" opacity="0.8" />
          </svg>
        </div>
      ),
    },
    {
      title: "Custom Layer Support",
      description: "Extend with your own custom implementations. Full flexibility for research and production needs.",
      icon: <Wrench size={36} className="text-cyan-400" weight="duotone" />,
      visual: (
        <div className="mb-6 h-32 flex flex-col justify-center space-y-2">
          <div className="flex items-center gap-2 p-2 rounded bg-slate-800/60 border border-cyan-500/30">
            <Cube size={18} className="text-cyan-400" weight="duotone" />
            <span className="text-xs text-cyan-300 font-mono">Conv2d</span>
          </div>
          <div className="flex items-center gap-2 p-2 rounded bg-slate-800/60 border border-purple-500/30">
            <Cube size={18} className="text-purple-400" weight="duotone" />
            <span className="text-xs text-purple-300 font-mono">CustomAttention</span>
          </div>
          <div className="flex items-center gap-2 p-2 rounded bg-slate-800/60 border border-orange-500/30">
            <Wrench size={18} className="text-orange-400" weight="duotone" />
            <span className="text-xs text-orange-300 font-mono">MySpecialLayer</span>
          </div>
        </div>
      ),
    },
  ];

  return (
    <div className="w-full bg-slate-950 py-24 px-4 relative overflow-hidden">
      {/* Background gradient blobs */}
      <div className="absolute top-0 left-1/4 w-96 h-96 bg-cyan-500/5 rounded-full blur-3xl"></div>
      <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-blue-500/5 rounded-full blur-3xl"></div>

      <div className="max-w-7xl mx-auto relative z-10">
        {/* Section Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          viewport={{ once: true }}
          className="text-center mb-20"
        >
          <div className="inline-block mb-4 px-4 py-1.5 rounded-full bg-cyan-500/10 border border-cyan-500/30 backdrop-blur-sm">
            <span className="text-cyan-400 text-sm font-semibold">Features</span>
          </div>
          <h2 className="text-4xl md:text-5xl lg:text-6xl font-extrabold text-white mb-6 leading-tight">
            <span className="block mb-2">Powerful Tools for</span>
            <span className="block bg-gradient-to-r from-cyan-400 via-cyan-300 to-blue-400 bg-clip-text text-transparent">
              Modern AI Development
            </span>
          </h2>
          <p className="text-xl text-gray-400 max-w-3xl mx-auto leading-relaxed">
            Everything you need to design, validate, and deploy neural networks—
            <span className="text-cyan-400 font-semibold"> from concept to production code</span>.
          </p>
        </motion.div>

        {/* Feature Grid with 3D Tilt Cards - Uniform 4-column layout */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 max-w-7xl mx-auto">
          {features.map((feature, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: (i % 4) * 0.1 }}
              viewport={{ once: true }}
            >
              <TiltCard className="h-full">
                <div className="flex flex-col h-full min-h-[320px]">
                  {feature.visual}
                  <div className="flex items-center gap-3 mb-3">
                    {feature.icon}
                    <h3 className="text-xl font-bold text-white">
                      {feature.title}
                    </h3>
                  </div>
                  <p className="text-gray-400 leading-relaxed text-sm flex-1">
                    {feature.description}
                  </p>
                </div>
              </TiltCard>
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  );
}
