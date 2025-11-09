import { motion } from "framer-motion";
import { BentoGrid, BentoGridItem } from "./aceternity/BentoGrid";
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
    {
      title: "Intuitive Visual Builder",
      description: "Design complex neural network architectures with our powerful drag-and-drop canvas. No coding required.",
      icon: <Cube size={36} className="text-cyan-400" weight="duotone" />,
      className: "md:col-span-1 lg:col-span-1",
      header: (
        <div className="flex flex-1 w-full h-full min-h-[10rem] rounded-xl bg-gradient-to-br from-slate-900/90 via-slate-800/90 to-slate-900/90 border-2 border-cyan-500/20 relative overflow-hidden group">
          <div className="absolute inset-0 bg-[linear-gradient(to_right,#1e293b_1px,transparent_1px),linear-gradient(to_bottom,#1e293b_1px,transparent_1px)] bg-[size:1.5rem_1.5rem] opacity-10"></div>
          <div className="relative p-6 flex items-center justify-center w-full">
            <div className="grid grid-cols-3 gap-3">
              {[1, 2, 3, 4, 5, 6].map((i) => (
                <motion.div
                  key={i}
                  whileHover={{ scale: 1.1, rotate: 5 }}
                  className="w-16 h-16 rounded-xl bg-gradient-to-br from-cyan-500/30 to-cyan-600/30 border-2 border-cyan-500/40 flex items-center justify-center backdrop-blur-sm group-hover:border-cyan-400/60 transition-all duration-300"
                >
                  <div className="w-3 h-3 bg-cyan-400 rounded-full shadow-[0_0_15px_rgba(0,229,255,0.6)]"></div>
                </motion.div>
              ))}
            </div>
          </div>
          <div className="absolute inset-0 bg-gradient-to-t from-cyan-500/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
        </div>
      ),
    },
    {
      title: "Multi-Framework Support",
      description: "Export to PyTorch or TensorFlow with a single click. Maintain flexibility in your ML workflow.",
      icon: <Lightning size={36} className="text-cyan-400" weight="duotone" />,
      className: "md:col-span-1 lg:col-span-1",
      header: (
        <div className="flex flex-1 w-full h-full min-h-[10rem] rounded-xl bg-gradient-to-br from-slate-900/90 via-slate-800/90 to-slate-900/90 border-2 border-cyan-500/20 items-center justify-center gap-3 p-6 relative overflow-hidden group">
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_50%,rgba(0,229,255,0.1),transparent_70%)] opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
          <div className="relative flex flex-col gap-3">
            <div className="px-5 py-2.5 rounded-lg bg-gradient-to-r from-orange-500/20 to-red-500/20 border-2 border-orange-500/40 text-orange-400 text-sm font-bold backdrop-blur-sm hover:scale-105 transition-transform">
              PyTorch
            </div>
            <div className="px-5 py-2.5 rounded-lg bg-gradient-to-r from-orange-500/20 to-orange-600/20 border-2 border-orange-500/40 text-orange-400 text-sm font-bold backdrop-blur-sm hover:scale-105 transition-transform">
              TensorFlow
            </div>
          </div>
        </div>
      ),
    },
    {
      title: "Clean, Production Code",
      description: "Generate well-structured, documented code with type hints and best practices built-in.",
      icon: <Code size={36} className="text-cyan-400" weight="duotone" />,
      className: "md:col-span-1 lg:col-span-1",
      header: (
        <div className="flex flex-1 w-full h-full min-h-[10rem] rounded-xl bg-gradient-to-br from-slate-900/90 via-slate-800/90 to-slate-900/90 border-2 border-cyan-500/20 p-6 relative overflow-hidden group">
          <div className="absolute inset-0 bg-gradient-to-br from-purple-500/5 to-cyan-500/5 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
          <div className="relative w-full font-mono text-xs text-cyan-400 space-y-1.5 overflow-hidden">
            <div className="text-purple-400 font-semibold">class <span className="text-cyan-300">CustomModel</span>:</div>
            <div className="text-gray-500 pl-4">def __init__(self):</div>
            <div className="text-gray-600 pl-8">super().__init__()</div>
            <div className="text-gray-600 pl-8">self.layers = [...] </div>
            <div className="mt-2 text-gray-700 pl-4"># Clean & documented</div>
          </div>
        </div>
      ),
    },
    {
      title: "Intelligent Validation",
      description: "Real-time architecture validation catches errors instantly. Get helpful suggestions before export.",
      icon: <CheckCircle size={36} className="text-cyan-400" weight="duotone" />,
      className: "md:col-span-1 lg:col-span-1",
      header: (
        <div className="flex flex-1 w-full h-full min-h-[10rem] rounded-xl bg-gradient-to-br from-slate-900/90 via-slate-800/90 to-slate-900/90 border-2 border-cyan-500/20 p-6 flex-col justify-center relative overflow-hidden group">
          <div className="absolute inset-0 bg-gradient-to-r from-green-500/5 to-cyan-500/5 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
          <div className="relative space-y-3">
            <div className="flex items-center gap-3 p-3 rounded-lg bg-green-500/10 border border-green-500/30 backdrop-blur-sm">
              <CheckCircle size={20} className="text-green-400 flex-shrink-0" weight="fill" />
              <span className="text-sm text-green-400 font-medium">Architecture validated successfully</span>
            </div>
            <div className="flex items-center gap-3 p-3 rounded-lg bg-yellow-500/10 border border-yellow-500/30 backdrop-blur-sm">
              <div className="w-2.5 h-2.5 bg-yellow-400 rounded-full flex-shrink-0"></div>
              <span className="text-sm text-yellow-400">2 optimization suggestions available</span>
            </div>
          </div>
        </div>
      ),
    },
    {
      title: "Automatic Shape Inference",
      description: "Tensor dimensions calculated automatically as you build. No manual dimension tracking needed.",
      icon: <ArrowsClockwise size={36} className="text-cyan-400" weight="duotone" />,
      className: "md:col-span-1 lg:col-span-1",
      header: (
        <div className="flex flex-1 w-full h-full min-h-[10rem] rounded-xl bg-gradient-to-br from-slate-900/90 via-slate-800/90 to-slate-900/90 border-2 border-cyan-500/20 items-center justify-center p-6 relative overflow-hidden group">
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_50%,rgba(0,229,255,0.1),transparent_70%)] opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
          <div className="relative text-center">
            <div className="text-xs text-gray-500 mb-2 font-mono">Input Shape</div>
            <div className="text-base text-cyan-400 font-mono font-semibold mb-4 px-3 py-1 bg-slate-800/60 rounded border border-cyan-500/30">[B, 3, 224, 224]</div>
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
            >
              <ArrowsClockwise size={24} className="text-cyan-400 mx-auto mb-4" />
            </motion.div>
            <div className="text-xs text-gray-500 mb-2 font-mono">Output Shape</div>
            <div className="text-base text-cyan-400 font-mono font-semibold px-3 py-1 bg-slate-800/60 rounded border border-cyan-500/30">[B, 1000]</div>
          </div>
        </div>
      ),
    },
    {
      title: "Save & Share Projects",
      description: "Export architectures as JSON. Import pre-built models or share with your team effortlessly.",
      icon: <Download size={36} className="text-cyan-400" weight="duotone" />,
      className: "md:col-span-1 lg:col-span-1",
      header: (
        <div className="flex flex-1 w-full h-full min-h-[10rem] rounded-xl bg-gradient-to-br from-slate-900/90 via-slate-800/90 to-slate-900/90 border-2 border-cyan-500/20 items-center justify-center relative overflow-hidden group">
          <motion.div
            whileHover={{ scale: 1.2, rotate: 10 }}
            transition={{ type: "spring", stiffness: 300 }}
            className="text-7xl"
          >
            📦
          </motion.div>
          <div className="absolute inset-0 bg-gradient-to-t from-cyan-500/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
        </div>
      ),
    },
    {
      title: "Complex Architectures",
      description: "Build multi-branch models with skip connections, residual blocks, merge, concatenate, and add operations. Handle any topology including ResNet, U-Net, and custom designs.",
      icon: <GitBranch size={36} className="text-cyan-400" weight="duotone" />,
      className: "md:col-span-1 lg:col-span-1",
      header: (
        <div className="flex flex-1 w-full h-full min-h-[10rem] rounded-xl bg-gradient-to-br from-slate-900/90 via-slate-800/90 to-slate-900/90 border-2 border-cyan-500/20 p-6 items-center justify-center relative overflow-hidden group">
          <div className="absolute inset-0 bg-[linear-gradient(45deg,transparent_48%,rgba(0,229,255,0.1)_49%,rgba(0,229,255,0.1)_51%,transparent_52%)] bg-[length:20px_20px] opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
          <div className="relative w-full h-full flex items-center justify-center">
            <svg className="w-full h-full max-w-md" viewBox="0 0 200 120">
              {/* Input layer */}
              <circle cx="20" cy="60" r="8" fill="#00E5FF" opacity="0.8" />
              <text x="20" y="85" fontSize="8" fill="#00E5FF" textAnchor="middle">Input</text>

              {/* First branch - top path */}
              <line x1="28" y1="60" x2="70" y2="30" stroke="#00E5FF" strokeWidth="2" opacity="0.6" />
              <rect x="70" y="20" width="30" height="20" rx="4" fill="#00E5FF" opacity="0.3" stroke="#00E5FF" strokeWidth="1.5" />
              <text x="85" y="33" fontSize="7" fill="#FFF" textAnchor="middle">Conv</text>

              {/* Second branch - bottom path */}
              <line x1="28" y1="60" x2="70" y2="90" stroke="#00E5FF" strokeWidth="2" opacity="0.6" />
              <rect x="70" y="80" width="30" height="20" rx="4" fill="#00E5FF" opacity="0.3" stroke="#00E5FF" strokeWidth="1.5" />
              <text x="85" y="93" fontSize="7" fill="#FFF" textAnchor="middle">Conv</text>

              {/* Skip connection - direct path */}
              <line x1="28" y1="60" x2="130" y2="60" stroke="#FF6B6B" strokeWidth="2" opacity="0.5" strokeDasharray="4,4" />
              <text x="75" y="55" fontSize="7" fill="#FF6B6B" textAnchor="middle">Skip</text>

              {/* Merge point */}
              <line x1="100" y1="30" x2="130" y2="60" stroke="#00E5FF" strokeWidth="2" opacity="0.6" />
              <line x1="100" y1="90" x2="130" y2="60" stroke="#00E5FF" strokeWidth="2" opacity="0.6" />
              <circle cx="130" cy="60" r="10" fill="#A78BFA" opacity="0.8" />
              <text x="130" y="64" fontSize="8" fill="#FFF" textAnchor="middle" fontWeight="bold">+</text>
              <text x="130" y="85" fontSize="8" fill="#A78BFA" textAnchor="middle">Merge</text>

              {/* Output */}
              <line x1="140" y1="60" x2="170" y2="60" stroke="#00E5FF" strokeWidth="2" opacity="0.6" />
              <circle cx="180" cy="60" r="8" fill="#3B82F6" opacity="0.8" />
              <text x="180" y="85" fontSize="8" fill="#3B82F6" textAnchor="middle">Output</text>
            </svg>
          </div>
        </div>
      ),
    },
    {
      title: "Custom Layer Support",
      description: "Extend with your own custom layer implementations. Define novel architectures with full flexibility for research and production needs. Import from your codebase or create inline.",
      icon: <Wrench size={36} className="text-cyan-400" weight="duotone" />,
      className: "md:col-span-1 lg:col-span-1",
      header: (
        <div className="flex flex-1 w-full h-full min-h-[10rem] rounded-xl bg-gradient-to-br from-slate-900/90 via-slate-800/90 to-slate-900/90 border-2 border-cyan-500/20 p-6 flex-col justify-center items-center relative overflow-hidden group">
          <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/10 to-purple-500/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
          <div className="relative space-y-3 w-full">
            <div className="flex items-center gap-2 p-2 rounded bg-slate-800/60 border border-cyan-500/30 backdrop-blur-sm group-hover:border-cyan-400/50 transition-all">
              <Cube size={20} className="text-cyan-400 flex-shrink-0" weight="duotone" />
              <span className="text-xs text-cyan-300 font-mono">Conv2d</span>
            </div>
            <div className="flex items-center gap-2 p-2 rounded bg-slate-800/60 border border-cyan-500/30 backdrop-blur-sm group-hover:border-cyan-400/50 transition-all">
              <Cube size={20} className="text-purple-400 flex-shrink-0" weight="duotone" />
              <span className="text-xs text-purple-300 font-mono">CustomAttention</span>
            </div>
            <div className="flex items-center gap-2 p-2 rounded bg-slate-800/60 border border-cyan-500/30 backdrop-blur-sm group-hover:border-cyan-400/50 transition-all">
              <Wrench size={20} className="text-orange-400 flex-shrink-0" weight="duotone" />
              <span className="text-xs text-orange-300 font-mono">MySpecialLayer</span>
            </div>
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

        {/* Bento Grid */}
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          viewport={{ once: true }}
        >
          <BentoGrid className="max-w-6xl mx-auto">
            {features.map((item, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: i * 0.1 }}
                viewport={{ once: true }}
              >
                <BentoGridItem
                  title={item.title}
                  description={item.description}
                  header={item.header}
                  icon={item.icon}
                  className={`${item.className} bg-slate-900/30 border-2 border-cyan-500/20 hover:border-cyan-500/50 hover:shadow-[0_0_40px_rgba(0,229,255,0.2)] transition-all duration-500 backdrop-blur-sm hover:bg-slate-900/50`}
                />
              </motion.div>
            ))}
          </BentoGrid>
        </motion.div>

      </div>
    </div>
  );
}
