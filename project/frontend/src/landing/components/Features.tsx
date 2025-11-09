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
      title: "Visual Drag & Drop Builder",
      description: "Design neural network architectures intuitively with our powerful canvas interface.",
      icon: <Cube size={32} className="text-cyan-400" weight="duotone" />,
      className: "md:col-span-2",
      header: (
        <div className="flex flex-1 w-full h-full min-h-[6rem] rounded-xl bg-gradient-to-br from-slate-900 to-slate-800 border border-cyan-500/20">
          <div className="p-4 flex items-center justify-center w-full">
            <div className="grid grid-cols-3 gap-2">
              {[1, 2, 3, 4, 5, 6].map((i) => (
                <div
                  key={i}
                  className="w-12 h-12 rounded-lg bg-gradient-to-br from-cyan-500/20 to-cyan-600/20 border border-cyan-500/30 flex items-center justify-center"
                >
                  <div className="w-2 h-2 bg-cyan-400 rounded-full"></div>
                </div>
              ))}
            </div>
          </div>
        </div>
      ),
    },
    {
      title: "Dual Framework Support",
      description: "Export to PyTorch or TensorFlow - your choice, your workflow.",
      icon: <Lightning size={32} className="text-cyan-400" weight="duotone" />,
      className: "md:col-span-1",
      header: (
        <div className="flex flex-1 w-full h-full min-h-[6rem] rounded-xl bg-gradient-to-br from-slate-900 to-slate-800 border border-cyan-500/20 items-center justify-center gap-4 p-4">
          <div className="px-3 py-1.5 rounded bg-orange-500/20 border border-orange-500/40 text-orange-400 text-xs font-bold">
            PyTorch
          </div>
          <div className="px-3 py-1.5 rounded bg-orange-500/20 border border-orange-500/40 text-orange-400 text-xs font-bold">
            TensorFlow
          </div>
        </div>
      ),
    },
    {
      title: "Production-Ready Code",
      description: "Generate clean, documented code ready for training and deployment.",
      icon: <Code size={32} className="text-cyan-400" weight="duotone" />,
      className: "md:col-span-1",
      header: (
        <div className="flex flex-1 w-full h-full min-h-[6rem] rounded-xl bg-gradient-to-br from-slate-900 to-slate-800 border border-cyan-500/20 p-4">
          <div className="w-full font-mono text-xs text-cyan-400 space-y-1">
            <div className="text-purple-400">class</div>
            <div className="text-cyan-300 pl-2">CustomModel:</div>
            <div className="text-gray-500 pl-4">def __init__</div>
            <div className="text-gray-600 pl-6">...</div>
          </div>
        </div>
      ),
    },
    {
      title: "Real-Time Validation",
      description: "Catch architecture errors before export with intelligent validation.",
      icon: <CheckCircle size={32} className="text-cyan-400" weight="duotone" />,
      className: "md:col-span-2",
      header: (
        <div className="flex flex-1 w-full h-full min-h-[6rem] rounded-xl bg-gradient-to-br from-slate-900 to-slate-800 border border-cyan-500/20 p-4 flex-col justify-center">
          <div className="flex items-center gap-2 mb-2">
            <CheckCircle size={16} className="text-green-400" weight="fill" />
            <span className="text-xs text-green-400">Architecture is valid</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 bg-yellow-400 rounded-full"></div>
            <span className="text-xs text-yellow-400">2 warnings found</span>
          </div>
        </div>
      ),
    },
    {
      title: "Smart Shape Inference",
      description: "Automatic tensor dimension calculation throughout your network.",
      icon: <ArrowsClockwise size={32} className="text-cyan-400" weight="duotone" />,
      className: "md:col-span-1",
      header: (
        <div className="flex flex-1 w-full h-full min-h-[6rem] rounded-xl bg-gradient-to-br from-slate-900 to-slate-800 border border-cyan-500/20 items-center justify-center p-4">
          <div className="text-center">
            <div className="text-xs text-gray-500 mb-1">Input</div>
            <div className="text-sm text-cyan-400 font-mono mb-3">[32, 3, 224, 224]</div>
            <ArrowsClockwise size={20} className="text-cyan-400 mx-auto mb-3" />
            <div className="text-xs text-gray-500 mb-1">Output</div>
            <div className="text-sm text-cyan-400 font-mono">[32, 1000]</div>
          </div>
        </div>
      ),
    },
    {
      title: "Import/Export Projects",
      description: "Save and share architectures as JSON for easy collaboration.",
      icon: <Download size={32} className="text-cyan-400" weight="duotone" />,
      className: "md:col-span-1",
      header: (
        <div className="flex flex-1 w-full h-full min-h-[6rem] rounded-xl bg-gradient-to-br from-slate-900 to-slate-800 border border-cyan-500/20 items-center justify-center">
          <div className="text-4xl">📦</div>
        </div>
      ),
    },
    {
      title: "Multi-Input Models",
      description: "Build complex architectures with concatenate and merge operations.",
      icon: <GitBranch size={32} className="text-cyan-400" weight="duotone" />,
      className: "md:col-span-1",
      header: (
        <div className="flex flex-1 w-full h-full min-h-[6rem] rounded-xl bg-gradient-to-br from-slate-900 to-slate-800 border border-cyan-500/20 p-4 items-center justify-center">
          <div className="relative w-full h-full">
            <div className="absolute top-2 left-4 w-12 h-8 rounded bg-cyan-500/20 border border-cyan-500/30"></div>
            <div className="absolute bottom-2 left-4 w-12 h-8 rounded bg-cyan-500/20 border border-cyan-500/30"></div>
            <div className="absolute top-1/2 right-4 -translate-y-1/2 w-12 h-8 rounded bg-cyan-500/30 border border-cyan-500/40"></div>
            <svg className="absolute inset-0 w-full h-full">
              <line x1="64" y1="18" x2="80" y2="32" stroke="#00E5FF" strokeWidth="2" opacity="0.3" />
              <line x1="64" y1="50" x2="80" y2="32" stroke="#00E5FF" strokeWidth="2" opacity="0.3" />
            </svg>
          </div>
        </div>
      ),
    },
    {
      title: "Custom Layers",
      description: "Extend the builder with your own custom layer implementations.",
      icon: <Wrench size={32} className="text-cyan-400" weight="duotone" />,
      className: "md:col-span-1",
      header: (
        <div className="flex flex-1 w-full h-full min-h-[6rem] rounded-xl bg-gradient-to-br from-slate-900 to-slate-800 border border-cyan-500/20 items-center justify-center">
          <Wrench size={48} className="text-cyan-400/30" weight="duotone" />
        </div>
      ),
    },
  ];

  return (
    <div className="w-full bg-slate-950 py-20 px-4">
      <div className="max-w-7xl mx-auto">
        {/* Section Header */}
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold text-white mb-4">
            Everything You Need to{" "}
            <span className="text-cyan-400">Build AI</span>
          </h2>
          <p className="text-xl text-gray-400 max-w-2xl mx-auto">
            Powerful features that make model building intuitive, fast, and production-ready.
          </p>
        </div>

        {/* Bento Grid */}
        <BentoGrid className="max-w-6xl mx-auto">
          {features.map((item, i) => (
            <BentoGridItem
              key={i}
              title={item.title}
              description={item.description}
              header={item.header}
              icon={item.icon}
              className={`${item.className} bg-slate-900/50 border-cyan-500/20 hover:border-cyan-500/40 hover:shadow-[0_0_30px_rgba(0,229,255,0.2)] transition-all duration-300`}
            />
          ))}
        </BentoGrid>
      </div>
    </div>
  );
}
