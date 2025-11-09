import { motion } from "framer-motion";
import { Cube, Link as LinkIcon, Sliders, Code } from "@phosphor-icons/react";

export default function HowItWorks() {
  const steps = [
    {
      number: "01",
      title: "Drag & Drop Layers",
      description: "Browse our library of neural network building blocks. Simply drag layers onto the canvas to start designing your architecture.",
      icon: <Cube size={52} weight="duotone" className="text-cyan-400" />,
      gradient: "from-cyan-500/20 via-cyan-600/10 to-transparent",
      iconBg: "from-cyan-500/30 to-cyan-600/30",
      borderColor: "border-cyan-500/40",
    },
    {
      number: "02",
      title: "Connect & Validate",
      description: "Draw connections between blocks with automatic shape validation. Our engine checks compatibility and infers dimensions in real-time.",
      icon: <LinkIcon size={52} weight="duotone" className="text-blue-400" />,
      gradient: "from-blue-500/20 via-blue-600/10 to-transparent",
      iconBg: "from-blue-500/30 to-blue-600/30",
      borderColor: "border-blue-500/40",
    },
    {
      number: "03",
      title: "Configure Parameters",
      description: "Fine-tune layer settings, activation functions, and hyperparameters through our intuitive configuration panel.",
      icon: <Sliders size={52} weight="duotone" className="text-purple-400" />,
      gradient: "from-purple-500/20 via-purple-600/10 to-transparent",
      iconBg: "from-purple-500/30 to-purple-600/30",
      borderColor: "border-purple-500/40",
    },
    {
      number: "04",
      title: "Export Production Code",
      description: "Generate clean, documented PyTorch or TensorFlow code with type hints and best practices—ready for training and deployment.",
      icon: <Code size={52} weight="duotone" className="text-green-400" />,
      gradient: "from-green-500/20 via-green-600/10 to-transparent",
      iconBg: "from-green-500/30 to-green-600/30",
      borderColor: "border-green-500/40",
    },
  ];

  return (
    <div className="w-full bg-gradient-to-b from-slate-950 via-slate-900 to-slate-950 py-28 px-4 relative overflow-hidden">
      {/* Background effects */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#0a1628_1px,transparent_1px),linear-gradient(to_bottom,#0a1628_1px,transparent_1px)] bg-[size:4rem_4rem] opacity-10"></div>
      <div className="absolute top-1/4 right-0 w-96 h-96 bg-cyan-500/5 rounded-full blur-3xl"></div>
      <div className="absolute bottom-1/4 left-0 w-96 h-96 bg-blue-500/5 rounded-full blur-3xl"></div>

      <div className="max-w-7xl mx-auto relative z-10">
        {/* Section Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          viewport={{ once: true }}
          className="text-center mb-24"
        >
          <div className="inline-block mb-4 px-4 py-1.5 rounded-full bg-cyan-500/10 border border-cyan-500/30 backdrop-blur-sm">
            <span className="text-cyan-400 text-sm font-semibold">How It Works</span>
          </div>
          <h2 className="text-4xl md:text-5xl lg:text-6xl font-extrabold text-white mb-6 leading-tight">
            <span className="block mb-2">From Idea to Deployment</span>
            <span className="block bg-gradient-to-r from-cyan-400 via-blue-300 to-purple-400 bg-clip-text text-transparent">
              In Four Simple Steps
            </span>
          </h2>
          <p className="text-xl text-gray-400 max-w-3xl mx-auto leading-relaxed">
            Build production-ready neural networks faster than ever before
          </p>
        </motion.div>

        {/* Steps Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8 relative">
          {/* Connection Lines (Desktop) */}
          <div className="hidden lg:block absolute top-32 left-0 w-full h-0.5 px-16">
            <div className="relative w-full h-full">
              <motion.div
                className="absolute top-0 left-0 h-full bg-gradient-to-r from-cyan-500/50 via-blue-500/50 to-green-500/50 rounded-full"
                initial={{ width: "0%" }}
                whileInView={{ width: "100%" }}
                transition={{ duration: 2, delay: 0.5, ease: "easeInOut" }}
                viewport={{ once: true }}
              />
              <div className="absolute top-0 left-0 w-full h-full bg-gradient-to-r from-cyan-500/30 via-blue-500/30 to-green-500/30 blur-sm"></div>
            </div>
          </div>

          {steps.map((step, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 40 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: index * 0.15 }}
              viewport={{ once: true }}
              className="relative"
            >
              {/* Card */}
              <div className="relative group h-full">
                {/* Glow effect */}
                <div className={`absolute -inset-0.5 bg-gradient-to-br ${step.gradient} rounded-2xl blur-xl opacity-0 group-hover:opacity-100 transition-all duration-500`}></div>

                {/* Main card */}
                <div className={`relative bg-gradient-to-br from-slate-900/90 via-slate-800/90 to-slate-900/90 border-2 ${step.borderColor} rounded-2xl p-8 h-full flex flex-col backdrop-blur-sm hover:border-opacity-80 transition-all duration-500`}>
                  {/* Step Number Badge */}
                  <div className="absolute -top-4 -right-4 w-14 h-14 rounded-full bg-gradient-to-br from-slate-900 to-slate-800 border-2 border-cyan-500/40 flex items-center justify-center backdrop-blur-sm shadow-lg">
                    <span className="text-cyan-400 font-black text-lg">{step.number}</span>
                    <div className="absolute inset-0 rounded-full bg-gradient-to-br from-cyan-500/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
                  </div>

                  {/* Icon Container */}
                  <motion.div
                    whileHover={{ scale: 1.05, rotate: 5 }}
                    transition={{ type: "spring", stiffness: 300 }}
                    className={`w-24 h-24 rounded-2xl bg-gradient-to-br ${step.iconBg} border-2 ${step.borderColor} flex items-center justify-center mb-6 mx-auto shadow-lg backdrop-blur-sm`}
                  >
                    {step.icon}
                  </motion.div>

                  {/* Content */}
                  <div className="flex-1 flex flex-col">
                    <h3 className="text-2xl font-bold text-white mb-4 text-center group-hover:text-cyan-300 transition-colors duration-300">
                      {step.title}
                    </h3>
                    <p className="text-gray-400 text-center leading-relaxed text-base">
                      {step.description}
                    </p>
                  </div>

                  {/* Decorative bottom dots */}
                  <div className="flex justify-center gap-1.5 mt-6 pt-4 border-t border-cyan-500/10">
                    {[...Array(3)].map((_, i) => (
                      <motion.div
                        key={i}
                        initial={{ scale: 0 }}
                        whileInView={{ scale: 1 }}
                        transition={{ delay: index * 0.15 + 0.1 * i }}
                        viewport={{ once: true }}
                        className="w-1.5 h-1.5 rounded-full bg-cyan-500/40"
                      />
                    ))}
                  </div>
                </div>
              </div>

              {/* Arrow (Mobile/Tablet) */}
              {index < steps.length - 1 && (
                <div className="lg:hidden flex justify-center my-8">
                  <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: index * 0.15 + 0.3 }}
                    viewport={{ once: true }}
                    className="flex flex-col items-center"
                  >
                    <div className="w-0.5 h-8 bg-gradient-to-b from-cyan-500/50 to-blue-500/50 rounded-full"></div>
                    <svg
                      width="24"
                      height="24"
                      viewBox="0 0 24 24"
                      fill="none"
                      className="text-cyan-500/60"
                    >
                      <path
                        d="M12 5v14m0 0l7-7m-7 7l-7-7"
                        stroke="currentColor"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />
                    </svg>
                  </motion.div>
                </div>
              )}
            </motion.div>
          ))}
        </div>

        {/* Bottom CTA */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.8 }}
          viewport={{ once: true }}
          className="text-center mt-20"
        >
          <div className="inline-flex items-center gap-3 px-8 py-4 rounded-full bg-slate-900/50 border border-cyan-500/30 backdrop-blur-sm">
            <div className="flex -space-x-2">
              <div className="w-8 h-8 rounded-full bg-gradient-to-br from-cyan-500 to-blue-500 border-2 border-slate-900 flex items-center justify-center text-xs font-bold text-white">1</div>
              <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-purple-500 border-2 border-slate-900 flex items-center justify-center text-xs font-bold text-white">2</div>
              <div className="w-8 h-8 rounded-full bg-gradient-to-br from-purple-500 to-pink-500 border-2 border-slate-900 flex items-center justify-center text-xs font-bold text-white">3</div>
              <div className="w-8 h-8 rounded-full bg-gradient-to-br from-pink-500 to-green-500 border-2 border-slate-900 flex items-center justify-center text-xs font-bold text-white">4</div>
            </div>
            <span className="text-gray-400 text-base">
              Ready to get started?{" "}
              <a
                href="/project"
                className="text-cyan-400 hover:text-cyan-300 font-bold underline decoration-cyan-500/50 hover:decoration-cyan-400 transition-all"
              >
                Try it now →
              </a>
            </span>
          </div>
        </motion.div>
      </div>
    </div>
  );
}
