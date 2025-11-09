import { motion } from "framer-motion";
import { Cube, Link as LinkIcon, Sliders, Code } from "@phosphor-icons/react";

export default function HowItWorks() {
  const steps = [
    {
      number: "01",
      title: "Drag Blocks",
      description: "Choose from our library of neural network layers and drag them onto the canvas.",
      icon: <Cube size={48} weight="duotone" className="text-cyan-400" />,
      color: "from-cyan-500/20 to-cyan-600/20",
    },
    {
      number: "02",
      title: "Connect Layers",
      description: "Draw connections between blocks. Shape validation happens automatically in real-time.",
      icon: <LinkIcon size={48} weight="duotone" className="text-cyan-400" />,
      color: "from-blue-500/20 to-blue-600/20",
    },
    {
      number: "03",
      title: "Configure",
      description: "Adjust layer parameters, activation functions, and hyperparameters in the side panel.",
      icon: <Sliders size={48} weight="duotone" className="text-cyan-400" />,
      color: "from-purple-500/20 to-purple-600/20",
    },
    {
      number: "04",
      title: "Export Code",
      description: "Generate production-ready PyTorch or TensorFlow code ready for training and deployment.",
      icon: <Code size={48} weight="duotone" className="text-cyan-400" />,
      color: "from-green-500/20 to-green-600/20",
    },
  ];

  return (
    <div className="w-full bg-gradient-to-b from-slate-950 to-slate-900 py-20 px-4">
      <div className="max-w-7xl mx-auto">
        {/* Section Header */}
        <div className="text-center mb-20">
          <h2 className="text-4xl md:text-5xl font-bold text-white mb-4">
            How It <span className="text-cyan-400">Works</span>
          </h2>
          <p className="text-xl text-gray-400 max-w-2xl mx-auto">
            Build production-ready AI models in four simple steps
          </p>
        </div>

        {/* Steps Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8 relative">
          {/* Connection Lines (Desktop) */}
          <div className="hidden lg:block absolute top-24 left-0 w-full h-0.5">
            <div className="relative w-full h-full">
              <motion.div
                className="absolute top-0 left-0 h-full bg-gradient-to-r from-transparent via-cyan-500/50 to-transparent"
                initial={{ width: "0%" }}
                whileInView={{ width: "100%" }}
                transition={{ duration: 2, delay: 0.5 }}
                viewport={{ once: true }}
              />
            </div>
          </div>

          {steps.map((step, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.2 }}
              viewport={{ once: true }}
              className="relative"
            >
              {/* Card */}
              <div className="relative bg-slate-900/50 border border-cyan-500/20 rounded-2xl p-6 hover:border-cyan-500/40 hover:shadow-[0_0_30px_rgba(0,229,255,0.15)] transition-all duration-300">
                {/* Step Number */}
                <div className="absolute -top-4 -right-4 w-12 h-12 rounded-full bg-gradient-to-br from-cyan-500 to-cyan-600 flex items-center justify-center font-bold text-white text-lg shadow-lg shadow-cyan-500/50">
                  {step.number}
                </div>

                {/* Icon Container */}
                <div className={`w-20 h-20 rounded-xl bg-gradient-to-br ${step.color} border border-cyan-500/30 flex items-center justify-center mb-6 mx-auto`}>
                  {step.icon}
                </div>

                {/* Content */}
                <h3 className="text-2xl font-bold text-white mb-3 text-center">
                  {step.title}
                </h3>
                <p className="text-gray-400 text-center leading-relaxed">
                  {step.description}
                </p>

                {/* Decorative Dots */}
                <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 flex gap-1">
                  {[...Array(3)].map((_, i) => (
                    <div
                      key={i}
                      className="w-1 h-1 rounded-full bg-cyan-500/30"
                    />
                  ))}
                </div>
              </div>

              {/* Arrow (Mobile/Tablet) */}
              {index < steps.length - 1 && (
                <div className="lg:hidden flex justify-center my-6">
                  <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: index * 0.2 + 0.3 }}
                    viewport={{ once: true }}
                  >
                    <svg
                      width="24"
                      height="24"
                      viewBox="0 0 24 24"
                      fill="none"
                      className="text-cyan-500/50"
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
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          transition={{ duration: 0.8, delay: 1 }}
          viewport={{ once: true }}
          className="text-center mt-16"
        >
          <p className="text-gray-400 text-lg">
            Ready to get started?{" "}
            <a
              href="/app"
              className="text-cyan-400 hover:text-cyan-300 font-semibold underline decoration-cyan-500/50 hover:decoration-cyan-400 transition-colors"
            >
              Try it now
            </a>
          </p>
        </motion.div>
      </div>
    </div>
  );
}
