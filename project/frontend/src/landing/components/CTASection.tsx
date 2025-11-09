import { motion } from "framer-motion";
import { LampContainer } from "./aceternity/LampEffect";
import { Button } from "@/components/ui/button";
import { ArrowRight, Upload } from "@phosphor-icons/react";
import { useNavigate } from "react-router-dom";

export default function CTASection() {
  const navigate = useNavigate();

  return (
    <LampContainer className="bg-slate-950">
      <motion.h1
        initial={{ opacity: 0, y: 100 }}
        whileInView={{ opacity: 1, y: 0 }}
        transition={{
          delay: 0.3,
          duration: 0.8,
          ease: "easeInOut",
        }}
        className="mt-8 bg-gradient-to-br from-slate-300 to-slate-500 py-4 bg-clip-text text-center text-4xl font-medium tracking-tight text-transparent md:text-7xl"
      >
        Ready to Build Your <br /> First AI Model?
      </motion.h1>

      <motion.div
        initial={{ opacity: 0, y: 50 }}
        whileInView={{ opacity: 1, y: 0 }}
        transition={{
          delay: 0.6,
          duration: 0.8,
          ease: "easeInOut",
        }}
        className="mt-12 flex flex-col sm:flex-row gap-4 justify-center items-center z-50"
      >
        <Button
          size="lg"
          onClick={() => navigate("/project")}
          className="bg-gradient-to-r from-cyan-500 to-cyan-600 hover:from-cyan-600 hover:to-cyan-700 text-white border-0 px-8 py-6 text-lg font-semibold shadow-[0_0_30px_rgba(0,229,255,0.4)] hover:shadow-[0_0_50px_rgba(0,229,255,0.6)] transition-all duration-300"
        >
          Start Building Now
          <ArrowRight size={24} className="ml-2" weight="bold" />
        </Button>

        <Button
          size="lg"
          variant="outline"
          className="border-cyan-500/50 text-cyan-400 bg-slate-900/50 hover:bg-cyan-500/10 hover:border-cyan-400 px-8 py-6 text-lg font-semibold transition-all duration-300"
        >
          <Upload size={24} className="mr-2" weight="bold" />
          Import Project
        </Button>
      </motion.div>

      <motion.p
        initial={{ opacity: 0 }}
        whileInView={{ opacity: 1 }}
        transition={{
          delay: 0.9,
          duration: 0.8,
        }}
        className="mt-8 text-center text-gray-400 text-sm z-50"
      >
        No credit card required • Free forever • Export unlimited projects
      </motion.p>
    </LampContainer>
  );
}
