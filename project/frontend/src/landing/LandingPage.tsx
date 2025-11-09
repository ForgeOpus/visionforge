import { FloatingNav } from "./components/aceternity/FloatingNav";
import Hero from "./components/Hero";
import Features from "./components/Features";
import HowItWorks from "./components/HowItWorks";
import CodeShowcase from "./components/CodeShowcase";
import CTASection from "./components/CTASection";
import Footer from "./components/Footer";
import { Cube, Lightning, Code, Book } from "@phosphor-icons/react";

export default function LandingPage() {
  const navItems = [
    {
      name: "Features",
      link: "#features",
      icon: <Lightning size={16} />,
    },
    {
      name: "How It Works",
      link: "#how-it-works",
      icon: <Cube size={16} />,
    },
    {
      name: "Code",
      link: "#code",
      icon: <Code size={16} />,
    },
    {
      name: "Docs",
      link: "#docs",
      icon: <Book size={16} />,
    },
  ];

  return (
    <div className="relative w-full bg-slate-950">
      {/* Floating Navigation */}
      <FloatingNav navItems={navItems} />

      {/* Hero Section */}
      <section id="hero">
        <Hero />
      </section>

      {/* Features Section */}
      <section id="features">
        <Features />
      </section>

      {/* How It Works Section */}
      <section id="how-it-works">
        <HowItWorks />
      </section>

      {/* Code Showcase Section */}
      <section id="code">
        <CodeShowcase />
      </section>

      {/* CTA Section */}
      <section id="cta">
        <CTASection />
      </section>

      {/* Footer */}
      <Footer />
    </div>
  );
}
