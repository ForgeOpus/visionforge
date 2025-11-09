import { GithubLogo, Book, Code } from "@phosphor-icons/react";
import { ThemeToggle } from "@/components/ThemeToggle";

export default function Footer() {
  return (
    <footer className="w-full bg-slate-950 border-t border-cyan-500/20 py-12 px-4">
      <div className="max-w-7xl mx-auto">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8 mb-8">
          {/* Brand */}
          <div className="md:col-span-1">
            <div className="flex items-center gap-3 mb-4">
              <img
                src="/vision_logo.png"
                alt="VisionForge"
                className="w-10 h-10 object-contain"
              />
              <span className="text-xl font-bold text-white">VisionForge</span>
            </div>
            <p className="text-gray-400 text-sm mb-4">
              Build AI models visually. Export production-ready code.
            </p>
            <div className="flex items-center gap-2">
              <div className="px-3 py-1 rounded bg-orange-500/10 border border-orange-500/30 text-orange-400 text-xs font-semibold">
                PyTorch
              </div>
              <div className="px-3 py-1 rounded bg-orange-500/10 border border-orange-500/30 text-orange-400 text-xs font-semibold">
                TensorFlow
              </div>
            </div>
          </div>

          {/* Product */}
          <div>
            <h3 className="text-white font-semibold mb-4">Product</h3>
            <ul className="space-y-2 text-sm">
              <li>
                <a href="/project" className="text-gray-400 hover:text-cyan-400 transition-colors">
                  Model Builder
                </a>
              </li>
              <li>
                <a href="#features" className="text-gray-400 hover:text-cyan-400 transition-colors">
                  Features
                </a>
              </li>
              <li>
                <a href="#how-it-works" className="text-gray-400 hover:text-cyan-400 transition-colors">
                  How It Works
                </a>
              </li>
              <li>
                <a href="#" className="text-gray-400 hover:text-cyan-400 transition-colors">
                  Examples
                </a>
              </li>
            </ul>
          </div>

          {/* Resources */}
          <div>
            <h3 className="text-white font-semibold mb-4">Resources</h3>
            <ul className="space-y-2 text-sm">
              <li>
                <a href="#" className="text-gray-400 hover:text-cyan-400 transition-colors flex items-center gap-2">
                  <Book size={16} />
                  Documentation
                </a>
              </li>
              <li>
                <a href="#" className="text-gray-400 hover:text-cyan-400 transition-colors flex items-center gap-2">
                  <Code size={16} />
                  API Reference
                </a>
              </li>
              <li>
                <a href="#" className="text-gray-400 hover:text-cyan-400 transition-colors flex items-center gap-2">
                  <GithubLogo size={16} />
                  GitHub
                </a>
              </li>
              <li>
                <a href="#" className="text-gray-400 hover:text-cyan-400 transition-colors">
                  Community
                </a>
              </li>
            </ul>
          </div>

          {/* Company */}
          <div>
            <h3 className="text-white font-semibold mb-4">Company</h3>
            <ul className="space-y-2 text-sm">
              <li>
                <a href="#" className="text-gray-400 hover:text-cyan-400 transition-colors">
                  About
                </a>
              </li>
              <li>
                <a href="#" className="text-gray-400 hover:text-cyan-400 transition-colors">
                  Blog
                </a>
              </li>
              <li>
                <a href="#" className="text-gray-400 hover:text-cyan-400 transition-colors">
                  Privacy Policy
                </a>
              </li>
              <li>
                <a href="#" className="text-gray-400 hover:text-cyan-400 transition-colors">
                  Terms of Service
                </a>
              </li>
            </ul>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="pt-8 border-t border-cyan-500/10 flex flex-col md:flex-row justify-between items-center gap-4">
          <div className="text-gray-500 text-sm">
            © {new Date().getFullYear()} VisionForge. All rights reserved.
          </div>

          <div className="flex items-center gap-4">
            <ThemeToggle />
            <div className="flex items-center gap-4 text-gray-500">
              <a
                href="#"
                className="hover:text-cyan-400 transition-colors"
                aria-label="GitHub"
              >
                <GithubLogo size={20} />
              </a>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
}
