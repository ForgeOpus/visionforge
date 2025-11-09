import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Copy, Check, FileCode, FilePy, FileText } from "@phosphor-icons/react";
import { Button } from "@/components/ui/button";

export default function CodeShowcase() {
  const [copied, setCopied] = useState(false);
  const [activeTab, setActiveTab] = useState<"model" | "train" | "config">("model");

  const codeExamples = {
    model: `import torch
import torch.nn as nn
from typing import Tuple

class VisionModel(nn.Module):
    """
    Custom vision model built with VisionForge
    Input: [B, 3, 224, 224]
    Output: [B, 1000]
    """
    def __init__(self, num_classes: int = 1000):
        super().__init__()

        # Feature extraction layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual blocks
        self.layer1 = self._make_layer(64, 128, blocks=2)
        self.layer2 = self._make_layer(128, 256, blocks=2)

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, in_channels: int, out_channels: int, blocks: int) -> nn.Sequential:
        layers = []
        for _ in range(blocks):
            layers.append(nn.Conv2d(in_channels, out_channels, 3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stem
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))

        # Feature extraction
        x = self.layer1(x)
        x = self.layer2(x)

        # Classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x`,
    train: `import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import VisionModel

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    device: str = 'cuda'
):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_acc = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_acc += (preds == labels).sum().item()

        val_acc /= len(val_loader.dataset)
        scheduler.step()

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')

        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Loss: {train_loss/len(train_loader):.4f}, Val Acc: {val_acc:.4f}')`,
    config: `from dataclasses import dataclass
import torch

@dataclass
class ModelConfig:
    """Configuration for model architecture and training"""

    # Architecture
    input_shape: tuple = (3, 224, 224)
    num_classes: int = 1000
    dropout_rate: float = 0.5

    # Training
    batch_size: int = 32
    num_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    # Hardware
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers: int = 4

    # Data augmentation
    use_augmentation: bool = True
    random_crop: bool = True
    random_flip: bool = True
    color_jitter: bool = True

    # Checkpointing
    save_every: int = 5
    checkpoint_dir: str = './checkpoints'

    # Normalization (ImageNet stats)
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)

# Usage
config = ModelConfig()
print(f"Training on: {config.device}")
print(f"Batch size: {config.batch_size}")
print(f"Learning rate: {config.learning_rate}")`
  };

  const tabIcons = {
    model: <FileCode size={18} weight="duotone" />,
    train: <FilePy size={18} weight="duotone" />,
    config: <FileText size={18} weight="duotone" />
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(codeExamples[activeTab]);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };


  return (
    <div className="w-full bg-slate-950 py-28 px-4 relative overflow-hidden">
      {/* Background effects */}
      <div className="absolute top-0 left-1/3 w-96 h-96 bg-purple-500/5 rounded-full blur-3xl"></div>
      <div className="absolute bottom-0 right-1/3 w-96 h-96 bg-cyan-500/5 rounded-full blur-3xl"></div>

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
            <span className="text-cyan-400 text-sm font-semibold">Code Generation</span>
          </div>
          <h2 className="text-4xl md:text-5xl lg:text-6xl font-extrabold text-white mb-6 leading-tight">
            <span className="block mb-2">From Visual Design to</span>
            <span className="block bg-gradient-to-r from-purple-400 via-cyan-300 to-blue-400 bg-clip-text text-transparent">
              Production-Ready Code
            </span>
          </h2>
          <p className="text-xl text-gray-400 max-w-3xl mx-auto leading-relaxed">
            Export clean, well-documented Python code with type hints,
            <span className="text-cyan-400 font-semibold"> best practices built-in</span>
          </p>
        </motion.div>

        {/* Code Display */}
        <div className="max-w-6xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
            className="relative group"
          >
            {/* Glow effect */}
            <div className="absolute -inset-1 bg-gradient-to-r from-purple-500/20 via-cyan-500/20 to-blue-500/20 rounded-3xl blur-2xl opacity-50 group-hover:opacity-100 transition-all duration-500"></div>

            {/* Code container */}
            <div className="relative bg-gradient-to-br from-slate-900/95 via-slate-800/95 to-slate-900/95 border-2 border-cyan-500/30 rounded-2xl overflow-hidden shadow-[0_0_60px_rgba(0,229,255,0.15)] backdrop-blur-sm">
              {/* Tabs Header */}
              <div className="border-b border-cyan-500/20 bg-slate-900/80 backdrop-blur-sm px-6 py-4">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex gap-2">
                    {(['model', 'train', 'config'] as const).map((tab) => (
                      <button
                        key={tab}
                        onClick={() => setActiveTab(tab)}
                        className={`group/tab flex items-center gap-2 px-5 py-2.5 rounded-lg font-mono text-sm font-medium transition-all duration-300 ${
                          activeTab === tab
                            ? "bg-gradient-to-r from-cyan-500/20 to-blue-500/20 text-cyan-400 border-2 border-cyan-500/40 shadow-[0_0_20px_rgba(0,229,255,0.2)]"
                            : "text-gray-500 hover:text-gray-300 border-2 border-transparent hover:border-cyan-500/20 hover:bg-slate-800/50"
                        }`}
                      >
                        <span className={activeTab === tab ? "text-cyan-400" : "text-gray-500 group-hover/tab:text-gray-300"}>
                          {tabIcons[tab]}
                        </span>
                        {tab}.py
                      </button>
                    ))}
                  </div>

                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={handleCopy}
                    className="text-gray-400 hover:text-cyan-400 hover:bg-cyan-500/10 border border-transparent hover:border-cyan-500/30 transition-all duration-300"
                  >
                    <AnimatePresence mode="wait">
                      {copied ? (
                        <motion.div
                          key="check"
                          initial={{ scale: 0 }}
                          animate={{ scale: 1 }}
                          exit={{ scale: 0 }}
                          className="flex items-center gap-2"
                        >
                          <Check size={18} weight="bold" className="text-green-400" />
                          <span className="text-green-400 font-semibold">Copied!</span>
                        </motion.div>
                      ) : (
                        <motion.div
                          key="copy"
                          initial={{ scale: 0 }}
                          animate={{ scale: 1 }}
                          exit={{ scale: 0 }}
                          className="flex items-center gap-2"
                        >
                          <Copy size={18} />
                          <span>Copy Code</span>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </Button>
                </div>

                {/* File info */}
                <div className="flex items-center gap-4 text-xs text-gray-500">
                  <div className="flex items-center gap-1.5">
                    <div className="w-2 h-2 bg-green-400 rounded-full shadow-[0_0_8px_rgba(74,222,128,0.6)]"></div>
                    <span>Syntax Valid</span>
                  </div>
                  <div>•</div>
                  <span>Framework: <span className="text-cyan-400 font-semibold">PyTorch</span></span>
                  <div>•</div>
                  <span>{codeExamples[activeTab].split('\n').length} lines</span>
                </div>
              </div>

              {/* Code Content */}
              <div className="relative overflow-hidden">
                <div className="absolute inset-0 bg-[linear-gradient(to_right,#1e293b_1px,transparent_1px),linear-gradient(to_bottom,#1e293b_1px,transparent_1px)] bg-[size:2rem_2rem] opacity-5 pointer-events-none"></div>

                <AnimatePresence mode="wait">
                  <motion.div
                    key={activeTab}
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: -20 }}
                    transition={{ duration: 0.3 }}
                    className="p-8 overflow-x-auto overflow-y-auto custom-scrollbar"
                    style={{ maxHeight: '500px' }}
                  >
                    <pre className="font-mono text-sm leading-relaxed min-w-max">
                      <code className="text-gray-300">
                        {codeExamples[activeTab].split('\n').map((line, i) => (
                          <div key={i} className="hover:bg-cyan-500/5 px-3 -mx-3 rounded transition-colors whitespace-pre">
                            <span className="text-gray-600 select-none inline-block w-10 text-right mr-6 font-medium">
                              {i + 1}
                            </span>
                            <span className="text-cyan-300">
                              {line || '\u00A0'}
                            </span>
                          </div>
                        ))}
                      </code>
                    </pre>
                  </motion.div>
                </AnimatePresence>
              </div>
            </div>
          </motion.div>

          {/* Features below code */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.3 }}
            viewport={{ once: true }}
            className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-12"
          >
            {[
              { title: "Type Hints", desc: "Full Python type annotations", icon: "📝" },
              { title: "Documentation", desc: "Comprehensive docstrings", icon: "📚" },
              { title: "Best Practices", desc: "Industry-standard patterns", icon: "⭐" }
            ].map((feature, i) => (
              <motion.div
                key={i}
                whileHover={{ scale: 1.05, y: -5 }}
                className="bg-slate-900/40 backdrop-blur-sm border-2 border-cyan-500/20 rounded-xl p-6 text-center hover:border-cyan-500/40 hover:shadow-[0_0_30px_rgba(0,229,255,0.15)] transition-all duration-300"
              >
                <div className="text-4xl mb-3">{feature.icon}</div>
                <div className="text-base font-bold text-white mb-2">{feature.title}</div>
                <div className="text-sm text-gray-500">{feature.desc}</div>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </div>

      <style>{`
        .custom-scrollbar::-webkit-scrollbar {
          width: 10px;
          height: 10px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: rgba(15, 23, 42, 0.5);
          border-radius: 6px;
          margin: 8px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(0, 229, 255, 0.4);
          border-radius: 6px;
          border: 2px solid rgba(15, 23, 42, 0.5);
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: rgba(0, 229, 255, 0.6);
        }
        .custom-scrollbar::-webkit-scrollbar-corner {
          background: rgba(15, 23, 42, 0.5);
        }

        /* Firefox */
        .custom-scrollbar {
          scrollbar-width: thin;
          scrollbar-color: rgba(0, 229, 255, 0.4) rgba(15, 23, 42, 0.5);
        }
      `}</style>
    </div>
  );
}
