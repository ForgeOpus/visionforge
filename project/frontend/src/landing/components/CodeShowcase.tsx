import { useState } from "react";
import { motion } from "framer-motion";
import { Copy, Check } from "@phosphor-icons/react";
import { Button } from "@/components/ui/button";

export default function CodeShowcase() {
  const [copied, setCopied] = useState(false);
  const [activeTab, setActiveTab] = useState<"model" | "train" | "config">("model");

  const codeExamples = {
    model: `import torch
import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 56 * 56, 512)
        self.relu3 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # Input: [batch, 3, 224, 224]
        x = self.pool1(self.relu1(self.conv1(x)))  # [batch, 64, 112, 112]
        x = self.pool2(self.relu2(self.conv2(x)))  # [batch, 128, 56, 56]
        x = self.flatten(x)  # [batch, 128*56*56]
        x = self.dropout(self.relu3(self.fc1(x)))  # [batch, 512]
        x = self.fc2(x)  # [batch, 10]
        return x`,
    train: `import torch
import torch.optim as optim
from model import CustomModel

# Initialize model
model = CustomModel()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train(model, dataloader, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(dataloader):.4f}')`,
    config: `# Model Configuration
MODEL_CONFIG = {
    'input_shape': (3, 224, 224),
    'num_classes': 10,
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 50,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# Training Configuration
TRAIN_CONFIG = {
    'optimizer': 'Adam',
    'loss_function': 'CrossEntropyLoss',
    'metrics': ['accuracy', 'precision', 'recall'],
    'early_stopping': True,
    'patience': 5
}

# Data Augmentation
DATA_AUG_CONFIG = {
    'random_flip': True,
    'random_rotation': 15,
    'normalization': {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    }
}`
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(codeExamples[activeTab]);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="w-full bg-slate-950 py-20 px-4">
      <div className="max-w-7xl mx-auto">
        {/* Section Header */}
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold text-white mb-4">
            From Visual to <span className="text-cyan-400">Production Code</span>
          </h2>
          <p className="text-xl text-gray-400 max-w-2xl mx-auto">
            Generate clean, documented, production-ready PyTorch or TensorFlow code instantly
          </p>
        </div>

        {/* Code Display */}
        <div className="max-w-5xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
            className="bg-slate-900/80 border border-cyan-500/30 rounded-2xl overflow-hidden shadow-[0_0_50px_rgba(0,229,255,0.1)]"
          >
            {/* Tabs */}
            <div className="border-b border-cyan-500/20 bg-slate-900/60 px-6 py-4 flex items-center justify-between">
              <div className="flex gap-2">
                <button
                  onClick={() => setActiveTab("model")}
                  className={`px-4 py-2 rounded-lg font-mono text-sm transition-all duration-200 ${
                    activeTab === "model"
                      ? "bg-cyan-500/20 text-cyan-400 border border-cyan-500/40"
                      : "text-gray-500 hover:text-gray-300"
                  }`}
                >
                  model.py
                </button>
                <button
                  onClick={() => setActiveTab("train")}
                  className={`px-4 py-2 rounded-lg font-mono text-sm transition-all duration-200 ${
                    activeTab === "train"
                      ? "bg-cyan-500/20 text-cyan-400 border border-cyan-500/40"
                      : "text-gray-500 hover:text-gray-300"
                  }`}
                >
                  train.py
                </button>
                <button
                  onClick={() => setActiveTab("config")}
                  className={`px-4 py-2 rounded-lg font-mono text-sm transition-all duration-200 ${
                    activeTab === "config"
                      ? "bg-cyan-500/20 text-cyan-400 border border-cyan-500/40"
                      : "text-gray-500 hover:text-gray-300"
                  }`}
                >
                  config.py
                </button>
              </div>

              <Button
                variant="ghost"
                size="sm"
                onClick={handleCopy}
                className="text-gray-400 hover:text-cyan-400 hover:bg-cyan-500/10"
              >
                {copied ? (
                  <>
                    <Check size={16} className="mr-2" weight="bold" />
                    Copied!
                  </>
                ) : (
                  <>
                    <Copy size={16} className="mr-2" />
                    Copy
                  </>
                )}
              </Button>
            </div>

            {/* Code Content */}
            <div className="p-6 overflow-x-auto">
              <pre className="font-mono text-sm leading-relaxed">
                <code className="text-gray-300">
                  {codeExamples[activeTab].split('\n').map((line, i) => (
                    <div key={i} className="hover:bg-cyan-500/5 px-2 -mx-2 rounded">
                      <span className="text-gray-600 select-none inline-block w-8 text-right mr-4">
                        {i + 1}
                      </span>
                      <span
                        dangerouslySetInnerHTML={{
                          __html: line
                            .replace(/(import|from|class|def|return|if|for|in|else)/g, '<span class="text-purple-400">$1</span>')
                            .replace(/(torch|nn|optim|device|model|criterion|optimizer)/g, '<span class="text-cyan-400">$1</span>')
                            .replace(/(['"].*?['"])/g, '<span class="text-green-400">$1</span>')
                            .replace(/(#.*$)/g, '<span class="text-gray-500">$1</span>')
                            .replace(/(\d+\.?\d*)/g, '<span class="text-orange-400">$1</span>')
                        }}
                      />
                    </div>
                  ))}
                </code>
              </pre>
            </div>

            {/* Footer */}
            <div className="border-t border-cyan-500/20 bg-slate-900/60 px-6 py-3 flex items-center justify-between text-xs">
              <div className="flex items-center gap-4">
                <span className="text-gray-500">Framework: <span className="text-cyan-400 font-semibold">PyTorch</span></span>
                <span className="text-gray-500">Lines: <span className="text-cyan-400 font-semibold">{codeExamples[activeTab].split('\n').length}</span></span>
              </div>
              <div className="flex gap-2">
                <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                <span className="text-gray-500">Ready to use</span>
              </div>
            </div>
          </motion.div>

          {/* Features below code */}
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            transition={{ duration: 0.6, delay: 0.3 }}
            viewport={{ once: true }}
            className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-8"
          >
            {[
              { title: "Type Hints", desc: "Full Python type annotations" },
              { title: "Documentation", desc: "Comprehensive docstrings" },
              { title: "Best Practices", desc: "Industry-standard patterns" }
            ].map((feature, i) => (
              <div key={i} className="bg-slate-900/40 border border-cyan-500/20 rounded-lg p-4 text-center">
                <div className="text-sm font-semibold text-white mb-1">{feature.title}</div>
                <div className="text-xs text-gray-500">{feature.desc}</div>
              </div>
            ))}
          </motion.div>
        </div>
      </div>
    </div>
  );
}
