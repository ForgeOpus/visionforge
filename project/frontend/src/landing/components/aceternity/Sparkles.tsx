"use client";
import React, { useId, useEffect, useState } from "react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

interface SparklesProps {
  id?: string;
  background?: string;
  minSize?: number;
  maxSize?: number;
  particleDensity?: number;
  className?: string;
  particleColor?: string;
}

export const SparklesCore = (props: SparklesProps) => {
  const {
    id,
    className,
    background = "transparent",
    minSize = 1,
    maxSize = 3,
    particleDensity = 100,
    particleColor = "#00E5FF",
  } = props;
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });

  useEffect(() => {
    const updateDimensions = () => {
      const element = document.getElementById(id || "sparkles");
      if (element) {
        setDimensions({ width: element.offsetWidth, height: element.offsetHeight });
      }
    };
    updateDimensions();
    window.addEventListener("resize", updateDimensions);
    return () => window.removeEventListener("resize", updateDimensions);
  }, [id]);

  const particles = Array.from({ length: particleDensity }).map((_, i) => ({
    id: i,
    x: Math.random() * dimensions.width,
    y: Math.random() * dimensions.height,
    size: Math.random() * (maxSize - minSize) + minSize,
    duration: Math.random() * 3 + 2,
  }));

  return (
    <div
      id={id || "sparkles"}
      className={cn("absolute inset-0", className)}
      style={{ background }}
    >
      {particles.map((particle) => (
        <motion.div
          key={particle.id}
          className="absolute rounded-full"
          style={{
            left: particle.x,
            top: particle.y,
            width: particle.size,
            height: particle.size,
            backgroundColor: particleColor,
            filter: "blur(1px)",
          }}
          initial={{ opacity: 0, scale: 0 }}
          animate={{
            opacity: [0, 1, 0],
            scale: [0, 1, 0],
          }}
          transition={{
            duration: particle.duration,
            repeat: Infinity,
            repeatType: "loop",
            ease: "easeInOut",
          }}
        />
      ))}
    </div>
  );
};
